"""Dataset generation service — create, list, and manage evaluation datasets."""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Generator

import openai

from config.settings import settings
from src.api.dependencies import DATASETS_DIR
from src.api.schemas import DatasetCategoryConfig, DatasetCreateRequest
from src.utils.logger import get_logger
from src.vector_store.qdrant_manager import QdrantManager

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Chunk relevance heuristics per question type
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"\b(1[89]\d{2}|20[0-4]\d)\b")
_MONTH_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\b",
    re.IGNORECASE,
)
_TEMPORAL_KW = re.compile(
    r"\b(before|after|during|following|prior to|until|since|meanwhile)\b",
    re.IGNORECASE,
)
_COMPARISON_KW = re.compile(
    r"\b(compared to|unlike|whereas|while|however|in contrast|on the other hand|"
    r"similar to|difference|similarities)\b",
    re.IGNORECASE,
)
_CAUSAL_KW = re.compile(
    r"\b(because|resulted in|led to|caused|consequence|therefore|thus|due to|"
    r"impact|effect|influence)\b",
    re.IGNORECASE,
)
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b")


def _chunk_relevant_for_type(text: str, question_type: str) -> bool:
    """Lightweight heuristic to check if a chunk is suitable for a question type."""
    if question_type == "factual":
        return len(text) >= 200
    if question_type == "temporal":
        return bool(_DATE_RE.search(text) or _MONTH_RE.search(text) or _TEMPORAL_KW.search(text))
    if question_type == "comparative":
        entities = _PROPER_NOUN_RE.findall(text)
        return len(entities) >= 2 or bool(_COMPARISON_KW.search(text))
    if question_type == "entity_centric":
        return len(_PROPER_NOUN_RE.findall(text)) >= 1
    if question_type == "relationship":
        entities = _PROPER_NOUN_RE.findall(text)
        return len(entities) >= 2
    if question_type == "analytical":
        return len(text) >= 300 and bool(_CAUSAL_KW.search(text))
    return len(text) >= 200


# ---------------------------------------------------------------------------
# Default prompts per question type
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS: dict[str, str] = {
    "factual": (
        "Generate factual recall questions about specific events, dates, names, "
        "and places mentioned in the passage."
    ),
    "temporal": (
        "Generate questions about chronological order, time periods, "
        "before/after relationships, and sequences of events."
    ),
    "comparative": (
        "Generate questions that compare or contrast events, strategies, "
        "figures, or outcomes described in the passage."
    ),
    "entity_centric": (
        "Generate questions focused on the roles, actions, and significance "
        "of specific people, organizations, or places."
    ),
    "relationship": (
        "Generate questions about causal links, alliances, conflicts, "
        "and connections between entities or events."
    ),
    "analytical": (
        "Generate questions requiring analysis, synthesis, or evaluation "
        "of impacts, consequences, and broader significance."
    ),
}


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DatasetService:
    """Manages evaluation dataset lifecycle: create, list, get, delete."""

    def __init__(self) -> None:
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # ── List / Get / Delete ───────────────────────────────────────────

    def list_datasets(self) -> list[dict]:
        datasets: list[dict] = []
        for fp in sorted(DATASETS_DIR.glob("*.json"), reverse=True):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                datasets.append({
                    "id": data["id"],
                    "name": data["name"],
                    "created_at": data["created_at"],
                    "status": data["status"],
                    "collection_name": data["collection_name"],
                    "total_questions": data.get("total_questions", 0),
                    "categories": data.get("categories", []),
                })
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(f"Skipping invalid dataset file {fp}: {exc}")
        return datasets

    def get_dataset(self, dataset_id: str) -> dict | None:
        fp = DATASETS_DIR / f"{dataset_id}.json"
        if not fp.exists():
            return None
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)

    def delete_dataset(self, dataset_id: str) -> bool:
        fp = DATASETS_DIR / f"{dataset_id}.json"
        if not fp.exists():
            return False
        fp.unlink()
        return True

    # ── Generate (SSE streaming) ─────────────────────────────────────

    def generate_dataset(
        self, request: DatasetCreateRequest
    ) -> Generator[str, None, None]:
        """Generate questions per category, yielding SSE events."""
        dataset_id = f"ds_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        now_iso = datetime.now().isoformat()

        # Build initial dataset structure
        categories_state = []
        for cat in request.categories:
            categories_state.append({
                "type": cat.type,
                "prompt": cat.prompt,
                "model": cat.model,
                "count": cat.count,
                "generated": 0,
            })

        dataset: dict = {
            "id": dataset_id,
            "name": request.name,
            "created_at": now_iso,
            "status": "generating",
            "collection_name": request.collection_name,
            "total_questions": sum(c.count for c in request.categories),
            "categories": categories_state,
            "questions": [],
            "metadata": {
                "total_generated": 0,
                "unique_articles": 0,
                "generation_time_s": 0.0,
            },
        }
        self._save(dataset)
        yield self._sse("started", {"dataset_id": dataset_id})

        # Load chunks from collection
        try:
            chunks = self._load_chunks(request.collection_name)
        except Exception as exc:
            dataset["status"] = "failed"
            self._save(dataset)
            yield self._sse("error", {"message": f"Failed to load chunks: {exc}"})
            return

        if not chunks:
            dataset["status"] = "failed"
            self._save(dataset)
            yield self._sse("error", {"message": "No chunks found in collection."})
            return

        # Init OpenAI client (OpenRouter)
        client = openai.OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
        )

        start_time = time.time()
        question_counter = 0
        all_questions: list[dict] = []

        import random
        random.shuffle(chunks)

        for cat_idx, cat in enumerate(request.categories):
            cat_generated = 0
            # Pool: filter relevant chunks for this type
            pool = [c for c in chunks if _chunk_relevant_for_type(c["content"], cat.type)]
            if not pool:
                pool = [c for c in chunks if len(c["content"]) >= 200]
            random.shuffle(pool)
            pool_iter = iter(pool)

            for _ in range(cat.count):
                MAX_RETRIES = 5
                retries = 0
                success = False

                while retries < MAX_RETRIES:
                    chunk = next(pool_iter, None)
                    if chunk is None:
                        random.shuffle(pool)
                        pool_iter = iter(pool)
                        chunk = next(pool_iter, None)
                        if chunk is None:
                            break

                    try:
                        questions = self._generate_for_chunk(
                            client, chunk, cat, question_counter
                        )
                    except Exception as exc:
                        retries += 1
                        logger.warning("LLM error (%d/%d) for %s: %s", retries, MAX_RETRIES, cat.type, exc)
                        yield self._sse("retry", {
                            "category": cat.type,
                            "attempt": retries,
                            "max_retries": MAX_RETRIES,
                            "message": str(exc),
                        })
                        time.sleep(2)
                        continue

                    if questions is None:
                        # LLM said "next" — chunk not relevant, don't count as retry
                        logger.info("next — LLM deemed chunk not relevant for %s", cat.type)
                        yield self._sse("skip", {"category": cat.type, "reason": "chunk_not_relevant"})
                        continue

                    if not questions:
                        retries += 1
                        continue

                    # Success
                    success = True
                    for q in questions:
                        question_counter += 1
                        q["id"] = f"gen_q{question_counter:03d}"
                        all_questions.append(q)
                        cat_generated += 1

                    categories_state[cat_idx]["generated"] = cat_generated
                    yield self._sse("progress", {
                        "category": cat.type,
                        "generated": cat_generated,
                        "total": cat.count,
                        "question_id": questions[0]["id"] if questions else None,
                    })

                    # Rate limiting
                    time.sleep(1.5)
                    break

                if not success:
                    logger.warning("Skipping question for %s after %d failed attempts", cat.type, retries)
                    yield self._sse("skip", {"category": cat.type, "reason": "max_retries"})

            yield self._sse("category_complete", {
                "category": cat.type,
                "generated": cat_generated,
                "total": cat.count,
            })

        elapsed = time.time() - start_time
        unique_articles = len(set(q.get("source_article", "") for q in all_questions))

        dataset["questions"] = all_questions
        dataset["status"] = "completed"
        dataset["metadata"] = {
            "total_generated": len(all_questions),
            "unique_articles": unique_articles,
            "generation_time_s": round(elapsed, 1),
        }
        dataset["total_questions"] = len(all_questions)
        self._save(dataset)

        yield self._sse("complete", {
            "dataset_id": dataset_id,
            "total_generated": len(all_questions),
        })

    # ── Private helpers ───────────────────────────────────────────────

    def _save(self, dataset: dict) -> None:
        fp = DATASETS_DIR / f"{dataset['id']}.json"
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

    def _load_chunks(self, collection_name: str) -> list[dict]:
        """Load all chunks from the given Qdrant collection."""
        mgr = QdrantManager()
        chunks: list[dict] = []
        offset = None

        while True:
            points, next_offset = mgr.client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in points:
                chunks.append({
                    "chunk_id": str(pt.id),
                    "content": pt.payload.get("content", ""),
                    "article_title": pt.payload.get("article_title", ""),
                    "article_id": str(pt.payload.get("pageid", "")),
                    "chunk_index": pt.payload.get("chunk_index", 0),
                })
            if next_offset is None:
                break
            offset = next_offset

        return chunks

    def _generate_for_chunk(
        self,
        client: openai.OpenAI,
        chunk: dict,
        cat: DatasetCategoryConfig,
        counter: int,
    ) -> list[dict] | None:
        """Call LLM to generate 1 question for a chunk.

        Returns parsed question list, or ``None`` when the LLM deems
        the chunk not relevant (responds with "next").
        """
        prompt = f"""{cat.prompt}

Source Article: {chunk['article_title']}
Text Passage:
{chunk['content'][:2000]}

If the passage is not relevant or suitable for generating a {cat.type} question, respond with exactly: next

Generate exactly 1 question as a JSON array. Target question type: {cat.type}
Output ONLY the JSON array, no other text:
[
  {{
    "question": "...",
    "type": "{cat.type}",
    "difficulty": "easy|medium|hard",
    "expected_answer_hint": "brief hint about what the answer should contain"
  }}
]"""

        response = client.chat.completions.create(
            model=cat.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert question generator for historical content evaluation.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=settings.QUESTION_GEN_TEMPERATURE,
            max_tokens=settings.QUESTION_GEN_MAX_TOKENS,
        )

        raw = response.choices[0].message.content
        if not raw:
            raise ValueError("LLM returned empty content")
        text = raw.strip()

        # LLM deemed chunk not relevant
        if text.lower() == "next":
            return None

        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        # Try parsing as-is first; if truncated, attempt to close the JSON
        try:
            questions = json.loads(text)
        except json.JSONDecodeError:
            # Try closing truncated JSON: add missing `}]`
            for suffix in ('"}]', "}]", "]"):
                try:
                    questions = json.loads(text + suffix)
                    break
                except json.JSONDecodeError:
                    continue
            else:
                raise
        for q in questions:
            q["source_chunk_id"] = chunk["chunk_id"]
            q["source_article"] = chunk["article_title"]
            q["source_article_id"] = chunk["article_id"]
            q["generated_at"] = datetime.now().isoformat()
            q["model"] = cat.model

        return questions

    @staticmethod
    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"
