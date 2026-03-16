"""
Automated question generation for evaluation using OpenRouter Mistral Small.

Generates diverse questions from document chunks with taxonomic variety:
- Factual (25%)
- Temporal (20%)
- Comparative (15%)
- Entity-Centric (15%)
- Relationship (15%)
- Analytical (10%)

Usage:
    from src.evaluation.question_generator import QuestionGenerator

    generator = QuestionGenerator()
    questions = generator.generate_evaluation_questions(num_chunks=30)
    generator.save_questions(questions, "data/evaluation/generated_questions.json")
"""

import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import openai
from tqdm import tqdm

from config.settings import settings
from src.utils.logger import get_logger
from src.vector_store.qdrant_manager import QdrantManager

logger = get_logger(__name__)


# Question generation prompt template for chunks
QUESTION_GENERATION_PROMPT = """You are an expert in World War II history. Given the following text passage from a historical document, generate {num_questions} diverse questions that can be answered using information from this passage.

Requirements:
- Questions must be answerable from the passage content
- Vary question types: factual, temporal, comparative, entity_centric, relationship, analytical
- Target question type for this round: {target_type}
- Include specific dates, names, places, and events from the passage
- Questions should test different difficulty levels
- Make questions specific to the content, not generic

Source Article: {article_title}
Text Passage:
{chunk_text}

Generate exactly {num_questions} questions as a JSON array. Output ONLY the JSON array, no other text:
[
  {{
    "question": "...",
    "type": "factual|temporal|comparative|entity_centric|relationship|analytical",
    "difficulty": "easy|medium|hard",
    "expected_answer_hint": "brief hint about what the answer should contain"
  }}
]
"""


# Question type taxonomy with target distribution
QUESTION_TAXONOMY = {
    "factual": 0.25,
    "temporal": 0.20,
    "comparative": 0.15,
    "entity_centric": 0.15,
    "relationship": 0.15,
    "analytical": 0.10,
}


class QuestionGenerator:
    """Generates evaluation questions from document chunks using LLM."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        qdrant_manager: Optional[QdrantManager] = None,
        skip_api_init: bool = False,
    ):
        """
        Initialize question generator.

        Args:
            api_key: OpenRouter API key (default: from settings)
            model: Model to use (default: from settings)
            base_url: API base URL (default: OpenRouter)
            qdrant_manager: QdrantManager instance for loading chunks
            skip_api_init: Skip API client initialization (for testing)
        """
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.model = model or settings.QUESTION_GEN_MODEL
        self.base_url = base_url or settings.OPENROUTER_BASE_URL
        self.qdrant = qdrant_manager
        self.client = None

        if not skip_api_init:
            if not self.api_key:
                raise ValueError(
                    "OpenRouter API key not set. Set OPENROUTER_API_KEY in .env file."
                )

            # Initialize OpenAI client with OpenRouter endpoint
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

            logger.info(
                f"QuestionGenerator initialized with model: {self.model}",
                extra={"model": self.model, "base_url": self.base_url},
            )

    def load_chunks_from_qdrant(
        self, collection_name: str, max_chunks: Optional[int] = None
    ) -> List[Dict]:
        """
        Load chunks from Qdrant collection.

        Args:
            collection_name: Qdrant collection name
            max_chunks: Maximum chunks to load (default: all)

        Returns:
            List of chunk dictionaries
        """
        if not self.qdrant:
            self.qdrant = QdrantManager()

        logger.info(f"Loading chunks from collection '{collection_name}'")

        # Get all points from collection
        # Qdrant doesn't have a "get all" method, so we use scroll
        chunks = []
        offset = None

        while True:
            result = self.qdrant.client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            points, next_offset = result

            for point in points:
                chunk = {
                    "chunk_id": point.id,
                    "content": point.payload.get("content", ""),
                    "article_title": point.payload.get("article_title", ""),
                    "article_id": point.payload.get("pageid", ""),
                    "chunk_index": point.payload.get("chunk_index", 0),
                    "metadata": point.payload,
                }
                chunks.append(chunk)

            if next_offset is None or (max_chunks and len(chunks) >= max_chunks):
                break

            offset = next_offset

        if max_chunks:
            chunks = chunks[:max_chunks]

        logger.info(f"Loaded {len(chunks)} chunks from Qdrant")
        return chunks

    def sample_chunks(
        self,
        chunks: List[Dict],
        num_samples: int,
        strategy: str = "stratified",
        min_length: int = 200,
    ) -> List[Dict]:
        """
        Sample chunks for question generation.

        Args:
            chunks: List of chunk dictionaries
            num_samples: Number of chunks to sample
            strategy: Sampling strategy ('random', 'stratified', 'diverse')
            min_length: Minimum chunk length in characters

        Returns:
            Sampled chunks
        """
        # Filter chunks by minimum length
        valid_chunks = [c for c in chunks if len(c["content"]) >= min_length]

        logger.info(
            f"Sampling {num_samples} chunks from {len(valid_chunks)} valid chunks "
            f"(min_length={min_length})"
        )

        if strategy == "random":
            sampled = random.sample(valid_chunks, min(num_samples, len(valid_chunks)))

        elif strategy == "stratified":
            # Stratify by article to get diverse coverage
            chunks_by_article = {}
            for chunk in valid_chunks:
                article = chunk["article_title"]
                if article not in chunks_by_article:
                    chunks_by_article[article] = []
                chunks_by_article[article].append(chunk)

            # Sample evenly across articles
            sampled = []
            articles = list(chunks_by_article.keys())
            random.shuffle(articles)

            while len(sampled) < num_samples and articles:
                for article in articles[:]:
                    if chunks_by_article[article]:
                        chunk = chunks_by_article[article].pop(
                            random.randint(0, len(chunks_by_article[article]) - 1)
                        )
                        sampled.append(chunk)

                        if len(sampled) >= num_samples:
                            break
                    else:
                        articles.remove(article)

        elif strategy == "diverse":
            # Sample chunks with diverse content length and article coverage
            # Group by article and length
            chunks_by_article = {}
            for chunk in valid_chunks:
                article = chunk["article_title"]
                if article not in chunks_by_article:
                    chunks_by_article[article] = []
                chunks_by_article[article].append(chunk)

            sampled = []
            for article, article_chunks in chunks_by_article.items():
                # Sort chunks by length
                article_chunks.sort(key=lambda c: len(c["content"]))

                # Sample short, medium, long chunks from each article
                if len(article_chunks) >= 3:
                    sampled.append(article_chunks[0])  # Shortest
                    sampled.append(article_chunks[len(article_chunks) // 2])  # Medium
                    sampled.append(article_chunks[-1])  # Longest
                else:
                    sampled.extend(article_chunks)

                if len(sampled) >= num_samples:
                    break

            random.shuffle(sampled)
            sampled = sampled[:num_samples]

        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        logger.info(
            f"Sampled {len(sampled)} chunks using '{strategy}' strategy from "
            f"{len(set(c['article_title'] for c in sampled))} unique articles"
        )

        return sampled

    def generate_question_for_chunk(
        self,
        chunk: Dict,
        num_questions: int,
        target_type: str,
    ) -> List[Dict]:
        """
        Generate questions for a single chunk using LLM.

        Args:
            chunk: Chunk dictionary
            num_questions: Number of questions to generate
            target_type: Target question type

        Returns:
            List of generated question dictionaries
        """
        # Prepare prompt
        prompt = QUESTION_GENERATION_PROMPT.format(
            num_questions=num_questions,
            target_type=target_type,
            article_title=chunk["article_title"],
            chunk_text=chunk["content"][:2000],  # Limit to 2000 chars
        )

        try:
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
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

            # Parse response
            response_text = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            questions = json.loads(response_text)

            # Validate and enrich questions
            for q in questions:
                q["source_chunk_id"] = chunk["chunk_id"]
                q["source_chunk_index"] = chunk.get("chunk_index", 0)
                q["source_article"] = chunk["article_title"]
                q["source_article_id"] = chunk["article_id"]
                q["generated_at"] = datetime.now().isoformat()
                q["model"] = self.model

            return questions

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse LLM response as JSON: {e}",
                extra={"response": response_text[:200]},
            )
            return []

        except Exception as e:
            logger.error(f"Failed to generate questions for chunk: {e}")
            return []

    def generate_evaluation_questions(
        self,
        collection_name: Optional[str] = None,
        num_chunks: int = 30,
        questions_per_chunk: int = 1,
        chunks: Optional[List[Dict]] = None,
        sampling_strategy: str = "stratified",
        ensure_taxonomy_diversity: bool = True,
    ) -> Dict:
        """
        Generate evaluation questions from chunks.

        Args:
            collection_name: Qdrant collection to load chunks from
            num_chunks: Number of chunks to sample
            questions_per_chunk: Questions to generate per chunk
            chunks: Pre-loaded chunks (if None, loads from Qdrant)
            sampling_strategy: Chunk sampling strategy
            ensure_taxonomy_diversity: Enforce taxonomy distribution

        Returns:
            Dictionary with metadata and questions
        """
        logger.info(
            f"Starting question generation: {num_chunks} chunks × "
            f"{questions_per_chunk} questions = {num_chunks * questions_per_chunk} total"
        )

        # Load chunks
        if chunks is None:
            collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
            chunks = self.load_chunks_from_qdrant(collection_name)

        # Sample chunks
        sampled_chunks = self.sample_chunks(
            chunks, num_chunks, strategy=sampling_strategy
        )

        # Determine question types to generate
        total_questions = num_chunks * questions_per_chunk
        question_types = []

        if ensure_taxonomy_diversity:
            # Distribute questions according to taxonomy
            for qtype, proportion in QUESTION_TAXONOMY.items():
                count = round(total_questions * proportion)
                question_types.extend([qtype] * count)

            # Fill any remaining with random types
            while len(question_types) < total_questions:
                question_types.append(random.choice(list(QUESTION_TAXONOMY.keys())))

            # Shuffle to randomize order
            random.shuffle(question_types)
        else:
            # Random question types
            question_types = [
                random.choice(list(QUESTION_TAXONOMY.keys()))
                for _ in range(total_questions)
            ]

        # Generate questions
        all_questions = []
        question_id_counter = 1

        for i, chunk in enumerate(tqdm(sampled_chunks, desc="Generating questions")):
            target_type = question_types[i] if i < len(question_types) else "factual"

            questions = self.generate_question_for_chunk(
                chunk,
                num_questions=questions_per_chunk,
                target_type=target_type,
            )

            for q in questions:
                q["id"] = f"gen_q{question_id_counter:03d}"
                question_id_counter += 1

            all_questions.extend(questions)

        # Count question types
        type_distribution = Counter(q["type"] for q in all_questions)

        logger.info(
            f"Generated {len(all_questions)} questions from {len(sampled_chunks)} chunks"
        )
        logger.info(f"Type distribution: {dict(type_distribution)}")

        # Build result
        result = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "model": self.model,
                "total_questions": len(all_questions),
                "chunks_sampled": len(sampled_chunks),
                "questions_per_chunk": questions_per_chunk,
                "sampling_strategy": sampling_strategy,
                "taxonomy_distribution": dict(type_distribution),
                "unique_articles": len(
                    set(q["source_article"] for q in all_questions)
                ),
            },
            "questions": all_questions,
        }

        return result

    def save_questions(self, questions_data: Dict, output_path: Path) -> None:
        """
        Save generated questions to JSON file.

        Args:
            questions_data: Questions data dictionary
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(questions_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(questions_data['questions'])} questions to {output_path}")

    def load_questions(self, questions_path: Path) -> Dict:
        """
        Load questions from JSON file.

        Args:
            questions_path: Path to questions file

        Returns:
            Questions data dictionary
        """
        with open(questions_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(
            f"Loaded {len(data['questions'])} questions from {questions_path}"
        )
        return data


if __name__ == "__main__":
    # Test question generator
    print("=" * 70)
    print("Question Generator Test - Chunk-based Generation")
    print("=" * 70)

    # Initialize generator
    generator = QuestionGenerator(skip_api_init=True)

    # Test chunk sampling with mock data
    mock_chunks = [
        {
            "chunk_id": f"chunk_{i}",
            "content": f"This is test content for chunk {i}. " * 50,
            "article_title": f"Article {i % 5}",
            "article_id": f"art_{i % 5}",
            "chunk_index": i,
            "metadata": {},
        }
        for i in range(100)
    ]

    print(f"\nCreated {len(mock_chunks)} mock chunks")

    # Test different sampling strategies
    for strategy in ["random", "stratified", "diverse"]:
        print(f"\n--- Testing '{strategy}' sampling ---")
        sampled = generator.sample_chunks(
            mock_chunks, num_samples=10, strategy=strategy
        )
        print(f"Sampled {len(sampled)} chunks")
        print(
            f"Unique articles: {len(set(c['article_title'] for c in sampled))}"
        )
        print(
            f"Content lengths: min={min(len(c['content']) for c in sampled)}, "
            f"max={max(len(c['content']) for c in sampled)}"
        )

    print("\n" + "=" * 70)
    print("✓ Question generator test complete")
    print("=" * 70)
