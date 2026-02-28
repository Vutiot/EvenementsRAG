#!/usr/bin/env python3
"""
Index 10,000 articles and generate 200 evaluation questions.

Pipeline:
  1. Index all articles from data/raw/wikipedia_articles_10000/ into
     an in-memory Qdrant collection.
  2. Sample 200 chunks (stratified across articles).
  3. Generate 1 question per chunk with full taxonomy diversity.
  4. Save results to data/evaluation/eval_10k_200q.json.

Rate limiting: 4-second delay between LLM calls (fits within OpenRouter
free tier of ~16 req/min).
Retry logic: 3 retries with exponential backoff on API failures.
"""

import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.utils.logger import get_logger
from src.vector_store.qdrant_manager import QdrantManager
from src.vector_store.indexer import DocumentIndexer
from src.evaluation.question_generator import QuestionGenerator, QUESTION_TAXONOMY

logger = get_logger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
ARTICLES_DIR = settings.DATA_DIR / "raw" / "wikipedia_articles_10000"
COLLECTION_NAME = "ww2_events_10000"
OUTPUT_FILE = settings.DATA_DIR / "evaluation" / "eval_10k_200q.json"

NUM_QUESTIONS = 200          # Total questions to generate
QUESTIONS_PER_CHUNK = 1      # 1 question per chunk → 200 chunks sampled
SAMPLING_STRATEGY = "stratified"

LLM_DELAY_SECONDS = 4.0     # Delay between LLM API calls (fits free-tier limits)
MAX_RETRIES = 3              # Retries per chunk on API failure
RETRY_BASE_DELAY = 10.0     # Base delay for exponential backoff (seconds)


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_question_type_sequence(total: int) -> List[str]:
    """
    Build an ordered list of question types matching taxonomy proportions.

    Expected distribution for 200 questions:
      factual       50  (25%)
      temporal      40  (20%)
      comparative   30  (15%)
      entity_centric 30 (15%)
      relationship  30  (15%)
      analytical    20  (10%)
    """
    types: List[str] = []
    for qtype, proportion in QUESTION_TAXONOMY.items():
        count = round(total * proportion)
        types.extend([qtype] * count)

    # Fill any rounding gap with random types
    while len(types) < total:
        types.append(random.choice(list(QUESTION_TAXONOMY.keys())))

    random.shuffle(types)
    return types[:total]


def generate_with_retry(
    generator: QuestionGenerator,
    chunk: Dict,
    num_questions: int,
    target_type: str,
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_BASE_DELAY,
) -> List[Dict]:
    """
    Call generator.generate_question_for_chunk with exponential-backoff retry.

    Returns:
        List of generated question dicts (empty on total failure).
    """
    for attempt in range(1, max_retries + 1):
        result = generator.generate_question_for_chunk(
            chunk=chunk,
            num_questions=num_questions,
            target_type=target_type,
        )
        if result:
            return result

        if attempt < max_retries:
            wait = base_delay * (2 ** (attempt - 1))
            logger.warning(
                f"Attempt {attempt}/{max_retries} failed for chunk "
                f"'{chunk.get('chunk_id')}'. Retrying in {wait:.0f}s..."
            )
            time.sleep(wait)
        else:
            logger.error(
                f"All {max_retries} attempts failed for chunk "
                f"'{chunk.get('chunk_id')}'. Skipping."
            )

    return []


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Generate 200 Evaluation Questions from 10k Articles")
    print("=" * 70)
    print()

    # ── Preflight checks ──────────────────────────────────────────────────────
    if not ARTICLES_DIR.exists():
        print(f"ERROR: Articles directory not found: {ARTICLES_DIR}")
        print("Please run: python scripts/scrape_10k_articles.py")
        sys.exit(1)

    article_files = list(ARTICLES_DIR.glob("*.json"))
    article_count = len(article_files)
    print(f"Articles directory : {ARTICLES_DIR}")
    print(f"Articles found     : {article_count}")
    print()

    if article_count == 0:
        print("ERROR: No articles found. Run scrape_10k_articles.py first.")
        sys.exit(1)

    if article_count < 1000:
        print(
            f"WARNING: Only {article_count} articles found. "
            "Evaluation quality may be limited with a small corpus."
        )
        print()

    # ── Step 1: Initialize in-memory Qdrant ───────────────────────────────────
    print("=" * 70)
    print("Step 1: Initializing In-Memory Qdrant")
    print("=" * 70)
    print()

    qdrant = QdrantManager(use_memory=True)
    logger.info("In-memory Qdrant initialized")
    print("In-memory Qdrant ready.")
    print()

    # ── Step 2: Index articles ────────────────────────────────────────────────
    print("=" * 70)
    print(f"Step 2: Indexing {article_count} Articles")
    print("=" * 70)
    print()
    print("This may take 5–15 minutes depending on corpus size...")
    print("(Embedding cache accelerates repeated runs.)")
    print()

    indexer = DocumentIndexer(qdrant_manager=qdrant)
    stats = indexer.index_all_articles(
        collection_name=COLLECTION_NAME,
        articles_dir=ARTICLES_DIR,
        recreate_collection=True,
    )

    if not stats.get("success"):
        print(f"ERROR: Indexing failed — {stats.get('error')}")
        sys.exit(1)

    chunks_indexed = stats["chunks_indexed"]
    articles_loaded = stats["articles_loaded"]
    print(f"Indexed {chunks_indexed:,} chunks from {articles_loaded:,} articles")
    print(
        f"Average chunks/article: "
        f"{chunks_indexed / articles_loaded:.1f}"
    )
    print()

    # ── Step 3: Load chunks from Qdrant ───────────────────────────────────────
    print("=" * 70)
    print("Step 3: Loading Chunks & Sampling")
    print("=" * 70)
    print()

    generator = QuestionGenerator(qdrant_manager=qdrant)

    all_chunks = generator.load_chunks_from_qdrant(COLLECTION_NAME)
    print(f"Loaded {len(all_chunks):,} chunks from Qdrant")

    sampled_chunks = generator.sample_chunks(
        all_chunks,
        num_samples=NUM_QUESTIONS,
        strategy=SAMPLING_STRATEGY,
    )

    unique_articles = len(set(c["article_title"] for c in sampled_chunks))
    print(f"Sampled {len(sampled_chunks)} chunks from {unique_articles} unique articles")
    print()

    # ── Step 4: Generate questions ────────────────────────────────────────────
    print("=" * 70)
    print(f"Step 4: Generating {NUM_QUESTIONS} Questions (1 per chunk)")
    print("=" * 70)
    print()
    print(f"Rate limit delay : {LLM_DELAY_SECONDS}s between LLM calls")
    print(f"Max retries      : {MAX_RETRIES} per chunk")
    print()

    estimated_minutes = (NUM_QUESTIONS * LLM_DELAY_SECONDS) / 60
    print(f"Estimated time   : ~{estimated_minutes:.0f} minutes")
    print()

    question_types = build_question_type_sequence(NUM_QUESTIONS)

    # Print planned taxonomy
    planned_dist = Counter(question_types)
    print("Planned taxonomy distribution:")
    for qtype in sorted(planned_dist.keys()):
        count = planned_dist[qtype]
        pct = count / NUM_QUESTIONS * 100
        print(f"  {qtype:15s}: {count:3d} ({pct:.0f}%)")
    print()

    all_questions: List[Dict] = []
    question_id_counter = 1

    for i, chunk in enumerate(sampled_chunks):
        target_type = question_types[i] if i < len(question_types) else "factual"

        questions = generate_with_retry(
            generator=generator,
            chunk=chunk,
            num_questions=QUESTIONS_PER_CHUNK,
            target_type=target_type,
        )

        for q in questions:
            q["id"] = f"gen_q{question_id_counter:03d}"
            question_id_counter += 1

        all_questions.extend(questions)

        if (i + 1) % 20 == 0 or (i + 1) == len(sampled_chunks):
            pct = (i + 1) / len(sampled_chunks) * 100
            print(
                f"  [{i+1:3d}/{len(sampled_chunks)}] "
                f"{pct:5.1f}% — {len(all_questions)} questions so far"
            )

        # Rate limiting: sleep between LLM calls (not after the last one)
        if i < len(sampled_chunks) - 1:
            time.sleep(LLM_DELAY_SECONDS)

    print()
    print(f"Generated {len(all_questions)} questions total")
    print()

    # ── Step 5: Build result and save ─────────────────────────────────────────
    print("=" * 70)
    print("Step 5: Saving Results")
    print("=" * 70)
    print()

    actual_dist = Counter(q.get("type", "unknown") for q in all_questions)

    questions_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": generator.model,
            "total_questions": len(all_questions),
            "chunks_sampled": len(sampled_chunks),
            "questions_per_chunk": QUESTIONS_PER_CHUNK,
            "sampling_strategy": SAMPLING_STRATEGY,
            "taxonomy_distribution": dict(actual_dist),
            "unique_articles": len(
                set(q.get("source_article", "") for q in all_questions)
            ),
            "articles_indexed": articles_loaded,
            "chunks_indexed": chunks_indexed,
        },
        "questions": all_questions,
    }

    generator.save_questions(questions_data, OUTPUT_FILE)

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Complete!")
    print("=" * 70)
    print()
    print(f"Output file    : {OUTPUT_FILE}")
    print(f"Total questions: {len(all_questions)}")
    print()

    print("Actual taxonomy distribution:")
    for qtype in sorted(actual_dist.keys()):
        count = actual_dist[qtype]
        pct = count / len(all_questions) * 100 if all_questions else 0
        print(f"  {qtype:15s}: {count:3d} ({pct:.1f}%)")
    print()

    print("Verification commands:")
    print(
        "  ls data/raw/wikipedia_articles_10000/ | wc -l"
    )
    print(
        "  python -c \""
        "import json; d=json.load(open('data/evaluation/eval_10k_200q.json')); "
        "print(len(d['questions']), 'questions')\""
    )
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
