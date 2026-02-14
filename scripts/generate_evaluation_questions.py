#!/usr/bin/env python3
"""
Generate evaluation questions from document chunks using OpenRouter Mistral Small.

This script samples chunks from Qdrant and uses LLM to generate diverse questions
across different taxonomic categories.

Usage:
    # Generate 30 questions (30 chunks × 1 question each)
    python scripts/generate_evaluation_questions.py

    # Custom configuration
    python scripts/generate_evaluation_questions.py --num-chunks 50 --questions-per-chunk 2

    # Specify output file
    python scripts/generate_evaluation_questions.py --output data/evaluation/my_questions.json
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.evaluation.question_generator import QuestionGenerator
from src.utils.logger import get_logger
from src.vector_store.qdrant_manager import QdrantManager

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation questions from document chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 30 questions (30 chunks × 1 question each)
  python scripts/generate_evaluation_questions.py

  # Generate more questions
  python scripts/generate_evaluation_questions.py --num-chunks 50 --questions-per-chunk 2

  # Custom output location
  python scripts/generate_evaluation_questions.py --output results/questions.json

  # Use in-memory Qdrant for testing
  python scripts/generate_evaluation_questions.py --use-memory

Question Type Distribution:
  - Factual:        25% (dates, names, events)
  - Temporal:       20% (before/after, chronology)
  - Comparative:    15% (differences, similarities)
  - Entity-Centric: 15% (roles, actions of entities)
  - Relationship:   15% (causal, connections)
  - Analytical:     10% (summaries, impacts)
        """,
    )

    parser.add_argument(
        "--num-chunks",
        "-n",
        type=int,
        default=30,
        help="Number of chunks to sample (default: 30)",
    )

    parser.add_argument(
        "--questions-per-chunk",
        "-q",
        type=int,
        default=1,
        help="Questions to generate per chunk (default: 1)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: data/evaluation/generated_questions.json)",
    )

    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        default=settings.QDRANT_COLLECTION_NAME,
        help=f"Qdrant collection name (default: {settings.QDRANT_COLLECTION_NAME})",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=settings.QUESTION_GEN_MODEL,
        help=f"OpenRouter model to use (default: {settings.QUESTION_GEN_MODEL})",
    )

    parser.add_argument(
        "--sampling-strategy",
        "-s",
        type=str,
        choices=["random", "stratified", "diverse"],
        default="stratified",
        help="Chunk sampling strategy (default: stratified)",
    )

    parser.add_argument(
        "--no-diversity-check",
        action="store_true",
        help="Skip taxonomic diversity enforcement",
    )

    parser.add_argument(
        "--use-memory",
        action="store_true",
        help="Use in-memory Qdrant (requires indexing first)",
    )

    return parser.parse_args()


def print_header():
    """Print script header."""
    print("=" * 70)
    print("Evaluation Question Generator - Chunk-based")
    print("=" * 70)
    print()


def main():
    """Main function."""
    args = parse_args()
    print_header()

    # Check API key
    if not settings.OPENROUTER_API_KEY:
        logger.error("OpenRouter API key not set!")
        print("❌ Error: OPENROUTER_API_KEY not found in environment")
        print()
        print("Please set your OpenRouter API key:")
        print("  1. Get a free API key from https://openrouter.ai/")
        print("  2. Add to your .env file:")
        print("     OPENROUTER_API_KEY=your_key_here")
        print()
        sys.exit(1)

    # Prepare paths
    output_path = (
        Path(args.output)
        if args.output
        else (settings.DATA_DIR / "evaluation" / "generated_questions.json")
    )

    # Calculate total questions
    total_questions = args.num_chunks * args.questions_per_chunk

    print(f"Configuration:")
    print(f"  Model:                {args.model}")
    print(f"  Collection:           {args.collection}")
    print(f"  Chunks to sample:     {args.num_chunks}")
    print(f"  Questions per chunk:  {args.questions_per_chunk}")
    print(f"  Total questions:      {total_questions}")
    print(f"  Sampling strategy:    {args.sampling_strategy}")
    print(f"  Diversity check:      {not args.no_diversity_check}")
    print(f"  Output:               {output_path}")
    print()

    # Initialize Qdrant
    try:
        if args.use_memory:
            logger.info("Using in-memory Qdrant")
            print("⚠ Using in-memory Qdrant - make sure data is already indexed")
            print()
            qdrant = QdrantManager(use_memory=True)
        else:
            logger.info("Connecting to Qdrant...")
            qdrant = QdrantManager()

        # Check collection exists
        if not qdrant.collection_exists(args.collection):
            logger.error(f"Collection '{args.collection}' not found")
            print(f"❌ Error: Collection '{args.collection}' does not exist")
            print()
            print("Index documents first:")
            print("  python scripts/index_all_articles.py")
            print()
            sys.exit(1)

        collection_info = qdrant.get_collection_info(args.collection)
        print(f"Collection '{args.collection}':")
        print(f"  Chunks indexed: {collection_info['points_count']}")
        print(f"  Vector size:    {collection_info['vector_size']}")
        print()

    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        print(f"❌ Error: Cannot connect to Qdrant")
        print()
        if not args.use_memory:
            print("Start Qdrant first:")
            print("  bash scripts/setup_qdrant.sh start")
            print()
            print("Or use --use-memory flag for testing")
        sys.exit(1)

    # Initialize generator
    try:
        logger.info("Initializing question generator...")
        generator = QuestionGenerator(model=args.model, qdrant_manager=qdrant)
        logger.info("Question generator ready")
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

    # Generate questions
    try:
        print("=" * 70)
        print("Generating Questions")
        print("=" * 70)
        print()
        print("This may take a few minutes depending on the number of chunks...")
        print()

        questions_data = generator.generate_evaluation_questions(
            collection_name=args.collection,
            num_chunks=args.num_chunks,
            questions_per_chunk=args.questions_per_chunk,
            sampling_strategy=args.sampling_strategy,
            ensure_taxonomy_diversity=not args.no_diversity_check,
        )

        questions = questions_data.get("questions", [])
        metadata = questions_data.get("metadata", {})

        if not questions:
            logger.error("No questions were generated")
            print("❌ Error: No questions generated")
            sys.exit(1)

        print()
        print("=" * 70)
        print("Generation Summary")
        print("=" * 70)
        print()
        print(f"Total questions generated: {len(questions)}")
        print(f"Chunks sampled:            {metadata.get('chunks_sampled', 0)}")
        print(f"Unique articles covered:   {metadata.get('unique_articles', 0)}")
        print()

        # Show distribution
        type_dist = metadata.get("taxonomy_distribution", {})
        print("Question Type Distribution:")
        total = len(questions)
        for q_type in sorted(type_dist.keys()):
            count = type_dist[q_type]
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {q_type:15s}: {count:3d} ({percentage:5.1f}%)")

        # Save questions
        print()
        print("Saving questions...")
        generator.save_questions(questions_data, output_path)

        print()
        print("=" * 70)
        print(f"✓ Questions saved to: {output_path}")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Review questions in the output file")
        print("  2. Run evaluation: python scripts/run_evaluation.py")
        print()

    except KeyboardInterrupt:
        print("\n\n⚠ Generation interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        print(f"\n❌ Error during generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
