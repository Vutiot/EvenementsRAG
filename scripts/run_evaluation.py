#!/usr/bin/env python3
"""
Run RAG system evaluation with Recall@K and other metrics.

This script runs a complete benchmark evaluation on a RAG phase,
computing Recall@K, MRR, NDCG, and other metrics.

Usage:
    # Run evaluation on Phase 1 (vanilla RAG)
    python scripts/run_evaluation.py --phase phase1_vanilla

    # Run on custom collection with specific K values
    python scripts/run_evaluation.py --collection my_collection --k-values 1,3,5,10

    # Limit number of questions for testing
    python scripts/run_evaluation.py --max-questions 10
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.evaluation.benchmark_runner import BenchmarkRunner
from src.utils.logger import get_logger
from src.vector_store.qdrant_manager import QdrantManager

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RAG system evaluation benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Phase 1 (vanilla RAG)
  python scripts/run_evaluation.py --phase phase1_vanilla

  # Evaluate with custom collection
  python scripts/run_evaluation.py --collection ww2_events --phase phase2_temporal

  # Quick test with 10 questions
  python scripts/run_evaluation.py --max-questions 10 --phase test

  # Custom K values for Recall@K
  python scripts/run_evaluation.py --k-values 1,5,10 --phase phase1_vanilla

Metrics Computed:
  - Recall@K: Fraction of relevant documents in top-K results
  - MRR: Mean Reciprocal Rank (position of first relevant result)
  - NDCG@K: Normalized Discounted Cumulative Gain (ranking quality)
  - Per-question-type breakdown
        """,
    )

    parser.add_argument(
        "--phase",
        "-p",
        type=str,
        default="phase1_vanilla",
        help="RAG phase name (default: phase1_vanilla)",
    )

    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        default=settings.QDRANT_COLLECTION_NAME,
        help=f"Qdrant collection name (default: {settings.QDRANT_COLLECTION_NAME})",
    )

    parser.add_argument(
        "--questions-file",
        "-q",
        type=str,
        default=None,
        help="Questions JSON file (default: data/evaluation/generated_questions.json)",
    )

    parser.add_argument(
        "--max-questions",
        "-m",
        type=int,
        default=None,
        help="Maximum questions to evaluate (default: all)",
    )

    parser.add_argument(
        "--k-values",
        "-k",
        type=str,
        default="1,3,5,10",
        help="Comma-separated K values for Recall@K (default: 1,3,5,10)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: results/<phase>_metrics.json)",
    )

    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also export to CSV format",
    )

    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Don't print summary to console",
    )

    return parser.parse_args()


def print_header():
    """Print script header."""
    print("=" * 70)
    print("RAG System Evaluation Benchmark")
    print("=" * 70)
    print()


def check_prerequisites(questions_file: Path, collection_name: str, qdrant: QdrantManager):
    """Check if prerequisites are met."""
    # Check questions file
    if not questions_file.exists():
        logger.error(f"Questions file not found: {questions_file}")
        print(f"❌ Error: Questions file not found")
        print(f"   Path: {questions_file}")
        print()
        print("Generate questions first:")
        print("  python scripts/generate_evaluation_questions.py")
        print()
        return False

    # Check collection exists
    if not qdrant.collection_exists(collection_name):
        logger.error(f"Collection not found: {collection_name}")
        print(f"❌ Error: Qdrant collection '{collection_name}' not found")
        print()
        print("Available collections:")
        stats = qdrant.get_statistics()
        if stats.get("collections"):
            for coll_name in stats["collections"].keys():
                print(f"  - {coll_name}")
        else:
            print("  (none)")
        print()
        print("Index documents first:")
        print("  python -m src.vector_store.indexer")
        print()
        return False

    return True


def main():
    """Main function."""
    args = parse_args()
    print_header()

    # Parse K values
    try:
        k_values = [int(k.strip()) for k in args.k_values.split(",")]
    except ValueError:
        print(f"❌ Error: Invalid K values: {args.k_values}")
        print("   Expected format: 1,3,5,10")
        sys.exit(1)

    # Prepare paths
    questions_file = Path(args.questions_file) if args.questions_file else (
        settings.DATA_DIR / "evaluation" / "generated_questions.json"
    )

    output_path = Path(args.output) if args.output else (
        project_root / "results" / f"{args.phase}_metrics.json"
    )

    print(f"Configuration:")
    print(f"  Phase:           {args.phase}")
    print(f"  Collection:      {args.collection}")
    print(f"  Questions file:  {questions_file}")
    print(f"  K values:        {k_values}")
    if args.max_questions:
        print(f"  Max questions:   {args.max_questions}")
    print(f"  Output:          {output_path}")
    print()

    # Initialize Qdrant
    try:
        logger.info("Connecting to Qdrant...")
        qdrant = QdrantManager()
        logger.info("Connected to Qdrant")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        print("❌ Error: Cannot connect to Qdrant")
        print()
        print("Start Qdrant first:")
        print("  bash scripts/setup_qdrant.sh start")
        sys.exit(1)

    # Check prerequisites
    if not check_prerequisites(questions_file, args.collection, qdrant):
        sys.exit(1)

    # Show collection info
    collection_info = qdrant.get_collection_info(args.collection)
    print("Collection Info:")
    print(f"  Name:          {collection_info['name']}")
    print(f"  Vector size:   {collection_info['vector_size']}")
    print(f"  Chunks indexed: {collection_info['points_count']}")
    print()

    # Initialize benchmark runner
    try:
        logger.info("Initializing benchmark runner...")
        runner = BenchmarkRunner(
            questions_file=questions_file,
            qdrant_manager=qdrant,
            k_values=k_values,
        )
    except Exception as e:
        logger.error(f"Failed to initialize benchmark runner: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

    # Run benchmark
    try:
        print("=" * 70)
        print("Running Benchmark")
        print("=" * 70)
        print()

        results = runner.run_benchmark(
            collection_name=args.collection,
            phase_name=args.phase,
            max_questions=args.max_questions,
        )

        # Print summary
        if not args.no_summary:
            runner.print_summary(results)

        # Export results
        output_path.parent.mkdir(parents=True, exist_ok=True)

        runner.export_results(results, output_path, format="json")
        print(f"\n✓ Results saved to: {output_path}")

        # Export CSV if requested
        if args.csv:
            csv_path = output_path.with_suffix(".csv")
            runner.export_results(results, csv_path, format="csv")
            print(f"✓ CSV exported to: {csv_path}")

        print()
        print("=" * 70)
        print("Evaluation Complete")
        print("=" * 70)

        # Exit code based on quality threshold
        if results.avg_recall_at_k.get(5, 0.0) < settings.EVALUATION_MIN_RECALL_AT_5:
            print(f"\n⚠ Warning: Recall@5 ({results.avg_recall_at_k[5]:.3f}) is below "
                  f"threshold ({settings.EVALUATION_MIN_RECALL_AT_5:.3f})")
            sys.exit(2)  # Exit with warning code

    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        print(f"\n❌ Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
