#!/usr/bin/env python3
"""
Compare RAG performance across dataset sizes: 49, 976, and 10,000 articles.

This script analyzes how retrieval metrics scale with dataset size and
generates insights about the "precision gap paradox" and other scaling effects.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings

# Result file paths
RESULTS = {
    49: project_root / "results" / "phase1_baseline_35q.json",
    976: project_root / "results" / "phase1_baseline_976articles_49q.json",
    10000: project_root / "results" / "phase1_baseline_10000articles_50q.json",
}


def load_results(filepath: Path) -> Dict:
    """Load evaluation results from JSON file."""
    if not filepath.exists():
        return None

    with open(filepath, "r") as f:
        return json.load(f)


def extract_metrics(results: Dict, dataset_size: int) -> Dict:
    """Extract key metrics from results."""
    if not results:
        return None

    # Handle different result formats
    if "avg_article_hit_at_k" in results:
        # BenchmarkRunner format
        return {
            "article_hit_1": results["avg_article_hit_at_k"].get("1", 0.0),
            "article_hit_3": results["avg_article_hit_at_k"].get("3", 0.0),
            "article_hit_5": results["avg_article_hit_at_k"].get("5", 0.0),
            "article_hit_10": results["avg_article_hit_at_k"].get("10", 0.0),
            "chunk_hit_1": results["avg_chunk_hit_at_k"].get("1", 0.0),
            "chunk_hit_3": results["avg_chunk_hit_at_k"].get("3", 0.0),
            "chunk_hit_5": results["avg_chunk_hit_at_k"].get("5", 0.0),
            "chunk_hit_10": results["avg_chunk_hit_at_k"].get("10", 0.0),
            "mrr": results.get("avg_mrr", 0.0),
            "num_questions": results.get("total_questions", 0),
        }
    else:
        # Unknown format
        return None


def calculate_precision_gap(metrics: Dict) -> float:
    """Calculate precision gap (Article Hit - Chunk Hit)."""
    return metrics["article_hit_5"] - metrics["chunk_hit_5"]


def print_comparison_table(all_metrics: Dict[int, Dict]):
    """Print formatted comparison table."""
    print("=" * 100)
    print("DATASET SIZE COMPARISON: Phase 1 Pure Semantic Search")
    print("=" * 100)
    print()

    # Dataset info
    print("Dataset Specifications:")
    print()
    print(f"{'Metric':<25} | {'49 Articles':<15} | {'976 Articles':<15} | {'10,000 Articles':<15}")
    print("-" * 100)

    sizes = sorted(all_metrics.keys())
    questions = [all_metrics[s]["num_questions"] for s in sizes]

    print(f"{'Number of Questions':<25} | {questions[0]:<15} | {questions[1]:<15} | {questions[2]:<15}")
    print()

    # Performance metrics
    print("=" * 100)
    print("PERFORMANCE METRICS")
    print("=" * 100)
    print()

    metrics_to_compare = [
        ("Article Hit@1", "article_hit_1", "%"),
        ("Article Hit@3", "article_hit_3", "%"),
        ("Article Hit@5", "article_hit_5", "%"),
        ("Article Hit@10", "article_hit_10", "%"),
        ("", "", ""),  # Separator
        ("Chunk Hit@1", "chunk_hit_1", "%"),
        ("Chunk Hit@3", "chunk_hit_3", "%"),
        ("Chunk Hit@5", "chunk_hit_5", "%"),
        ("Chunk Hit@10", "chunk_hit_10", "%"),
        ("", "", ""),  # Separator
        ("MRR", "mrr", ""),
    ]

    print(f"{'Metric':<25} | {'49 Articles':<15} | {'976 Articles':<15} | {'10,000 Articles':<15}")
    print("-" * 100)

    for label, key, fmt in metrics_to_compare:
        if not label:  # Separator
            print()
            continue

        values = []
        for size in sizes:
            val = all_metrics[size][key]
            if fmt == "%":
                values.append(f"{val:.1%}")
            else:
                values.append(f"{val:.3f}")

        print(f"{label:<25} | {values[0]:<15} | {values[1]:<15} | {values[2]:<15}")

    print()

    # Precision gap analysis
    print("=" * 100)
    print("PRECISION GAP ANALYSIS")
    print("=" * 100)
    print()

    gaps = {size: calculate_precision_gap(all_metrics[size]) for size in sizes}

    print(f"{'Dataset Size':<25} | {'Precision Gap':<20} | {'Interpretation':<50}")
    print("-" * 100)

    for size in sizes:
        gap = gaps[size]
        if gap < 0.05:
            interpretation = "Excellent: Find chunk when we find article"
        elif gap < 0.15:
            interpretation = "Good: Usually find chunk with article"
        elif gap < 0.25:
            interpretation = "Moderate: Sometimes find wrong chunk"
        else:
            interpretation = "Poor: Often find article but wrong chunk"

        print(f"{f'{size:,} articles':<25} | {gap:>7.1%}{'':>13} | {interpretation:<50}")

    print()

    # Scaling trends
    print("=" * 100)
    print("SCALING TRENDS")
    print("=" * 100)
    print()

    print("As dataset size increases from 49 → 976 → 10,000 articles:")
    print()

    # Analyze trends
    article_hit_5_trend = [all_metrics[s]["article_hit_5"] for s in sizes]
    chunk_hit_5_trend = [all_metrics[s]["chunk_hit_5"] for s in sizes]
    mrr_trend = [all_metrics[s]["mrr"] for s in sizes]
    gap_trend = [gaps[s] for s in sizes]

    def trend_direction(values: List[float]) -> str:
        """Determine trend direction."""
        if values[-1] > values[0] + 0.05:
            return "↗ INCREASING"
        elif values[-1] < values[0] - 0.05:
            return "↘ DECREASING"
        else:
            return "→ STABLE"

    print(f"Article Hit@5:  {trend_direction(article_hit_5_trend):<15} ({article_hit_5_trend[0]:.1%} → {article_hit_5_trend[-1]:.1%})")
    print(f"Chunk Hit@5:    {trend_direction(chunk_hit_5_trend):<15} ({chunk_hit_5_trend[0]:.1%} → {chunk_hit_5_trend[-1]:.1%})")
    print(f"MRR:            {trend_direction(mrr_trend):<15} ({mrr_trend[0]:.3f} → {mrr_trend[-1]:.3f})")
    print(f"Precision Gap:  {trend_direction(gap_trend):<15} ({gap_trend[0]:.1%} → {gap_trend[-1]:.1%})")
    print()


def generate_insights(all_metrics: Dict[int, Dict]):
    """Generate key insights from the comparison."""
    print("=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)
    print()

    sizes = sorted(all_metrics.keys())

    # Insight 1: Precision Gap Paradox
    gaps = {size: calculate_precision_gap(all_metrics[size]) for size in sizes}

    print("1. THE PRECISION GAP PARADOX")
    print()
    print(f"   As dataset size increased from {sizes[0]} to {sizes[-1]:,} articles ({sizes[-1]/sizes[0]:.0f}x),")
    print(f"   the precision gap changed from {gaps[sizes[0]]:.1%} to {gaps[sizes[-1]]:.1%}.")
    print()

    if gaps[sizes[-1]] < gaps[sizes[0]]:
        improvement = gaps[sizes[0]] - gaps[sizes[-1]]
        print(f"   ✓ IMPROVED by {improvement:.1%}")
        print(f"     → Larger datasets force better semantic discrimination")
        print(f"     → Finding the right article now means finding the right chunk")
    else:
        print(f"   ⚠ WORSENED or stayed same")
        print(f"     → May need reranking strategies")
    print()

    # Insight 2: Chunk Precision
    chunk_hit_5_values = [all_metrics[s]["chunk_hit_5"] for s in sizes]

    print("2. CHUNK PRECISION AT SCALE")
    print()
    print(f"   Chunk Hit@5 performance:")
    for i, size in enumerate(sizes):
        print(f"     {size:>6,} articles: {chunk_hit_5_values[i]:.1%}")
    print()

    if chunk_hit_5_values[-1] >= chunk_hit_5_values[0]:
        print(f"   ✓ Maintained or improved despite {sizes[-1]/sizes[0]:.0f}x larger search space!")
        print(f"     → Pure semantic search scales well")
    else:
        decline = chunk_hit_5_values[0] - chunk_hit_5_values[-1]
        print(f"   ⚠ Declined by {decline:.1%}")
        print(f"     → Consider hybrid search or reranking")
    print()

    # Insight 3: Article Recall
    article_hit_5_values = [all_metrics[s]["article_hit_5"] for s in sizes]

    print("3. ARTICLE RECALL CHALLENGE")
    print()
    print(f"   Article Hit@5 performance:")
    for i, size in enumerate(sizes):
        print(f"     {size:>6,} articles: {article_hit_5_values[i]:.1%}")
    print()

    if article_hit_5_values[-1] < article_hit_5_values[0] - 0.05:
        decline = article_hit_5_values[0] - article_hit_5_values[-1]
        print(f"   ⚠ Declined by {decline:.1%} as expected")
        print(f"     → Finding the right article is harder with {sizes[-1]:,} options")
        print(f"     → Recommendation: Hybrid search (BM25 + semantic) for article-level recall")
    else:
        print(f"   ✓ Maintained article recall despite larger dataset")
    print()

    # Insight 4: MRR Trends
    mrr_values = [all_metrics[s]["mrr"] for s in sizes]

    print("4. MEAN RECIPROCAL RANK (SPEED TO FIRST RESULT)")
    print()
    print(f"   MRR performance:")
    for i, size in enumerate(sizes):
        print(f"     {size:>6,} articles: {mrr_values[i]:.3f}")
    print()

    if mrr_values[-1] >= mrr_values[0] - 0.05:
        print(f"   ✓ Remained stable despite {sizes[-1]/sizes[0]:.0f}x larger search space")
        print(f"     → HNSW index provides sub-linear scaling")
    else:
        print(f"   ⚠ Decreased as search space grew")
        print(f"     → Expected behavior: relevant results appear slightly later")
    print()


def save_summary_report(all_metrics: Dict[int, Dict], output_file: Path):
    """Save markdown summary report."""
    with open(output_file, "w") as f:
        f.write("# Dataset Size Scaling Analysis\n\n")
        f.write(f"**Date:** {Path(output_file).stat().st_mtime}\n\n")

        f.write("## Executive Summary\n\n")

        sizes = sorted(all_metrics.keys())
        gaps = {size: calculate_precision_gap(all_metrics[size]) for size in sizes}

        f.write(f"Testing across {len(sizes)} dataset sizes ({sizes[0]} → {sizes[-1]:,} articles, ")
        f.write(f"{sizes[-1]/sizes[0]:.0f}x increase) reveals:\n\n")

        # Key findings
        chunk_hit_5_values = [all_metrics[s]["chunk_hit_5"] for s in sizes]

        if chunk_hit_5_values[-1] >= chunk_hit_5_values[0]:
            f.write(f"✅ **Chunk Hit@5 maintained/improved**: {chunk_hit_5_values[0]:.1%} → {chunk_hit_5_values[-1]:.1%}\n")
        else:
            f.write(f"⚠️ **Chunk Hit@5 declined**: {chunk_hit_5_values[0]:.1%} → {chunk_hit_5_values[-1]:.1%}\n")

        if gaps[sizes[-1]] < gaps[sizes[0]]:
            f.write(f"✅ **Precision Gap improved**: {gaps[sizes[0]]:.1%} → {gaps[sizes[-1]]:.1%}\n")
        else:
            f.write(f"⚠️ **Precision Gap worsened**: {gaps[sizes[0]]:.1%} → {gaps[sizes[-1]]:.1%}\n")

        f.write("\n## Detailed Metrics\n\n")
        f.write("| Metric | 49 Articles | 976 Articles | 10,000 Articles |\n")
        f.write("|--------|-------------|--------------|------------------|\n")

        for size in sizes:
            m = all_metrics[size]
            f.write(f"| Article Hit@5 | {all_metrics[49]['article_hit_5']:.1%} | ")
            f.write(f"{all_metrics[976]['article_hit_5']:.1%} | ")
            f.write(f"{all_metrics[10000]['article_hit_5']:.1%} |\n")
            break  # Just need the header row

        # Add more rows...

    print(f"✓ Summary report saved to: {output_file}")


def main():
    print("=" * 100)
    print("DATASET SIZE SCALING ANALYSIS")
    print("Comparing RAG Performance: 49 vs 976 vs 10,000 Articles")
    print("=" * 100)
    print()

    # Load all results
    all_metrics = {}

    for size, filepath in RESULTS.items():
        print(f"Loading {size:,}-article results from {filepath.name}...")
        results = load_results(filepath)

        if results is None:
            print(f"  ⚠ Not found - skipping")
            continue

        metrics = extract_metrics(results, size)
        if metrics is None:
            print(f"  ⚠ Could not extract metrics - skipping")
            continue

        all_metrics[size] = metrics
        print(f"  ✓ Loaded ({metrics['num_questions']} questions)")

    print()

    if len(all_metrics) < 2:
        print("❌ Need at least 2 datasets to compare")
        print()
        print("Available results:")
        for size, filepath in RESULTS.items():
            status = "✓" if filepath.exists() else "✗"
            print(f"  {status} {size:>6,} articles: {filepath}")
        print()
        return

    # Print comparison table
    print_comparison_table(all_metrics)

    # Generate insights
    generate_insights(all_metrics)

    # Save report
    print("=" * 100)
    print("SAVING REPORT")
    print("=" * 100)
    print()

    output_file = project_root / "results" / "SCALING_ANALYSIS.md"
    # save_summary_report(all_metrics, output_file)  # TODO: Implement full markdown generation

    print()
    print("=" * 100)
    print("Analysis Complete!")
    print("=" * 100)
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", exc_info=True)
        sys.exit(1)
