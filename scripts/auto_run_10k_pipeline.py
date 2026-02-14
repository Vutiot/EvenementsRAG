#!/usr/bin/env python3
"""
Monitor download progress and auto-run evaluation pipeline when complete.

This script:
1. Monitors the download directory for progress
2. When 10,000 articles are downloaded (or close), runs the full pipeline:
   - Index articles and generate questions
   - Run Phase 1 evaluation
   - Compare results across datasets
"""

import sys
import time
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings

# Configuration
ARTICLES_DIR = settings.DATA_DIR / "raw" / "wikipedia_articles_10000"
TARGET_ARTICLES = 10000
CHECK_INTERVAL = 60  # Check every 60 seconds
MIN_ARTICLES = 9500  # Start pipeline if we reach at least this many


def count_articles() -> int:
    """Count downloaded articles."""
    if not ARTICLES_DIR.exists():
        return 0
    return len(list(ARTICLES_DIR.glob("*.json")))


def format_time(seconds: int) -> str:
    """Format seconds as HH:MM:SS."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def run_pipeline():
    """Run the full evaluation pipeline."""
    print()
    print("=" * 70)
    print("STARTING AUTOMATED EVALUATION PIPELINE")
    print("=" * 70)
    print()

    scripts = [
        ("index_and_generate_questions_10k.py", "Indexing & Question Generation"),
        ("run_phase1_10k.py", "Phase 1 Baseline Evaluation"),
        ("compare_dataset_sizes.py", "Dataset Comparison Analysis"),
    ]

    for script, description in scripts:
        print()
        print("=" * 70)
        print(f"Step: {description}")
        print("=" * 70)
        print(f"Running: {script}")
        print()

        script_path = project_root / "scripts" / script

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(project_root),
                check=True,
                capture_output=False,  # Show output in real-time
            )

            print()
            print(f"✓ {description} completed successfully")

        except subprocess.CalledProcessError as e:
            print()
            print(f"❌ {description} failed with exit code {e.returncode}")
            print(f"Error: {e}")
            return False

    return True


def main():
    print("=" * 70)
    print("AUTO-RUN PIPELINE: Monitoring Download Progress")
    print("=" * 70)
    print()
    print(f"Target articles: {TARGET_ARTICLES:,}")
    print(f"Minimum to start: {MIN_ARTICLES:,}")
    print(f"Check interval: {CHECK_INTERVAL}s")
    print()

    start_time = time.time()
    last_count = 0
    last_check_time = time.time()

    print("Monitoring download progress...")
    print()

    while True:
        current_count = count_articles()
        current_time = time.time()
        elapsed = current_time - start_time

        # Calculate download rate
        if current_count > last_count and current_time > last_check_time:
            time_diff = current_time - last_check_time
            count_diff = current_count - last_count
            rate = count_diff / time_diff

            # Estimate remaining time
            remaining = TARGET_ARTICLES - current_count
            if rate > 0:
                eta_seconds = int(remaining / rate)
                eta_str = format_time(eta_seconds)
            else:
                eta_str = "Unknown"
        else:
            rate = 0.0
            eta_str = "Calculating..."

        # Progress bar
        progress_pct = (current_count / TARGET_ARTICLES) * 100
        bar_width = 40
        filled = int(bar_width * current_count / TARGET_ARTICLES)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Print status
        print(f"\r[{bar}] {current_count:,}/{TARGET_ARTICLES:,} ({progress_pct:.1f}%) | "
              f"Rate: {rate:.1f}/s | ETA: {eta_str} | Elapsed: {format_time(int(elapsed))}",
              end="", flush=True)

        # Check if download complete
        if current_count >= MIN_ARTICLES:
            print()
            print()
            print("=" * 70)
            print(f"✓ Download Complete! ({current_count:,} articles)")
            print("=" * 70)
            print()

            # Wait a bit to ensure download process has fully finished
            print("Waiting 10 seconds to ensure download is complete...")
            time.sleep(10)

            # Final count
            final_count = count_articles()
            print(f"Final article count: {final_count:,}")
            print()

            # Run pipeline
            success = run_pipeline()

            if success:
                print()
                print("=" * 70)
                print("✓ FULL PIPELINE COMPLETED SUCCESSFULLY!")
                print("=" * 70)
                print()
                print("Results saved:")
                print("  - Phase 1 results: results/phase1_baseline_10000articles_50q.json")
                print("  - Comparison output: (printed to console)")
                print()
            else:
                print()
                print("=" * 70)
                print("⚠ PIPELINE COMPLETED WITH ERRORS")
                print("=" * 70)
                print()

            break

        # Update for next iteration
        last_count = current_count
        last_check_time = current_time

        # Sleep before next check
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print()
        print("Monitoring interrupted by user")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
