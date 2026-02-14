#!/usr/bin/env python3
"""
Download Wikipedia articles for historical events.

This script downloads Wikipedia articles based on a period configuration
(e.g., World War II) and saves them for use in the RAG system.

Usage:
    # Download first 50 priority WW2 articles
    python scripts/download_wikipedia_data.py --max-articles 50

    # Download all priority articles
    python scripts/download_wikipedia_data.py

    # Use custom config
    python scripts/download_wikipedia_data.py --config config/periods/custom.yaml

    # Download specific articles
    python scripts/download_wikipedia_data.py --titles "D-Day" "Pearl Harbor"

    # Faster download (less delay, but respect Wikipedia)
    python scripts/download_wikipedia_data.py --max-articles 10 --delay 0.5
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_ingestion.wikipedia_fetcher import WikipediaFetcher
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Wikipedia articles for historical events RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download first 50 WW2 articles
  python scripts/download_wikipedia_data.py --max-articles 50

  # Download all priority articles with 2 second delay
  python scripts/download_wikipedia_data.py --delay 2.0

  # Download specific articles
  python scripts/download_wikipedia_data.py --titles "Battle of Stalingrad" "Yalta Conference"

  # Use different period config
  python scripts/download_wikipedia_data.py --config config/periods/custom.yaml
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/periods/world_war_2.yaml",
        help="Path to period configuration file (default: world_war_2.yaml)",
    )

    parser.add_argument(
        "--max-articles",
        "-m",
        type=int,
        default=None,
        help="Maximum number of articles to download (default: all priority articles)",
    )

    parser.add_argument(
        "--titles",
        "-t",
        nargs="+",
        help="Specific article titles to download (overrides priority list)",
    )

    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0, min: 0.5)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for articles (default: data/raw/wikipedia_articles)",
    )

    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="en",
        help="Wikipedia language code (default: en)",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip articles that were already downloaded (default: True)",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if articles exist",
    )

    parser.add_argument(
        "--stats",
        "-s",
        action="store_true",
        help="Show statistics and exit (don't download)",
    )

    return parser.parse_args()


def print_header():
    """Print script header."""
    print("=" * 70)
    print("Wikipedia Article Downloader for EvenementsRAG")
    print("=" * 70)
    print()


def print_stats(fetcher: WikipediaFetcher):
    """Print fetcher statistics."""
    stats = fetcher.get_statistics()

    print("\n" + "=" * 70)
    print("Statistics")
    print("=" * 70)
    print(f"Period:              {stats['period']}")
    print(f"Config:              {stats['config_file']}")
    print(f"Output Directory:    {stats['output_directory']}")
    print(f"Total Fetched:       {stats['total_fetched']} articles")
    print(f"Files on Disk:       {stats['files_on_disk']} files")
    print(f"Total Size:          {stats['total_size_mb']} MB")
    print("=" * 70)


def main():
    """Main function."""
    args = parse_args()
    print_header()

    # Validate delay
    if args.delay < 0.5:
        logger.warning(
            f"Delay {args.delay}s is too short. Setting to 0.5s to respect Wikipedia."
        )
        args.delay = 0.5

    # Initialize fetcher
    try:
        logger.info(f"Initializing fetcher with config: {args.config}")

        output_dir = Path(args.output_dir) if args.output_dir else None

        fetcher = WikipediaFetcher(
            period_config_path=args.config,
            output_dir=output_dir,
            language=args.language,
        )

        logger.info("Fetcher initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize fetcher: {e}")
        sys.exit(1)

    # Show stats and exit if requested
    if args.stats:
        print_stats(fetcher)
        return

    # Determine what to fetch
    skip_existing = args.skip_existing and not args.force

    if args.titles:
        # Fetch specific titles
        logger.info(f"Fetching {len(args.titles)} specific articles")
        print(f"\nFetching {len(args.titles)} specific articles:")
        for title in args.titles:
            print(f"  - {title}")
        print()

        articles = fetcher.fetch_articles_from_list(
            titles=args.titles,
            delay_seconds=args.delay,
        )

    else:
        # Fetch priority articles
        max_articles = args.max_articles or len(
            fetcher.config.get("priority_articles", [])
        )

        logger.info(f"Fetching up to {max_articles} priority articles")
        print(f"Fetching up to {max_articles} priority articles")
        print(f"Delay between requests: {args.delay}s")
        print(f"Skip existing: {skip_existing}")
        print()

        articles = fetcher.fetch_priority_articles(
            max_articles=max_articles,
            delay_seconds=args.delay,
        )

    # Print results
    print("\n" + "=" * 70)
    print("Download Complete")
    print("=" * 70)
    print(f"Successfully fetched: {len(articles)} articles")
    print()

    # Show statistics
    print_stats(fetcher)

    # Show sample of downloaded articles
    if articles:
        print("\n" + "=" * 70)
        print("Sample of Downloaded Articles")
        print("=" * 70)

        for article in articles[:5]:
            print(f"\nTitle:      {article['title']}")
            print(f"URL:        {article['url']}")
            print(f"Word Count: {article['word_count']:,}")
            print(f"Summary:    {article['summary'][:150]}...")

        if len(articles) > 5:
            print(f"\n... and {len(articles) - 5} more articles")

    print("\n" + "=" * 70)
    logger.info("Download process completed successfully")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        logger.warning("Download interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
