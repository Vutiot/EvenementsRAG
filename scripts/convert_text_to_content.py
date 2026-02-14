#!/usr/bin/env python3
"""
Convert 'text' field to 'content' field in Wikipedia articles.

The Wikipedia fetcher saved articles with a 'text' field, but the chunker
expects a 'content' field. This script converts all articles in-place.

Usage:
    python scripts/convert_text_to_content.py
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings


def convert_article(article_path: Path) -> bool:
    """
    Convert 'text' field to 'content' in a single article.

    Args:
        article_path: Path to article JSON file

    Returns:
        True if converted, False if failed
    """
    try:
        # Load article
        with open(article_path, 'r', encoding='utf-8') as f:
            article = json.load(f)

        # Check if conversion needed
        if 'text' in article and 'content' not in article:
            # Rename field
            article['content'] = article.pop('text')

            # Save back
            with open(article_path, 'w', encoding='utf-8') as f:
                json.dump(article, f, indent=2, ensure_ascii=False)

            return True

        return False

    except Exception as e:
        print(f"Error converting {article_path.name}: {e}")
        return False


def main():
    """Main function."""
    print("=" * 70)
    print("Convert Wikipedia Articles: 'text' → 'content'")
    print("=" * 70)
    print()

    # Get articles directory
    articles_dir = settings.DATA_DIR / "raw" / "wikipedia_articles_1000"

    if not articles_dir.exists():
        print(f"❌ Directory not found: {articles_dir}")
        sys.exit(1)

    # Get all JSON files
    json_files = list(articles_dir.glob("*.json"))

    if not json_files:
        print(f"❌ No JSON files found in {articles_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} articles in {articles_dir}")
    print()

    # Convert all articles
    converted = 0
    failed = 0

    for article_file in tqdm(json_files, desc="Converting articles"):
        if convert_article(article_file):
            converted += 1
        else:
            failed += 1

    # Print summary
    print()
    print("=" * 70)
    print("Conversion Complete")
    print("=" * 70)
    print()
    print(f"✓ Articles converted:  {converted}")
    print(f"  Already correct:     {len(json_files) - converted - failed}")
    print(f"  Failed:              {failed}")
    print()

    if converted > 0:
        print("✓ All articles now have 'content' field instead of 'text'")
    else:
        print("✓ All articles already had correct field name")
    print()


if __name__ == "__main__":
    main()
