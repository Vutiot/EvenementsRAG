#!/usr/bin/env python3
"""
Download 10,000 WW2-related Wikipedia articles using category traversal.

This script uses Wikipedia's category system to find and download articles.
It starts with main WW2 categories and recursively explores subcategories.
"""

import json
import sys
import time
from pathlib import Path
from typing import Set, List
import wikipediaapi

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WikipediaCategoryFetcher:
    """Fetch Wikipedia articles from categories."""

    def __init__(self, output_dir: Path, language: str = "en"):
        """
        Initialize the category fetcher.

        Args:
            output_dir: Directory to save articles
            language: Wikipedia language code
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Wikipedia API with user agent
        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            user_agent='EvenementsRAG/1.0 (https://github.com/yourusername/EvenementsRAG)'
        )

        self.downloaded_titles: Set[str] = set()
        self.failed_titles: Set[str] = set()

        logger.info(f"WikipediaCategoryFetcher initialized, output: {output_dir}")

    def get_category_members(self, category_name: str, max_depth: int = 2) -> Set[str]:
        """
        Recursively get all article titles from a category and its subcategories.

        Args:
            category_name: Name of the category (e.g., "World War II")
            max_depth: Maximum depth to traverse subcategories

        Returns:
            Set of article titles
        """
        titles = set()

        category = self.wiki.page(f"Category:{category_name}")

        if not category.exists():
            logger.warning(f"Category does not exist: {category_name}")
            return titles

        logger.info(f"Exploring category: {category_name} (depth {max_depth})")

        # Get all members of this category
        for member_title, member in category.categorymembers.items():
            if member.ns == wikipediaapi.Namespace.MAIN:
                # This is an article
                titles.add(member_title)
            elif member.ns == wikipediaapi.Namespace.CATEGORY and max_depth > 0:
                # This is a subcategory - recurse
                subcategory_name = member_title.replace("Category:", "")
                subtitles = self.get_category_members(subcategory_name, max_depth - 1)
                titles.update(subtitles)

        logger.info(f"Found {len(titles)} articles in {category_name}")
        return titles

    def download_article(self, title: str) -> bool:
        """
        Download a single Wikipedia article.

        Args:
            title: Article title

        Returns:
            True if successful, False otherwise
        """
        try:
            page = self.wiki.page(title)

            if not page.exists():
                logger.warning(f"Article does not exist: {title}")
                return False

            # Get article data
            article_data = {
                "title": page.title,
                "pageid": page.pageid,
                "url": page.fullurl,
                "content": page.text,
                "summary": page.summary,
                "categories": list(page.categories.keys()),
                "links": list(page.links.keys())[:100],  # Limit links
                "word_count": len(page.text.split()),
                "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Save to file
            filename = self._sanitize_filename(page.title) + ".json"
            filepath = self.output_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(article_data, f, indent=2, ensure_ascii=False)

            self.downloaded_titles.add(title)
            return True

        except Exception as e:
            logger.error(f"Failed to download {title}: {e}")
            self.failed_titles.add(title)
            return False

    def _sanitize_filename(self, title: str) -> str:
        """Convert article title to safe filename."""
        # Replace problematic characters
        safe = title.replace("/", "_").replace("\\", "_")
        safe = safe.replace(":", "_").replace("*", "_")
        safe = safe.replace("?", "_").replace('"', "_")
        safe = safe.replace("<", "_").replace(">", "_")
        safe = safe.replace("|", "_")
        return safe[:200]  # Limit length

    def download_articles(
        self,
        titles: List[str],
        max_articles: int = 10000,
        delay_seconds: float = 0.5,
    ) -> int:
        """
        Download multiple articles with rate limiting.

        Args:
            titles: List of article titles to download
            max_articles: Maximum number to download
            delay_seconds: Delay between downloads

        Returns:
            Number of articles successfully downloaded
        """
        print(f"Downloading up to {max_articles} articles...")
        print(f"Total candidates: {len(titles)}")
        print()

        count = 0
        for i, title in enumerate(titles, 1):
            if count >= max_articles:
                break

            # Skip if already exists
            filename = self._sanitize_filename(title) + ".json"
            filepath = self.output_dir / filename
            if filepath.exists():
                logger.debug(f"Skipping existing: {title}")
                continue

            # Download
            if self.download_article(title):
                count += 1

                if count % 100 == 0:
                    print(f"  Downloaded {count}/{max_articles} articles...")

            # Rate limiting
            time.sleep(delay_seconds)

        print()
        print(f"✓ Successfully downloaded {count} articles")
        print(f"  Failed: {len(self.failed_titles)}")

        return count


def main():
    print("=" * 70)
    print("Download 10,000 WW2-Related Wikipedia Articles")
    print("=" * 70)
    print()

    output_dir = settings.DATA_DIR / "raw" / "wikipedia_articles_10000"

    # WW2 categories to explore
    # These are comprehensive categories that cover different aspects of WW2
    ww2_categories = [
        "World War II",
        "Battles of World War II",
        "Military operations of World War II",
        "World War II occupied territories",
        "The Holocaust",
        "World War II by country",
        "World War II military equipment",
        "Military history of World War II",
        "Axis powers",
        "Allies of World War II",
        "World War II conferences",
        "World War II crimes",
        "World War II resistance movements",
        "World War II prisoners of war",
        "World War II naval operations",
        "World War II aerial operations",
        "Strategic bombing during World War II",
        "Nazi Germany",
        "Empire of Japan",
        "Fascist Italy",
        "World War II sites",
    ]

    print("Target categories:")
    for cat in ww2_categories:
        print(f"  - {cat}")
    print()

    # Initialize fetcher
    fetcher = WikipediaCategoryFetcher(output_dir)

    # Collect article titles from categories
    print("Step 1: Collecting article titles from categories...")
    print("This may take several minutes...")
    print()

    all_titles = set()
    for category in ww2_categories:
        print(f"Exploring: {category}")
        titles = fetcher.get_category_members(category, max_depth=1)
        all_titles.update(titles)
        print(f"  Found {len(titles)} articles, total unique: {len(all_titles)}")
        time.sleep(1)  # Be nice to Wikipedia

    print()
    print(f"✓ Collected {len(all_titles)} unique article titles")
    print()

    # Download articles
    print("=" * 70)
    print("Step 2: Downloading Articles")
    print("=" * 70)
    print()

    # Convert to list for downloading
    titles_list = sorted(list(all_titles))

    # Download with rate limiting
    downloaded = fetcher.download_articles(
        titles=titles_list,
        max_articles=10000,
        delay_seconds=0.5,  # 0.5 second delay = ~2 articles/sec
    )

    # Print summary
    print()
    print("=" * 70)
    print("Download Complete")
    print("=" * 70)
    print(f"Total articles downloaded: {downloaded}")
    print(f"Output directory: {output_dir}")
    print(f"Total file count: {len(list(output_dir.glob('*.json')))}")
    print()

    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.json"))
    print(f"Total size: {total_size / (1024**2):.2f} MB")
    print()

    print("Next steps:")
    print("  1. Index articles: python scripts/index_and_generate_questions.py")
    print("  2. Run Phase 1 evaluation")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
