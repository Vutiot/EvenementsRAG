#!/usr/bin/env python3
"""
Download more WW2 articles to reach ~10,000 total.

This script continues from the existing 6,844 articles by:
- Using deeper category traversal (max_depth=2)
- Adding more WW2-related categories
- Skipping already downloaded articles
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
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Wikipedia API with user agent
        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            user_agent='EvenementsRAG/1.0 (https://github.com/yourusername/EvenementsRAG)'
        )

        self.downloaded_titles: Set[str] = set()
        self.failed_titles: Set[str] = set()

        # Load already downloaded articles
        for filepath in output_dir.glob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.downloaded_titles.add(data.get('title', ''))
            except:
                pass

        logger.info(f"WikipediaCategoryFetcher initialized, already have {len(self.downloaded_titles)} articles")

    def get_category_members(self, category_name: str, max_depth: int = 2) -> Set[str]:
        """Recursively get all article titles from a category."""
        titles = set()

        category = self.wiki.page(f"Category:{category_name}")

        if not category.exists():
            logger.warning(f"Category does not exist: {category_name}")
            return titles

        logger.info(f"Exploring category: {category_name} (depth {max_depth})")

        for member_title, member in category.categorymembers.items():
            if member.ns == wikipediaapi.Namespace.MAIN:
                titles.add(member_title)
            elif member.ns == wikipediaapi.Namespace.CATEGORY and max_depth > 0:
                subcategory_name = member_title.replace("Category:", "")
                subtitles = self.get_category_members(subcategory_name, max_depth - 1)
                titles.update(subtitles)

        logger.info(f"Found {len(titles)} articles in {category_name}")
        return titles

    def download_article(self, title: str) -> bool:
        """Download a single Wikipedia article."""
        try:
            page = self.wiki.page(title)

            if not page.exists():
                logger.warning(f"Article does not exist: {title}")
                return False

            article_data = {
                "title": page.title,
                "pageid": page.pageid,
                "url": page.fullurl,
                "content": page.text,
                "summary": page.summary,
                "categories": list(page.categories.keys()),
                "links": list(page.links.keys())[:100],
                "word_count": len(page.text.split()),
                "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

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
        safe = title.replace("/", "_").replace("\\", "_")
        safe = safe.replace(":", "_").replace("*", "_")
        safe = safe.replace("?", "_").replace('"', "_")
        safe = safe.replace("<", "_").replace(">", "_")
        safe = safe.replace("|", "_")
        return safe[:200]

    def download_articles(
        self,
        titles: List[str],
        max_articles: int = 10000,
        delay_seconds: float = 0.5,
    ) -> int:
        """Download multiple articles with rate limiting."""
        print(f"Downloading up to {max_articles} articles...")
        print(f"Total candidates: {len(titles)}")
        print(f"Already have: {len(self.downloaded_titles)}")
        print()

        count = 0
        skipped = 0

        for i, title in enumerate(titles, 1):
            if len(self.downloaded_titles) >= max_articles:
                break

            # Skip if already exists
            if title in self.downloaded_titles:
                skipped += 1
                continue

            filename = self._sanitize_filename(title) + ".json"
            filepath = self.output_dir / filename
            if filepath.exists():
                skipped += 1
                continue

            # Download
            if self.download_article(title):
                count += 1

                if count % 100 == 0:
                    total = len(self.downloaded_titles)
                    print(f"  Progress: {total}/{max_articles} articles ({total/max_articles*100:.1f}%)")

            # Rate limiting
            time.sleep(delay_seconds)

        print()
        print(f"✓ Downloaded {count} new articles")
        print(f"  Skipped: {skipped} (already had)")
        print(f"  Total now: {len(self.downloaded_titles)}")

        return count


def main():
    print("=" * 70)
    print("Download More WW2 Articles (Continue to ~10,000)")
    print("=" * 70)
    print()

    output_dir = settings.DATA_DIR / "raw" / "wikipedia_articles_10000"

    # Expanded WW2 categories with deeper traversal
    ww2_categories = [
        # Main categories (will go 2 levels deep)
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

        # Additional categories for more coverage
        "World War II casualties",
        "World War II weapons",
        "World War II tanks",
        "World War II aircraft",
        "World War II ships",
        "World War II submarines",
        "World War II military units",
        "World War II battles by country",
        "World War II propaganda",
        "World War II espionage",
        "World War II diplomacy",
        "Home fronts during World War II",
        "Economic history of World War II",
        "Technology during World War II",
        "Medicine in World War II",
        "World War II memorials",
        "Aftermath of World War II",
    ]

    print("Exploring categories (depth=2 for more coverage):")
    for cat in ww2_categories[:10]:
        print(f"  - {cat}")
    print(f"  ... and {len(ww2_categories)-10} more")
    print()

    # Initialize fetcher
    fetcher = WikipediaCategoryFetcher(output_dir)

    current_count = len(fetcher.downloaded_titles)
    target = 10000
    needed = target - current_count

    print(f"Current articles: {current_count}")
    print(f"Target articles: {target}")
    print(f"Need to download: ~{needed} more")
    print()

    # Collect article titles from categories (depth=2)
    print("Step 1: Collecting article titles from categories (depth=2)...")
    print("This may take 5-10 minutes...")
    print()

    all_titles = set()
    for category in ww2_categories:
        print(f"Exploring: {category}")
        titles = fetcher.get_category_members(category, max_depth=2)  # Deeper traversal
        all_titles.update(titles)
        print(f"  Found {len(titles)} articles, total unique: {len(all_titles)}")
        time.sleep(1)

    print()
    print(f"✓ Collected {len(all_titles)} unique article candidates")
    print()

    # Download articles
    print("=" * 70)
    print("Step 2: Downloading Articles")
    print("=" * 70)
    print()

    titles_list = sorted(list(all_titles))

    # Download with rate limiting
    downloaded = fetcher.download_articles(
        titles=titles_list,
        max_articles=target,
        delay_seconds=0.5,
    )

    # Print summary
    print()
    print("=" * 70)
    print("Download Complete")
    print("=" * 70)
    print(f"New articles downloaded: {downloaded}")
    print(f"Total articles now: {len(fetcher.downloaded_titles)}")
    print(f"Output directory: {output_dir}")
    print(f"Total file count: {len(list(output_dir.glob('*.json')))}")
    print()

    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.json"))
    print(f"Total size: {total_size / (1024**2):.2f} MB")
    print()

    if len(fetcher.downloaded_titles) >= 9500:
        print("🎯 Target reached! Ready for evaluation.")
    else:
        progress = len(fetcher.downloaded_titles) / target * 100
        print(f"Progress: {progress:.1f}% of target")

    print()
    print("Next steps:")
    print("  1. Index articles: python scripts/index_and_generate_questions_10k.py")
    print("  2. Run Phase 1 evaluation: python scripts/run_phase1_10k.py")
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
