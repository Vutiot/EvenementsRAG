#!/usr/bin/env python3
"""
Scrape 10,000+ WW2-related Wikipedia articles from scratch.

Combines ALL categories from both download_10000_articles.py and
download_more_articles.py (40+ categories), uses max_depth=2 from the
start, and loops until the output directory contains >= 10,000 JSON files.

Supports resume: already-downloaded articles are detected on init and skipped.
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

# Union of all categories from both existing scripts + new additions
WW2_CATEGORIES = [
    # Core categories (from download_10000_articles.py)
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
    # Extended categories (from download_more_articles.py)
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
    # New additions to guarantee 10,000+ unique candidates at depth=2
    "World War II aerial warfare",
    "Women in World War II",
    "World War II leadership",
]


class WikipediaCategoryFetcher:
    """Fetch Wikipedia articles from categories with resume support."""

    def __init__(self, output_dir: Path, language: str = "en"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            user_agent="EvenementsRAG/1.0 (https://github.com/yourusername/EvenementsRAG)",
        )

        self.downloaded_titles: Set[str] = set()
        self.failed_titles: Set[str] = set()

        # Load already-downloaded articles for resume support
        existing_files = list(output_dir.glob("*.json"))
        for filepath in existing_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    title = data.get("title", "")
                    if title:
                        self.downloaded_titles.add(title)
            except Exception:
                pass

        logger.info(
            f"WikipediaCategoryFetcher initialized. "
            f"Already have {len(self.downloaded_titles)} articles in {output_dir}"
        )

    def get_category_members(
        self, category_name: str, max_depth: int = 2
    ) -> Set[str]:
        """
        Recursively get all article titles from a category and its subcategories.

        Args:
            category_name: Category name (e.g. "World War II")
            max_depth: Maximum subcategory traversal depth

        Returns:
            Set of article titles found
        """
        titles: Set[str] = set()

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
                subtitles = self.get_category_members(
                    subcategory_name, max_depth - 1
                )
                titles.update(subtitles)

        logger.info(f"Found {len(titles)} articles in {category_name}")
        return titles

    def download_article(self, title: str) -> bool:
        """
        Download a single Wikipedia article and save as JSON.

        Returns:
            True if successfully downloaded, False otherwise
        """
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
            logger.error(f"Failed to download '{title}': {e}")
            self.failed_titles.add(title)
            return False

    def _sanitize_filename(self, title: str) -> str:
        """Convert article title to a safe filename."""
        safe = title.replace("/", "_").replace("\\", "_")
        safe = safe.replace(":", "_").replace("*", "_")
        safe = safe.replace("?", "_").replace('"', "_")
        safe = safe.replace("<", "_").replace(">", "_")
        safe = safe.replace("|", "_")
        return safe[:200]

    def download_articles(
        self,
        titles: List[str],
        target: int = 10000,
        delay_seconds: float = 0.5,
    ) -> int:
        """
        Download articles with rate limiting until target is reached.

        Args:
            titles: Candidate article titles
            target: Stop once this many total articles are in the output dir
            delay_seconds: Delay between API calls

        Returns:
            Number of newly downloaded articles in this session
        """
        print(f"Downloading articles (target: {target})...")
        print(f"Candidates: {len(titles)}, already have: {len(self.downloaded_titles)}")
        print()

        new_count = 0
        skipped = 0

        for title in titles:
            # Stop when target is reached
            if len(self.downloaded_titles) >= target:
                break

            # Skip duplicates
            if title in self.downloaded_titles:
                skipped += 1
                continue

            filename = self._sanitize_filename(title) + ".json"
            if (self.output_dir / filename).exists():
                skipped += 1
                continue

            if self.download_article(title):
                new_count += 1
                total = len(self.downloaded_titles)
                if new_count % 100 == 0:
                    pct = total / target * 100
                    print(
                        f"  Progress: {total}/{target} articles "
                        f"({pct:.1f}%) — +{new_count} this session"
                    )

            time.sleep(delay_seconds)

        print()
        print(f"  Downloaded {new_count} new articles this session")
        print(f"  Skipped {skipped} (already present)")
        print(f"  Total now: {len(self.downloaded_titles)}")
        print(f"  Failed: {len(self.failed_titles)}")
        return new_count


def main():
    print("=" * 70)
    print("Scrape 10,000+ WW2-Related Wikipedia Articles")
    print("=" * 70)
    print()

    output_dir = settings.DATA_DIR / "raw" / "wikipedia_articles_10000"
    target = 10000
    max_depth = 2
    delay = 0.5  # seconds between article downloads

    print(f"Output directory : {output_dir}")
    print(f"Target articles  : {target}")
    print(f"Category depth   : {max_depth}")
    print(f"Categories       : {len(WW2_CATEGORIES)}")
    print()

    # Initialize fetcher (loads existing articles for resume)
    fetcher = WikipediaCategoryFetcher(output_dir)

    current_count = len(fetcher.downloaded_titles)
    if current_count >= target:
        print(f"Already have {current_count} articles — target met, nothing to do.")
        return

    print(f"Currently have {current_count} articles, need {target - current_count} more.")
    print()

    # Step 1: Collect candidate titles from all categories (depth=2)
    print("=" * 70)
    print("Step 1: Collecting article candidates from categories")
    print("=" * 70)
    print("This may take 10–20 minutes (Wikipedia API rate limits apply)...")
    print()

    all_titles: Set[str] = set()
    for i, category in enumerate(WW2_CATEGORIES, 1):
        print(f"[{i}/{len(WW2_CATEGORIES)}] Exploring: {category}")
        titles = fetcher.get_category_members(category, max_depth=max_depth)
        new = titles - all_titles
        all_titles.update(titles)
        print(f"  +{len(new)} new candidates (total unique: {len(all_titles)})")
        time.sleep(1)  # Be courteous to Wikipedia API between categories

    print()
    print(f"Total unique article candidates: {len(all_titles)}")
    print()

    if len(all_titles) < target:
        print(
            f"WARNING: Only {len(all_titles)} candidates found — "
            f"may not reach {target} articles."
        )
        print("Consider adding more categories or increasing max_depth.")
        print()

    # Step 2: Download articles
    print("=" * 70)
    print("Step 2: Downloading Articles")
    print("=" * 70)
    print()

    titles_list = sorted(all_titles)

    fetcher.download_articles(
        titles=titles_list,
        target=target,
        delay_seconds=delay,
    )

    # Final summary
    final_count = len(list(output_dir.glob("*.json")))
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.json"))

    print()
    print("=" * 70)
    print("Download Complete")
    print("=" * 70)
    print(f"Total files in output dir : {final_count}")
    print(f"Total size                : {total_size / (1024**2):.1f} MB")
    print(f"Output directory          : {output_dir}")
    print()

    if final_count >= target:
        print(f"Target reached ({final_count} >= {target}). Ready for indexing.")
    else:
        pct = final_count / target * 100
        print(f"Progress: {pct:.1f}% of target ({final_count}/{target})")
        print("Run this script again to continue (already-downloaded files are skipped).")

    print()
    print("Next steps:")
    print("  1. Generate eval questions: python scripts/generate_200_eval_questions.py")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user (progress is saved, re-run to resume)")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
