#!/usr/bin/env python3
"""
Fetch Wikipedia Articles for WW2 RAG Dataset

This script fetches Wikipedia articles related to World War II using the Wikipedia API.
Targets 1000 articles with rate limiting and progress tracking.
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Set
from datetime import datetime
import requests
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings


class WikipediaArticleFetcher:
    """Fetch Wikipedia articles using the API."""

    def __init__(self, target_count: int = 1000):
        self.target_count = target_count
        self.base_url = "https://en.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WW2-RAG-Research/1.0 (Educational Research Project)'
        })

        # WW2 seed categories and topics
        self.seed_categories = [
            "Category:World War II",
            "Category:Battles of World War II",
            "Category:World War II operations and battles",
            "Category:Military history of World War II",
            "Category:World War II by country",
            "Category:The Holocaust",
            "Category:World War II conferences",
            "Category:World War II military equipment",
            "Category:World War II weapons",
            "Category:World War II aircraft",
            "Category:World War II ships",
            "Category:World War II people",
            "Category:World War II commanders",
            "Category:World War II political leaders",
            "Category:Axis powers",
            "Category:Allies of World War II",
            "Category:World War II strategic bombing",
            "Category:World War II resistance movements",
            "Category:World War II crimes",
            "Category:World War II prisoners of war",
        ]

    def get_category_members(self, category: str, limit: int = 500) -> List[str]:
        """
        Get article titles from a Wikipedia category.

        Args:
            category: Category name (e.g., "Category:World War II")
            limit: Maximum number of members to fetch

        Returns:
            List of article titles
        """
        articles = []
        continue_token = None

        while len(articles) < limit:
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': category,
                'cmlimit': min(500, limit - len(articles)),
                'cmtype': 'page',  # Only pages, not subcategories
                'format': 'json'
            }

            if continue_token:
                params['cmcontinue'] = continue_token

            try:
                response = self.session.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                members = data.get('query', {}).get('categorymembers', [])
                articles.extend([m['title'] for m in members])

                # Check for continuation
                if 'continue' in data and len(articles) < limit:
                    continue_token = data['continue']['cmcontinue']
                else:
                    break

                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                print(f"Error fetching category {category}: {e}")
                break

        return articles[:limit]

    def search_articles(self, query: str, limit: int = 100) -> List[str]:
        """
        Search for articles using Wikipedia search API.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of article titles
        """
        articles = []
        offset = 0

        while len(articles) < limit:
            params = {
                'action': 'query',
                'list': 'search',
                'srsearch': query,
                'srlimit': min(50, limit - len(articles)),
                'sroffset': offset,
                'format': 'json'
            }

            try:
                response = self.session.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                results = data.get('query', {}).get('search', [])
                if not results:
                    break

                articles.extend([r['title'] for r in results])
                offset += len(results)

                time.sleep(0.1)

            except Exception as e:
                print(f"Error searching '{query}': {e}")
                break

        return articles[:limit]

    def get_article_content(self, title: str) -> Dict:
        """
        Fetch full article content and metadata.

        Args:
            title: Article title

        Returns:
            Article data dict or None if failed
        """
        params = {
            'action': 'query',
            'titles': title,
            'prop': 'extracts|info|categories',
            'explaintext': True,  # Plain text, not HTML
            'exsectionformat': 'plain',
            'inprop': 'url',
            'cllimit': 'max',
            'format': 'json'
        }

        try:
            response = self.session.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            pages = data.get('query', {}).get('pages', {})
            if not pages:
                return None

            # Get the first (and only) page
            page = list(pages.values())[0]

            # Skip if page doesn't exist or is redirect
            if 'missing' in page or 'redirect' in page:
                return None

            # Extract content
            article = {
                'title': page.get('title'),
                'pageid': page.get('pageid'),
                'url': page.get('fullurl'),
                'text': page.get('extract', ''),
                'categories': [c['title'] for c in page.get('categories', [])],
                'fetched_at': datetime.now().isoformat(),
            }

            # Skip if content is too short (likely stub or disambiguation)
            if len(article['text']) < 500:
                return None

            return article

        except Exception as e:
            print(f"Error fetching article '{title}': {e}")
            return None

    def collect_article_titles(self) -> List[str]:
        """
        Collect article titles from categories and searches.

        Returns:
            List of unique article titles
        """
        print("Collecting article titles from categories and searches...")
        titles = set()

        # 1. Get articles from main categories
        print(f"\nFetching from {len(self.seed_categories)} categories...")
        for category in tqdm(self.seed_categories, desc="Categories"):
            members = self.get_category_members(category, limit=100)
            titles.update(members)

            if len(titles) >= self.target_count * 2:
                break

        print(f"Got {len(titles)} articles from categories")

        # 2. Search for specific topics to fill gaps
        search_queries = [
            "World War II battle",
            "World War 2 campaign",
            "WW2 operation",
            "Second World War military",
            "1939-1945 war",
            "Nazi Germany military",
            "Allied forces WWII",
            "Pacific War",
            "European theatre World War II",
            "North Africa campaign",
            "Eastern Front World War II",
            "D-Day Normandy",
            "Pearl Harbor attack",
            "Battle of Britain",
            "Stalingrad",
            "Holocaust",
        ]

        print(f"\nSearching {len(search_queries)} queries...")
        for query in tqdm(search_queries, desc="Searches"):
            results = self.search_articles(query, limit=50)
            titles.update(results)

            if len(titles) >= self.target_count * 2:
                break

        print(f"Total unique titles collected: {len(titles)}")
        return list(titles)

    def fetch_articles(self, output_dir: Path, resume: bool = True) -> Dict:
        """
        Fetch articles and save to directory.

        Args:
            output_dir: Directory to save articles
            resume: Whether to resume from existing files

        Returns:
            Stats dict
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect titles
        all_titles = self.collect_article_titles()

        # Check existing files
        existing_files = set()
        if resume:
            existing_files = {f.stem for f in output_dir.glob("*.json")}
            print(f"Found {len(existing_files)} existing articles")

        # Filter out existing
        titles_to_fetch = [t for t in all_titles if self._safe_filename(t) not in existing_files]
        print(f"Need to fetch {len(titles_to_fetch)} new articles")

        # Limit to target count
        titles_to_fetch = titles_to_fetch[:self.target_count]

        # Fetch articles
        print(f"\nFetching {len(titles_to_fetch)} articles...")
        print(f"Target: {self.target_count} articles total")
        print()

        stats = {
            'attempted': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': datetime.now().isoformat(),
        }

        for title in tqdm(titles_to_fetch, desc="Fetching articles"):
            stats['attempted'] += 1

            # Fetch content
            article = self.get_article_content(title)

            if article:
                # Save to file
                filename = self._safe_filename(article['title']) + '.json'
                filepath = output_dir / filename

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(article, f, indent=2, ensure_ascii=False)

                stats['successful'] += 1
            else:
                stats['failed'] += 1

            # Rate limiting
            time.sleep(0.15)  # ~6-7 requests per second

            # Progress update every 50 articles
            if stats['attempted'] % 50 == 0:
                success_rate = stats['successful'] / stats['attempted'] * 100
                print(f"\nProgress: {stats['successful']} successful ({success_rate:.1f}%)")

        stats['end_time'] = datetime.now().isoformat()
        stats['total_existing'] = len(existing_files)

        return stats

    def _safe_filename(self, title: str) -> str:
        """Convert article title to safe filename."""
        # Replace problematic characters
        safe = title.replace('/', '_').replace('\\', '_').replace(':', '_')
        safe = safe.replace('?', '_').replace('*', '_').replace('"', '_')
        safe = safe.replace('<', '_').replace('>', '_').replace('|', '_')
        return safe


def main():
    print("=" * 70)
    print("Wikipedia Article Fetcher - WW2 Dataset")
    print("=" * 70)
    print()
    print("Target: 1000 Wikipedia articles")
    print("Topic: World War II")
    print()

    # Configuration
    output_dir = settings.DATA_DIR / "raw" / "wikipedia_articles_1000"
    target_count = 1000

    # Initialize fetcher
    fetcher = WikipediaArticleFetcher(target_count=target_count)

    # Confirm
    print(f"Output directory: {output_dir}")
    print()
    response = input("Continue? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return

    print()

    # Fetch articles
    stats = fetcher.fetch_articles(output_dir, resume=True)

    # Print summary
    print()
    print("=" * 70)
    print("Fetching Complete!")
    print("=" * 70)
    print()
    print(f"Attempted:       {stats['attempted']}")
    print(f"Successful:      {stats['successful']}")
    print(f"Failed:          {stats['failed']}")
    print(f"Existing:        {stats['total_existing']}")
    print(f"Total articles:  {stats['successful'] + stats['total_existing']}")
    print()
    print(f"Saved to: {output_dir}")
    print()

    # Calculate success rate
    if stats['attempted'] > 0:
        success_rate = stats['successful'] / stats['attempted'] * 100
        print(f"Success rate: {success_rate:.1f}%")

    total = stats['successful'] + stats['total_existing']
    if total >= target_count:
        print(f"✓ Target reached: {total}/{target_count} articles")
    else:
        print(f"⚠ Still need {target_count - total} more articles")
    print()


if __name__ == "__main__":
    main()
