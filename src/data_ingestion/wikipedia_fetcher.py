"""
Wikipedia article fetcher for historical events.

Fetches Wikipedia articles based on period configuration (e.g., WW2),
with support for:
- Batch downloading with rate limiting
- Article validation and filtering
- Metadata extraction
- Incremental/resumable downloads
- Caching

Usage:
    from src.data_ingestion.wikipedia_fetcher import WikipediaFetcher

    fetcher = WikipediaFetcher(period_config_path="config/periods/world_war_2.yaml")
    articles = fetcher.fetch_priority_articles(max_articles=50)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import quote

import wikipedia
import yaml
from bs4 import BeautifulSoup

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WikipediaFetcher:
    """Fetches and manages Wikipedia articles for historical events."""

    def __init__(
        self,
        period_config_path: Optional[str] = None,
        output_dir: Optional[Path] = None,
        language: str = "en",
    ):
        """
        Initialize the Wikipedia fetcher.

        Args:
            period_config_path: Path to period YAML config (e.g., world_war_2.yaml)
            output_dir: Directory to save articles (defaults to data/raw/wikipedia_articles)
            language: Wikipedia language code (default: "en")
        """
        self.language = language
        wikipedia.set_lang(self.language)

        # Load period configuration
        self.config = self._load_period_config(period_config_path)
        self.period_name = self.config.get("short_name", "unknown")

        # Set output directory
        self.output_dir = output_dir or (settings.DATA_DIR / "raw" / "wikipedia_articles")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metadata directory
        self.metadata_dir = settings.DATA_DIR / "raw" / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Track fetched articles
        self.fetched_articles: Set[str] = self._load_fetched_articles()

        logger.info(
            f"WikipediaFetcher initialized for period: {self.period_name}",
            extra={"already_fetched": len(self.fetched_articles)},
        )

    def _load_period_config(self, config_path: Optional[str] = None) -> Dict:
        """Load period configuration from YAML file."""
        if config_path is None:
            # Default to WW2 config
            config_path = "config/periods/world_war_2.yaml"

        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {"short_name": "unknown", "priority_articles": []}

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded period config: {config.get('name', 'Unknown')}")
        return config

    def _load_fetched_articles(self) -> Set[str]:
        """Load set of already fetched article titles."""
        metadata_file = self.metadata_dir / f"{self.period_name}_fetched.json"

        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                data = json.load(f)
                return set(data.get("fetched_articles", []))

        return set()

    def _save_fetched_articles(self) -> None:
        """Save list of fetched articles to metadata."""
        metadata_file = self.metadata_dir / f"{self.period_name}_fetched.json"

        data = {
            "period": self.period_name,
            "last_updated": datetime.now().isoformat(),
            "total_fetched": len(self.fetched_articles),
            "fetched_articles": sorted(list(self.fetched_articles)),
        }

        with open(metadata_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved fetched articles metadata: {len(self.fetched_articles)} articles")

    def _sanitize_filename(self, title: str) -> str:
        """Convert article title to safe filename."""
        # Replace problematic characters
        safe = title.replace("/", "_").replace(":", "_").replace(" ", "_")
        # Remove other special characters
        safe = "".join(c for c in safe if c.isalnum() or c in ("_", "-"))
        return safe[:200]  # Limit length

    def _get_article_path(self, title: str) -> Path:
        """Get file path for article."""
        filename = f"{self._sanitize_filename(title)}.json"
        return self.output_dir / filename

    def fetch_article(
        self,
        title: str,
        skip_if_exists: bool = True,
        retry_count: int = 3,
    ) -> Optional[Dict]:
        """
        Fetch a single Wikipedia article.

        Args:
            title: Article title
            skip_if_exists: Skip if already downloaded
            retry_count: Number of retries on failure

        Returns:
            Article data dictionary or None if failed
        """
        # Check if already fetched
        if skip_if_exists and title in self.fetched_articles:
            logger.debug(f"Skipping already fetched article: {title}")
            return None

        # Check if file exists
        article_path = self._get_article_path(title)
        if skip_if_exists and article_path.exists():
            logger.debug(f"Article file already exists: {title}")
            self.fetched_articles.add(title)
            return None

        # Fetch with retries
        for attempt in range(retry_count):
            try:
                logger.info(f"Fetching article: {title} (attempt {attempt + 1}/{retry_count})")

                # Search for page (handles redirects)
                page = wikipedia.page(title, auto_suggest=False)

                # Extract article data
                article_data = {
                    "title": page.title,
                    "original_title": title,
                    "url": page.url,
                    "content": page.content,
                    "summary": page.summary,
                    "categories": page.categories,
                    "links": page.links[:100],  # Limit links
                    "references": page.references[:50],  # Limit references
                    "images": page.images[:10],  # Limit images
                    "fetched_at": datetime.now().isoformat(),
                    "language": self.language,
                    "word_count": len(page.content.split()),
                    "pageid": page.pageid,
                }

                # Validate article
                if not self._validate_article(article_data):
                    logger.warning(f"Article validation failed: {title}")
                    return None

                # Save article
                self._save_article(article_data)

                # Update tracking
                self.fetched_articles.add(title)

                logger.info(
                    f"Successfully fetched: {page.title}",
                    extra={"words": article_data["word_count"]},
                )

                return article_data

            except wikipedia.exceptions.DisambiguationError as e:
                logger.warning(
                    f"Disambiguation page: {title}",
                    extra={"options": e.options[:5]},
                )
                return None

            except wikipedia.exceptions.PageError:
                logger.warning(f"Page not found: {title}")
                return None

            except Exception as e:
                logger.error(
                    f"Error fetching {title} (attempt {attempt + 1}): {str(e)}",
                    exc_info=attempt == retry_count - 1,
                )
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return None

        return None

    def _validate_article(self, article_data: Dict) -> bool:
        """
        Validate article meets requirements.

        Args:
            article_data: Article dictionary

        Returns:
            True if valid, False otherwise
        """
        # Get validation rules from config
        filters = self.config.get("filters", {})

        # Check minimum length
        min_length = filters.get("min_article_length", 500)
        word_count = article_data.get("word_count", 0)

        if word_count < min_length:
            logger.debug(
                f"Article too short: {article_data['title']} ({word_count} < {min_length})"
            )
            return False

        # Check maximum length
        max_length = filters.get("max_article_length", 50000)
        if word_count > max_length:
            logger.debug(
                f"Article too long: {article_data['title']} ({word_count} > {max_length})"
            )
            return False

        # Check for disambiguation
        if filters.get("exclude_disambiguation", True):
            categories = article_data.get("categories", [])
            if any("disambiguation" in cat.lower() for cat in categories):
                logger.debug(f"Excluding disambiguation: {article_data['title']}")
                return False

        return True

    def _save_article(self, article_data: Dict) -> None:
        """Save article to JSON file."""
        article_path = self._get_article_path(article_data["title"])

        with open(article_path, "w", encoding="utf-8") as f:
            json.dump(article_data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Saved article to: {article_path}")

    def fetch_priority_articles(
        self,
        max_articles: Optional[int] = None,
        delay_seconds: float = 1.0,
    ) -> List[Dict]:
        """
        Fetch priority articles from config.

        Args:
            max_articles: Maximum number of articles to fetch (None = all)
            delay_seconds: Delay between requests (respect Wikipedia)

        Returns:
            List of fetched article data
        """
        priority_list = self.config.get("priority_articles", [])

        if not priority_list:
            logger.warning("No priority articles defined in config")
            return []

        # Limit if specified
        if max_articles:
            priority_list = priority_list[:max_articles]

        logger.info(
            f"Fetching {len(priority_list)} priority articles",
            extra={"delay": delay_seconds},
        )

        articles = []
        success_count = 0
        failure_count = 0

        for i, title in enumerate(priority_list, 1):
            logger.info(f"[{i}/{len(priority_list)}] Processing: {title}")

            # Fetch article
            article_data = self.fetch_article(title)

            if article_data:
                articles.append(article_data)
                success_count += 1
            else:
                failure_count += 1

            # Rate limiting - respect Wikipedia
            if i < len(priority_list):
                time.sleep(delay_seconds)

            # Save progress periodically
            if i % 10 == 0:
                self._save_fetched_articles()

        # Final save
        self._save_fetched_articles()

        logger.info(
            f"Fetching complete: {success_count} succeeded, {failure_count} failed",
            extra={
                "total": len(priority_list),
                "success": success_count,
                "failed": failure_count,
            },
        )

        return articles

    def fetch_articles_from_list(
        self,
        titles: List[str],
        delay_seconds: float = 1.0,
    ) -> List[Dict]:
        """
        Fetch articles from a custom list of titles.

        Args:
            titles: List of article titles
            delay_seconds: Delay between requests

        Returns:
            List of fetched article data
        """
        logger.info(f"Fetching {len(titles)} articles from custom list")

        articles = []
        for i, title in enumerate(titles, 1):
            logger.info(f"[{i}/{len(titles)}] Processing: {title}")

            article_data = self.fetch_article(title)
            if article_data:
                articles.append(article_data)

            if i < len(titles):
                time.sleep(delay_seconds)

        self._save_fetched_articles()
        return articles

    def get_statistics(self) -> Dict:
        """Get fetcher statistics."""
        stats = {
            "period": self.period_name,
            "total_fetched": len(self.fetched_articles),
            "output_directory": str(self.output_dir),
            "config_file": self.config.get("name", "Unknown"),
        }

        # Count actual files
        json_files = list(self.output_dir.glob("*.json"))
        stats["files_on_disk"] = len(json_files)

        # Get total size
        total_size = sum(f.stat().st_size for f in json_files)
        stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)

        return stats

    def load_article(self, title: str) -> Optional[Dict]:
        """
        Load a previously fetched article.

        Args:
            title: Article title

        Returns:
            Article data or None if not found
        """
        article_path = self._get_article_path(title)

        if not article_path.exists():
            logger.warning(f"Article file not found: {title}")
            return None

        with open(article_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_all_articles(self) -> List[Dict]:
        """Load all fetched articles from disk."""
        articles = []

        for article_file in self.output_dir.glob("*.json"):
            try:
                with open(article_file, "r", encoding="utf-8") as f:
                    articles.append(json.load(f))
            except Exception as e:
                logger.error(f"Error loading {article_file}: {e}")

        logger.info(f"Loaded {len(articles)} articles from disk")
        return articles


if __name__ == "__main__":
    # Test the fetcher
    fetcher = WikipediaFetcher()

    # Test single article
    print("\n=== Testing single article fetch ===")
    article = fetcher.fetch_article("D-Day")

    if article:
        print(f"Title: {article['title']}")
        print(f"Words: {article['word_count']}")
        print(f"URL: {article['url']}")
        print(f"Summary: {article['summary'][:200]}...")

    # Show statistics
    print("\n=== Fetcher Statistics ===")
    stats = fetcher.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
