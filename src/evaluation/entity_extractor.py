"""LLM-based named entity extraction for entity-level retrieval metrics.

Uses the same NVIDIA API (OpenAI-compatible) as the rest of the project.
Follows RAGAS's ContextEntityRecall extraction approach but as a standalone
utility. Supports caching and rate limiting.

Usage:
    from src.evaluation.entity_extractor import EntityExtractor

    extractor = EntityExtractor()
    entities = extractor.extract_entities("Winston Churchill led Britain...")
    # {'winston churchill', 'britain'}
"""

import hashlib
import json
import time
from typing import Optional

import openai

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


_ENTITY_EXTRACTION_PROMPT = """\
Extract all named entities from the following text.
Entity types to extract: PERSON, GPE (geopolitical entity), ORG (organization), \
DATE, EVENT, LOC (location).

Return ONLY a JSON array of strings — no explanation, no markdown.
Example: ["Winston Churchill", "Battle of Britain", "London", "September 1940"]

Text:
{text}

JSON array:"""


class EntityExtractor:
    """LLM-based named entity extraction with caching and rate limiting."""

    def __init__(
        self,
        model: str = "nvidia/nemotron-3-nano-30b-a3b",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        rate_limit_seconds: float = 1.5,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._rate_limit_seconds = rate_limit_seconds
        self._max_retries = max_retries
        self._cache: dict[str, set[str]] = {}
        self._last_call_time: float = 0.0

        self._client = openai.OpenAI(
            api_key=api_key or settings.NVIDIA_API_KEY,
            base_url=base_url or settings.NVIDIA_BASE_URL,
        )

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _wait_rate_limit(self) -> None:
        elapsed = time.time() - self._last_call_time
        if elapsed < self._rate_limit_seconds:
            time.sleep(self._rate_limit_seconds - elapsed)

    def _parse_entities(self, raw: str) -> set[str]:
        """Parse LLM response into a set of normalized entity strings."""
        raw = raw.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines).strip()

        try:
            entities = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse entity extraction response: %s", raw[:200])
            return set()

        if not isinstance(entities, list):
            logger.warning("Entity extraction returned non-list: %s", type(entities))
            return set()

        return {str(e).strip().lower() for e in entities if isinstance(e, str) and e.strip()}

    def extract_entities(self, text: str) -> set[str]:
        """Extract named entities from text using LLM.

        Results are cached by text hash to avoid duplicate API calls.
        """
        if not text or not text.strip():
            return set()

        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]

        prompt = _ENTITY_EXTRACTION_PROMPT.format(text=text[:4000])

        for attempt in range(self._max_retries):
            try:
                self._wait_rate_limit()
                self._last_call_time = time.time()

                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1024,
                )
                raw = response.choices[0].message.content or ""
                entities = self._parse_entities(raw)
                self._cache[key] = entities
                return entities

            except Exception as exc:
                if attempt == self._max_retries - 1:
                    logger.error("Entity extraction failed after %d retries: %s", self._max_retries, exc)
                    self._cache[key] = set()
                    return set()
                wait = 10 * (2 ** attempt)
                logger.warning("Entity extraction attempt %d failed, retrying in %ds: %s", attempt + 1, wait, exc)
                time.sleep(wait)

        return set()  # unreachable but satisfies type checker

    def extract_entities_batch(self, texts: list[str]) -> list[set[str]]:
        """Extract entities from multiple texts with rate limiting.

        Reuses cache so repeated texts skip the API call.
        """
        results: list[set[str]] = []
        for text in texts:
            results.append(self.extract_entities(text))
        return results

    def clear_cache(self) -> None:
        """Clear the entity extraction cache."""
        self._cache.clear()
