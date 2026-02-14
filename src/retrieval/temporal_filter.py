"""
Temporal Filtering for RAG

Extracts temporal information from queries and applies filters to
prioritize historically relevant chunks.
"""

import re
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TemporalFilter:
    """
    Extract temporal information from queries and create Qdrant filters.

    Handles:
    - Explicit years: "in 1944", "during 1943"
    - Year ranges: "between 1941 and 1945"
    - Relative terms: "early war", "late war"
    - Event-based: "after Pearl Harbor", "before D-Day"
    """

    # WW2 timeframe
    WW2_START = 1939
    WW2_END = 1945

    # Key events and their dates
    KEY_EVENTS = {
        "pearl_harbor": 1941,
        "pearl harbor": 1941,
        "midway": 1942,
        "stalingrad": 1942,
        "d-day": 1944,
        "normandy": 1944,
        "overlord": 1944,
        "bulge": 1944,
        "berlin": 1945,
        "hiroshima": 1945,
        "nagasaki": 1945,
    }

    def __init__(self):
        # Year pattern: "in 1944", "during 1943", "1942"
        self.year_pattern = re.compile(r'\b(19[34][0-9])\b')

        # Range pattern: "between 1941 and 1945"
        self.range_pattern = re.compile(
            r'\bbetween\s+(19[34][0-9])\s+and\s+(19[34][0-9])\b'
        )

        # Early/late war patterns
        self.early_war_pattern = re.compile(
            r'\b(early|beginning|start|initial)\s+(war|years?|period)\b',
            re.IGNORECASE,
        )
        self.late_war_pattern = re.compile(
            r'\b(late|end|final|closing)\s+(war|years?|period)\b',
            re.IGNORECASE,
        )

        logger.info("TemporalFilter initialized")

    def extract_temporal_info(self, query: str) -> Optional[Dict]:
        """
        Extract temporal information from query.

        Args:
            query: Search query text

        Returns:
            Dict with temporal constraints or None
            Format: {
                "type": "year" | "range" | "early" | "late",
                "start_year": int,
                "end_year": int,
            }
        """
        query_lower = query.lower()

        # 1. Check for year ranges
        range_match = self.range_pattern.search(query)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            return {
                "type": "range",
                "start_year": min(start, end),
                "end_year": max(start, end),
            }

        # 2. Check for explicit years
        year_matches = self.year_pattern.findall(query)
        if year_matches:
            years = [int(y) for y in year_matches]
            if len(years) == 1:
                # Single year mentioned
                return {
                    "type": "year",
                    "start_year": years[0],
                    "end_year": years[0],
                }
            else:
                # Multiple years - use as range
                return {
                    "type": "range",
                    "start_year": min(years),
                    "end_year": max(years),
                }

        # 3. Check for early war
        if self.early_war_pattern.search(query):
            return {
                "type": "early",
                "start_year": self.WW2_START,
                "end_year": 1941,  # Through Pearl Harbor
            }

        # 4. Check for late war
        if self.late_war_pattern.search(query):
            return {
                "type": "late",
                "start_year": 1944,  # D-Day onwards
                "end_year": self.WW2_END,
            }

        # 5. Check for key events
        for event_name, event_year in self.KEY_EVENTS.items():
            if event_name in query_lower:
                # Use event year +/- 1 year window
                return {
                    "type": "event",
                    "start_year": event_year - 1,
                    "end_year": event_year + 1,
                }

        return None

    def create_qdrant_filter(self, temporal_info: Dict) -> Dict:
        """
        Create Qdrant filter from temporal information.

        Args:
            temporal_info: Dict from extract_temporal_info()

        Returns:
            Qdrant filter dict in QdrantManager expected format
        """
        start_year = temporal_info["start_year"]
        end_year = temporal_info["end_year"]

        # Qdrant filter for year range
        # Format expected by QdrantManager._build_filter()
        filter_dict = {
            "year": {
                "range": {
                    "gte": start_year,
                    "lte": end_year,
                }
            }
        }

        logger.debug(
            f"Created temporal filter: {start_year}-{end_year} "
            f"(type: {temporal_info['type']})"
        )

        return filter_dict

    def apply_temporal_boost(
        self,
        results: List[Dict],
        temporal_info: Dict,
        boost_factor: float = 1.5,
    ) -> List[Dict]:
        """
        Boost scores of temporally relevant chunks.

        Args:
            results: Search results from Qdrant
            temporal_info: Temporal constraints
            boost_factor: Multiplicative boost for matching chunks

        Returns:
            Results with adjusted scores
        """
        start_year = temporal_info["start_year"]
        end_year = temporal_info["end_year"]

        for result in results:
            chunk_year = result.get("payload", {}).get("year")

            if chunk_year and start_year <= chunk_year <= end_year:
                # Apply temporal boost
                result["score"] *= boost_factor

                logger.debug(
                    f"Boosted chunk {result['id']} "
                    f"(year={chunk_year}) by {boost_factor}x"
                )

        # Re-sort by score
        results.sort(key=lambda x: x["score"], reverse=True)

        return results

    def extract_and_filter(self, query: str) -> Tuple[str, Optional[Dict]]:
        """
        Extract temporal info and return cleaned query + filter.

        Args:
            query: Original search query

        Returns:
            (cleaned_query, qdrant_filter or None)
        """
        temporal_info = self.extract_temporal_info(query)

        if temporal_info:
            filter_dict = self.create_qdrant_filter(temporal_info)
            logger.info(
                f"Temporal query detected: {temporal_info['type']} "
                f"({temporal_info['start_year']}-{temporal_info['end_year']})"
            )
            return query, filter_dict
        else:
            return query, None
