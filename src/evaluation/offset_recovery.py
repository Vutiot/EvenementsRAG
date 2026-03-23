"""Recover char_start / char_end for chunks and questions missing offsets.

Legacy Qdrant collections (indexed before E6-F12) lack char-offset fields
in their payloads.  This module re-computes offsets on the fly by locating
each chunk's text inside the original article content on disk.

The algorithm mirrors ``TextChunker.chunk_document()`` (text_chunker.py:294-307):
``content.find(chunk_text, search_from)`` with an advancing cursor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_ARTICLES_DIR = Path("data/raw/wikipedia_articles_10000")


def recover_chunk_offsets(
    chunks: Sequence[dict],
    articles_dir: str | Path = _DEFAULT_ARTICLES_DIR,
) -> int:
    """Compute ``char_start`` / ``char_end`` for chunks that have zero offsets.

    Loads original article content from *articles_dir* and uses
    ``str.find()`` to locate each chunk's text within the article.
    Mutates chunks **in-place**.

    Each chunk dict must contain at least:
    - ``content`` (str): the chunk text
    - ``article_title`` (str): used to resolve the article file
    - ``chunk_index`` (int): used to sort chunks before sequential search

    Args:
        chunks: Mutable sequence of chunk dicts.
        articles_dir: Directory containing ``{article_title}.json`` files.

    Returns:
        Number of offsets successfully recovered.
    """
    articles_dir = Path(articles_dir)
    if not articles_dir.is_dir():
        logger.warning("Articles directory not found: %s — skipping offset recovery", articles_dir)
        return 0

    # Collect chunks that need recovery, grouped by article_title
    by_article: dict[str, list[dict]] = {}
    for chunk in chunks:
        cs = chunk.get("char_start") or 0
        ce = chunk.get("char_end") or 0
        if cs == 0 and ce == 0 and chunk.get("content"):
            title = chunk.get("article_title") or ""
            if title:
                by_article.setdefault(title, []).append(chunk)

    if not by_article:
        return 0

    recovered = 0

    for title, article_chunks in by_article.items():
        article_path = articles_dir / f"{title}.json"
        if not article_path.exists():
            logger.debug("Article file not found for '%s' — skipping", title)
            continue

        try:
            with open(article_path, encoding="utf-8") as f:
                article_data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Could not load article '%s': %s", title, exc)
            continue

        article_content = article_data.get("content", "")
        if not article_content:
            continue

        # Sort by chunk_index so advancing search_from works correctly
        article_chunks.sort(key=lambda c: c.get("chunk_index", 0))

        search_from = 0
        for chunk in article_chunks:
            chunk_text = chunk["content"]
            pos = article_content.find(chunk_text, search_from)
            if pos == -1:
                # Fallback: try from beginning (overlapping chunks)
                pos = article_content.find(chunk_text)
            if pos == -1:
                # Fallback: use search_from as start (structure-based chunking)
                chunk["char_start"] = search_from
                chunk["char_end"] = search_from + len(chunk_text)
            else:
                chunk["char_start"] = pos
                chunk["char_end"] = pos + len(chunk_text)
                search_from = pos + 1
            recovered += 1

    logger.info("Recovered char offsets for %d/%d chunks", recovered, len(chunks))
    return recovered


def recover_question_offsets(
    questions: Sequence[dict],
    source_collection: str,
    articles_dir: str | Path = _DEFAULT_ARTICLES_DIR,
) -> int:
    """Recover ``char_start`` / ``char_end`` for eval questions with zero offsets.

    Loads the source chunks from *source_collection* (Qdrant), builds a
    ``chunk_id → {content, article_title}`` map, then locates each chunk's
    text inside the original article on disk.

    Mutates *questions* **in-place**.

    Args:
        questions: Mutable sequence of question dicts (from eval dataset JSON).
        source_collection: Qdrant collection the questions were generated from.
        articles_dir: Directory containing article JSON files.

    Returns:
        Number of question offsets successfully recovered.
    """
    articles_dir = Path(articles_dir)

    # Only recover questions that need it
    needs_recovery = [
        q for q in questions
        if (q.get("char_start") or 0) == 0 and (q.get("char_end") or 0) == 0
    ]
    if not needs_recovery:
        return 0

    # Collect the source_chunk_ids we need
    needed_chunk_ids = {q.get("source_chunk_id") for q in needs_recovery if q.get("source_chunk_id")}
    if not needed_chunk_ids:
        return 0

    # Load relevant chunks from source collection
    try:
        from src.vector_store.qdrant_manager import QdrantManager
        mgr = QdrantManager()

        chunk_map: dict[str, dict] = {}  # chunk_uuid → {content, article_title}
        offset = None
        while True:
            points, next_offset = mgr.client.scroll(
                collection_name=source_collection,
                limit=250,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in points:
                pt_id = str(pt.id)
                if pt_id in needed_chunk_ids:
                    chunk_map[pt_id] = {
                        "content": pt.payload.get("content", ""),
                        "article_title": pt.payload.get("article_title", "") or pt.payload.get("title", ""),
                    }
            if next_offset is None:
                break
            offset = next_offset
    except Exception as exc:
        logger.warning("Could not load source collection '%s': %s", source_collection, exc)
        return 0

    if not chunk_map:
        logger.warning("No matching chunks found in source collection '%s'", source_collection)
        return 0

    # Build a cache of loaded article contents
    article_cache: dict[str, str] = {}
    recovered = 0

    for q in needs_recovery:
        chunk_id = q.get("source_chunk_id")
        if not chunk_id or chunk_id not in chunk_map:
            continue

        info = chunk_map[chunk_id]
        chunk_content = info["content"]
        title = info["article_title"]
        if not chunk_content or not title:
            continue

        # Load article (cached)
        if title not in article_cache:
            article_path = articles_dir / f"{title}.json"
            if not article_path.exists():
                article_cache[title] = ""
            else:
                try:
                    with open(article_path, encoding="utf-8") as f:
                        data = json.load(f)
                    article_cache[title] = data.get("content", "")
                except (json.JSONDecodeError, OSError):
                    article_cache[title] = ""

        article_content = article_cache[title]
        if not article_content:
            continue

        pos = article_content.find(chunk_content)
        if pos != -1:
            q["char_start"] = pos
            q["char_end"] = pos + len(chunk_content)
            recovered += 1

    logger.info(
        "Recovered char offsets for %d/%d questions from collection '%s'",
        recovered, len(needs_recovery), source_collection,
    )
    return recovered
