"""File-based benchmark result cache (E4-F1-T2, E4-F1-T3).

A run whose parameters were already evaluated reuses the existing result JSON
instead of recomputing retrieval, generation and judges. The result files stay
the source of truth; the cache is only a small index next to them:

    results/cache_index.json   {"<cache key>": "<technique>/<file>.json", ...}

Cache key = SHA-256 of
  * ``BenchmarkConfig.config_hash()`` (every parameter except name/description,
    including the collection and the questions file path),
  * a SHA-256 of the questions file contents, so a regenerated eval dataset
    misses the cache automatically,
  * ``max_questions`` and the evaluation mode (direct / mapped).

There is no code-version tracking: after changing metric code, rerun with
``use_cache=False`` (``no_cache`` in the API).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

INDEX_FILENAME = "cache_index.json"


def file_sha256(path: Path) -> str:
    """Content hash of a file, or a marker when it cannot be read."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return f"missing:{path}"


def cache_key(
    config_hash: str,
    questions_file: Path,
    max_questions: Optional[int] = None,
    evaluation_mode: Optional[str] = None,
) -> str:
    payload = {
        "config_hash": config_hash,
        "dataset_sha256": file_sha256(Path(questions_file)),
        "max_questions": max_questions,
        "evaluation_mode": evaluation_mode or "direct",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ResultCache:
    """Hash → result-file index stored in ``<results_dir>/cache_index.json``."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.index_path = self.results_dir / INDEX_FILENAME

    def _read_index(self) -> Dict[str, str]:
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_index(self, index: Dict[str, str]) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.results_dir, prefix=".cache_index_", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=1, sort_keys=True)
        os.replace(tmp, self.index_path)

    def lookup(self, key: str) -> Optional[Path]:
        """Path of the cached result file, or ``None``. Drops stale entries."""
        index = self._read_index()
        rel = index.get(key)
        if rel is None:
            return None
        path = self.results_dir / rel
        if path.is_file():
            return path
        logger.info(f"Result cache: dropping stale entry {key[:12]} → {rel}")
        index.pop(key, None)
        self._write_index(index)
        return None

    def register(self, key: str, result_path: Path) -> None:
        index = self._read_index()
        index[key] = Path(result_path).relative_to(self.results_dir).as_posix()
        self._write_index(index)
