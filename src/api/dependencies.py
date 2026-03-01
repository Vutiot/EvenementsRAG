"""Shared dependencies for the EvenementsRAG API."""

from pathlib import Path

# Resolve config directory relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PRESETS_DIR = _PROJECT_ROOT / "config" / "benchmarks"
