"""Benchmark result file endpoints."""

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.api.dependencies import RESULTS_DIR
from src.api.schemas import (
    NormalizedBenchmarkResult,
    NormalizedQuestion,
    ResultFileInfo,
)

router = APIRouter()

# Simple mtime-based cache: {filepath: (mtime, parsed_info)}
_file_info_cache: dict[str, tuple[float, ResultFileInfo]] = {}


def _infer_phase_name(filename: str, data: dict) -> str:
    """Infer phase name from data or filename."""
    if "phase_name" in data:
        return data["phase_name"]
    # Guess from filename: phase1_baseline_30q.json → phase1_baseline
    stem = Path(filename).stem
    # Remove trailing _NNq or _NNarticles_NNq suffixes
    cleaned = re.sub(r"_\d+articles_\d+q$", "", stem)
    cleaned = re.sub(r"_\d+q$", "", cleaned)
    return cleaned


def _parse_file_info(filepath: Path) -> ResultFileInfo | None:
    """Parse a result JSON file into a ResultFileInfo summary."""
    mtime = filepath.stat().st_mtime
    cache_key = str(filepath)
    if cache_key in _file_info_cache:
        cached_mtime, cached_info = _file_info_cache[cache_key]
        if cached_mtime == mtime:
            return cached_info

    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if not isinstance(data, dict):
        return None

    is_new_format = "config" in data
    fmt = "benchmark_result" if is_new_format else "legacy"

    # Extract metrics depending on format
    if is_new_format:
        eval_data = data.get("evaluation", {})
        avg_mrr = eval_data.get("avg_mrr", 0.0)
        recall_at_k = eval_data.get("avg_recall_at_k", {})
        per_q = data.get("per_question_full", eval_data.get("per_question_metrics", []))
        total = eval_data.get("total_questions", len(per_q))
        timestamp = data.get("timestamp")
    else:
        avg_mrr = data.get("avg_mrr", 0.0)
        recall_at_k = data.get("avg_recall_at_k", {})
        per_q = data.get("per_question_metrics", [])
        total = data.get("total_questions", len(per_q))
        timestamp = None

    # Get recall@5 (keys may be int-strings)
    avg_recall_at_5 = recall_at_k.get("5") or recall_at_k.get(5)

    phase_name = _infer_phase_name(filepath.name, data)

    # Store path relative to RESULTS_DIR so frontend can fetch subdirectory files
    try:
        rel_path = filepath.relative_to(RESULTS_DIR).as_posix()
    except ValueError:
        rel_path = filepath.name

    info = ResultFileInfo(
        filename=rel_path,
        phase_name=phase_name,
        timestamp=timestamp,
        total_questions=total,
        format=fmt,
        avg_mrr=round(avg_mrr, 4),
        avg_recall_at_5=round(avg_recall_at_5, 4) if avg_recall_at_5 is not None else None,
    )
    _file_info_cache[cache_key] = (mtime, info)
    return info


def _normalize_question_legacy(q: dict) -> NormalizedQuestion:
    """Normalize a per-question entry from the legacy format."""
    return NormalizedQuestion(
        question_id=q.get("question_id", "unknown"),
        question=q.get("question", ""),
        type=q.get("type", "unknown"),
        difficulty=q.get("difficulty"),
        source_article=q.get("source_article"),
        ground_truth_count=q.get("ground_truth_count"),
        retrieved_count=q.get("retrieved_count"),
        retrieval_time_ms=q.get("retrieval_time_ms"),
        metrics=q.get("metrics", {}),
    )


def _normalize_question_new(q: dict) -> NormalizedQuestion:
    """Normalize a per-question entry from the new BenchmarkResult format."""
    return NormalizedQuestion(
        question_id=q.get("question_id", "unknown"),
        question=q.get("question", ""),
        type=q.get("type", "unknown"),
        difficulty=q.get("difficulty"),
        source_article=q.get("source_article"),
        ground_truth_count=q.get("ground_truth_count"),
        retrieved_count=q.get("retrieved_count"),
        retrieval_time_ms=q.get("retrieval_time_ms"),
        metrics=q.get("metrics", {}),
        generated_answer=q.get("generated_answer"),
        generation_time_ms=q.get("generation_time_ms"),
        retrieved_contexts=q.get("retrieved_contexts"),
        generation_metrics=q.get("generation_metrics"),
        ragas_metrics=q.get("ragas_metrics"),
    )


def _stringify_keys(d: dict) -> dict[str, float]:
    """Ensure dict keys are strings (JSON ints become str)."""
    return {str(k): v for k, v in d.items()} if d else {}


def _normalize_result(filename: str, data: dict) -> NormalizedBenchmarkResult:
    """Normalize both legacy and new result formats into a unified shape."""
    is_new_format = "config" in data

    if is_new_format:
        eval_data = data.get("evaluation", {})
        per_q_raw = data.get("per_question_full", eval_data.get("per_question_metrics", []))
        per_question = [_normalize_question_new(q) for q in per_q_raw]
        return NormalizedBenchmarkResult(
            filename=filename,
            format="benchmark_result",
            phase_name=data.get("phase_name", "unknown"),
            timestamp=data.get("timestamp"),
            config=data.get("config"),
            avg_recall_at_k=_stringify_keys(eval_data.get("avg_recall_at_k", {})),
            avg_mrr=eval_data.get("avg_mrr", 0.0),
            avg_ndcg=_stringify_keys(eval_data.get("avg_ndcg", {})),
            avg_article_hit_at_k=_stringify_keys(eval_data.get("avg_article_hit_at_k", {})) or None,
            avg_chunk_hit_at_k=_stringify_keys(eval_data.get("avg_chunk_hit_at_k", {})) or None,
            metrics_by_type=eval_data.get("metrics_by_type", {}),
            per_question=per_question,
            total_questions=eval_data.get("total_questions", len(per_question)),
            avg_retrieval_time_ms=eval_data.get("avg_retrieval_time_ms", 0.0),
            total_wall_time_s=data.get("total_wall_time_s"),
            metrics_summary=data.get("metrics_summary") or None,
        )
    else:
        per_q_raw = data.get("per_question_metrics", [])
        per_question = [_normalize_question_legacy(q) for q in per_q_raw]
        phase_name = _infer_phase_name(filename, data)
        return NormalizedBenchmarkResult(
            filename=filename,
            format="legacy",
            phase_name=phase_name,
            avg_recall_at_k=_stringify_keys(data.get("avg_recall_at_k", {})),
            avg_mrr=data.get("avg_mrr", 0.0),
            avg_ndcg=_stringify_keys(data.get("avg_ndcg", {})),
            avg_article_hit_at_k=_stringify_keys(data.get("avg_article_hit_at_k", {})) or None,
            avg_chunk_hit_at_k=_stringify_keys(data.get("avg_chunk_hit_at_k", {})) or None,
            metrics_by_type=data.get("metrics_by_type", {}),
            per_question=per_question,
            total_questions=data.get("total_questions", len(per_question)),
            avg_retrieval_time_ms=data.get("avg_retrieval_time_ms", 0.0),
        )


@router.get("/results", response_model=list[ResultFileInfo])
def list_results():
    """List all JSON result files from the results/ directory."""
    if not RESULTS_DIR.exists():
        return []

    infos = []
    for path in sorted(RESULTS_DIR.rglob("*.json")):
        info = _parse_file_info(path)
        if info is not None:
            infos.append(info)
    return infos


@router.get("/results/{filename:path}", response_model=NormalizedBenchmarkResult)
def get_result(filename: str):
    """Load and normalize a single result file."""
    # Only allow .json files
    if not filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files are supported")

    filepath = (RESULTS_DIR / filename).resolve()
    results_root = RESULTS_DIR.resolve()
    # Prevent path traversal — must be under RESULTS_DIR
    try:
        filepath.relative_to(results_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Result file '{filename}' not found")

    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse: {exc}") from exc

    return _normalize_result(filename, data)
