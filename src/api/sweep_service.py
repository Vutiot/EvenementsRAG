"""Sweep run service — cartesian product of params, sequential benchmark execution with SSE."""

import hashlib
import itertools
import json
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from src.api.collection_service import CollectionService
from src.api.dependencies import DATASETS_DIR, PRESETS_DIR, RESULTS_DIR
from src.api.schemas import SweepRunRequest
from src.benchmarks.config import BenchmarkConfig, _deep_merge
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _set_nested(d: dict, path: str, value: object) -> None:
    """Set a value at a dotted path in a nested dict."""
    parts = path.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


# Embedding model → dimension mapping (avoids importing heavy config module constants)
_EMBEDDING_DIMENSIONS: dict[str, int] = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
}


def compute_cartesian_configs(
    base: BenchmarkConfig,
    sweep_params: dict[str, list],
    single_overrides: dict | None = None,
) -> list[tuple[dict, BenchmarkConfig]]:
    """Generate all config combinations from sweep_params.

    Returns list of (param_dict, BenchmarkConfig) tuples.
    """
    if not sweep_params:
        cfg = base
        if single_overrides:
            merged = cfg.model_dump()
            _deep_merge(merged, single_overrides)
            cfg = BenchmarkConfig.model_validate(merged)
        return [({}, cfg)]

    keys = sorted(sweep_params.keys())
    value_lists = [sweep_params[k] for k in keys]

    configs: list[tuple[dict, BenchmarkConfig]] = []
    for combo in itertools.product(*value_lists):
        param_dict = dict(zip(keys, combo))

        merged = base.model_dump()
        if single_overrides:
            _deep_merge(merged, single_overrides)

        for path, value in param_dict.items():
            _set_nested(merged, path, value)

        # Auto-resolve sparse_weight → dense_weight
        if "retrieval.sparse_weight" in param_dict:
            sw = float(param_dict["retrieval.sparse_weight"])
            _set_nested(merged, "retrieval.dense_weight", round(1.0 - sw, 2))

        # Auto-resolve embedding model → dimension
        if "embedding.model_name" in param_dict:
            model = param_dict["embedding.model_name"]
            dim = _EMBEDDING_DIMENSIONS.get(model)
            if dim is not None:
                _set_nested(merged, "embedding.dimension", dim)

        try:
            cfg = BenchmarkConfig.model_validate(merged)
            configs.append((param_dict, cfg))
        except Exception as exc:
            logger.warning(f"Skipping invalid config {param_dict}: {exc}")

    return configs


class SweepService:
    """Execute a multi-config sweep, yielding SSE events."""

    def run_sweep(self, request: SweepRunRequest) -> Generator[str, None, None]:
        """Execute a sweep run, yielding SSE events."""
        try:
            # 1. Load preset config
            preset_path = PRESETS_DIR / request.preset
            if not preset_path.exists():
                yield _sse("error", {"message": f"Preset '{request.preset}' not found"})
                return

            if request.preset == "default.yaml":
                user_config_path = PRESETS_DIR / "user-config.yaml"
                cfg = BenchmarkConfig.load_with_user_overrides(preset_path, user_config_path)
            else:
                cfg = BenchmarkConfig.from_yaml(preset_path)

            # 2. Compute cartesian product
            configs = compute_cartesian_configs(
                cfg, request.sweep_params, request.config_overrides,
            )
            total_configs = len(configs)

            if total_configs == 0:
                yield _sse("error", {"message": "No valid configurations generated from sweep parameters"})
                return

            # 3. Resolve eval dataset
            dataset_path = DATASETS_DIR / f"{request.eval_dataset_id}.json"
            if not dataset_path.exists():
                yield _sse("error", {"message": f"Eval dataset '{request.eval_dataset_id}' not found"})
                return

            with open(dataset_path, "r", encoding="utf-8") as f:
                ds_data = json.load(f)
            questions = ds_data.get("questions", [])
            total_questions = len(questions)

            # 4. Instantiate collection service (used for warnings + per-config loop)
            svc = CollectionService()

            # 4a. Warn if sweep has multiple chunk configurations
            chunk_combos = {
                (c.chunking.chunk_size, c.chunking.chunk_overlap)
                for _, c in configs
            }
            if len(chunk_combos) > 1:
                yield _sse("warning", {
                    "message": (
                        "Sweep uses multiple chunk configurations. "
                        "Only Document-ID and LLM Context metrics are valid across all configs. "
                        "Chunk-ID metrics only valid for the collection matching the eval dataset."
                    ),
                })

            # 4b. Warn if eval dataset collection doesn't match any sweep config
            ds_col = ds_data.get("collection_name")
            if ds_col:
                for _, config in configs:
                    test_col = svc.derive_collection_name(
                        dataset_name=config.dataset.dataset_name,
                        backend=config.vector_db.backend,
                        chunk_size=config.chunking.chunk_size,
                        chunk_overlap=config.chunking.chunk_overlap,
                        embedding_model=config.embedding.model_name,
                        distance_metric=config.vector_db.distance_metric,
                    )
                    if test_col != ds_col:
                        yield _sse("warning", {
                            "message": (
                                f"Eval dataset was generated from collection '{ds_col}' "
                                f"but sweep includes collection '{test_col}'. "
                                "Chunk-ID metrics may be unreliable for mismatched collections."
                            ),
                        })
                        break

            # 5. Generate sweep ID
            sweep_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            sweep_hash = hashlib.sha256(
                json.dumps(request.model_dump(), sort_keys=True, default=str).encode()
            ).hexdigest()[:8]
            sweep_id = f"sweep_{sweep_hash}_{sweep_ts}"

            yield _sse("sweep_started", {
                "sweep_id": sweep_id,
                "total_configs": total_configs,
                "total_questions_per_config": total_questions,
                "sweep_name": request.name or sweep_id,
            })

            # 6. Run each config sequentially
            result_files: list[str] = []

            for config_index, (param_dict, config) in enumerate(configs):
                # 5a. Derive collection name
                col_name = svc.derive_collection_name(
                    dataset_name=config.dataset.dataset_name,
                    backend=config.vector_db.backend,
                    chunk_size=config.chunking.chunk_size,
                    chunk_overlap=config.chunking.chunk_overlap,
                    embedding_model=config.embedding.model_name,
                    distance_metric=config.vector_db.distance_metric,
                )

                yield _sse("config_started", {
                    "config_index": config_index + 1,
                    "total_configs": total_configs,
                    "config_hash": config.config_hash(),
                    "params": {k: _serialize(v) for k, v in param_dict.items()},
                    "collection_name": col_name,
                })

                # 5b. Ensure collection exists
                try:
                    from src.vector_store.factory import VectorStoreFactory
                    store = VectorStoreFactory.from_config(config.vector_db)
                    if not store.collection_exists(col_name):
                        svc.create_and_index(
                            dataset_name=config.dataset.dataset_name,
                            collection_name=col_name,
                            backend=config.vector_db.backend,
                            chunk_size=config.chunking.chunk_size,
                            chunk_overlap=config.chunking.chunk_overlap,
                            embedding_model=config.embedding.model_name,
                            embedding_dimension=config.embedding.dimension,
                            distance_metric=config.vector_db.distance_metric,
                        )
                except Exception as exc:
                    logger.warning(f"Collection check/create failed for {col_name}: {exc}")

                # 5c. Update config with resolved values
                merged_dump = config.model_dump()
                merged_dump["dataset"]["collection_name"] = col_name
                merged_dump["dataset"]["questions_file"] = str(dataset_path)
                merged_dump["generation"]["highlight_chunks"] = False
                final_cfg = BenchmarkConfig.model_validate(merged_dump)

                # 5d. Run benchmark with progress callback (threaded)
                from src.benchmarks.runner import ParameterizedBenchmarkRunner

                progress_queue: queue.Queue[str | None] = queue.Queue()
                result_holder: list = []
                error_holder: list[str] = []

                ci = config_index  # capture for closure

                def _progress_callback(idx: int, total: int, evaluation: dict, _ci: int = ci) -> None:
                    progress_queue.put(_sse("config_progress", {
                        "config_index": _ci + 1,
                        "total_configs": total_configs,
                        "question_index": idx + 1,
                        "total_questions": total,
                        "question_id": evaluation.get("question_id", ""),
                        "question_type": evaluation.get("type", ""),
                        "retrieval_time_ms": round(evaluation.get("retrieval_time_ms", 0), 1),
                    }))

                def _worker() -> None:
                    try:
                        runner = ParameterizedBenchmarkRunner(config=final_cfg)
                        result = runner.run(
                            output_dir=RESULTS_DIR,
                            progress_callback=_progress_callback,
                        )
                        result_holder.append(result)
                    except Exception as exc:
                        error_holder.append(str(exc))
                    finally:
                        progress_queue.put(None)

                thread = threading.Thread(target=_worker, daemon=True)
                thread.start()

                while True:
                    try:
                        item = progress_queue.get(timeout=300)
                    except queue.Empty:
                        yield _sse("error", {"message": f"Config {config_index + 1} timed out"})
                        return
                    if item is None:
                        break
                    yield item

                thread.join(timeout=5)

                if error_holder:
                    yield _sse("config_complete", {
                        "config_index": config_index + 1,
                        "total_configs": total_configs,
                        "status": "error",
                        "error": error_holder[0],
                        "params": {k: _serialize(v) for k, v in param_dict.items()},
                    })
                    continue

                if not result_holder:
                    yield _sse("config_complete", {
                        "config_index": config_index + 1,
                        "total_configs": total_configs,
                        "status": "error",
                        "error": "No result returned",
                        "params": {k: _serialize(v) for k, v in param_dict.items()},
                    })
                    continue

                result = result_holder[0]
                technique = result.config.retrieval.technique
                ts = result.timestamp.replace(":", "").replace("-", "")
                hash8 = result.config_hash[:8]
                rel_filename = f"{technique}/{result.phase_name}_{hash8}_{ts}.json"
                result_files.append(rel_filename)

                ctx_prec = result.metrics_summary.get("ragas", {}).get("avg_context_precision")
                ent_r5 = result.metrics_summary.get("entity", {}).get("avg_entity_recall_at_5")

                yield _sse("config_complete", {
                    "config_index": config_index + 1,
                    "total_configs": total_configs,
                    "status": "ok",
                    "filename": rel_filename,
                    "params": {k: _serialize(v) for k, v in param_dict.items()},
                    "avg_mrr": round(result.evaluation.avg_mrr, 4),
                    "avg_recall_at_5": round(result.evaluation.avg_recall_at_k.get(5, 0.0), 4),
                    "avg_recall_at_10": round(result.evaluation.avg_recall_at_k.get(10, 0.0), 4),
                    "avg_doc_mrr": round(result.evaluation.avg_doc_mrr, 4),
                    "avg_doc_precision_at_5": round(result.evaluation.avg_doc_precision_at_k.get(5, 0.0), 4),
                    "avg_context_precision": round(ctx_prec, 4) if ctx_prec is not None else None,
                    "avg_entity_recall_at_5": round(ent_r5, 4) if ent_r5 is not None else None,
                    "total_wall_time_s": round(result.total_wall_time_s, 2),
                })

            # 6. Save sweep metadata
            sweep_meta = {
                "sweep_id": sweep_id,
                "sweep_name": request.name or sweep_id,
                "timestamp": sweep_ts,
                "preset": request.preset,
                "sweep_params": request.sweep_params,
                "config_overrides": request.config_overrides,
                "eval_dataset_id": request.eval_dataset_id,
                "total_configs": total_configs,
                "total_questions_per_config": total_questions,
                "result_files": result_files,
            }
            meta_dir = RESULTS_DIR / "sweeps"
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_path = meta_dir / f"sweep_meta_{sweep_hash}_{sweep_ts}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(sweep_meta, f, indent=2)

            yield _sse("sweep_complete", {
                "sweep_id": sweep_id,
                "total_configs": total_configs,
                "completed_configs": len(result_files),
                "result_files": result_files,
                "meta_filename": f"sweeps/{meta_path.name}",
            })

        except Exception as exc:
            logger.error(f"Sweep run failed: {exc}", exc_info=True)
            yield _sse("error", {"message": str(exc)})


def _serialize(v: object) -> object:
    """Ensure value is JSON-serializable."""
    if isinstance(v, float):
        return round(v, 4)
    return v
