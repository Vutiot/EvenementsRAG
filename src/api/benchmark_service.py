"""Benchmark run service — bridges API to ParameterizedBenchmarkRunner with SSE."""

import json
import time
from pathlib import Path
from typing import Generator

from src.api.collection_service import CollectionService
from src.api.dependencies import DATASETS_DIR, PRESETS_DIR, RESULTS_DIR
from src.api.schemas import BenchmarkRunRequest
from src.benchmarks.config import BenchmarkConfig, _deep_merge
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class BenchmarkService:
    """Run a full benchmark across all questions in an eval dataset."""

    def run_benchmark(self, request: BenchmarkRunRequest) -> Generator[str, None, None]:
        """Execute a benchmark run, yielding SSE events."""
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

            # 2. Apply config_overrides
            if request.config_overrides:
                merged = cfg.model_dump()
                _deep_merge(merged, request.config_overrides)
                cfg = BenchmarkConfig.model_validate(merged)

            # 3. Resolve eval dataset → questions file
            dataset_path = DATASETS_DIR / f"{request.eval_dataset_id}.json"
            if not dataset_path.exists():
                yield _sse("error", {"message": f"Eval dataset '{request.eval_dataset_id}' not found"})
                return

            # 4. Ensure collection exists
            svc = CollectionService()
            col_name = svc.derive_collection_name(
                dataset_name=cfg.dataset.dataset_name,
                backend=cfg.vector_db.backend,
                chunk_size=cfg.chunking.chunk_size,
                chunk_overlap=cfg.chunking.chunk_overlap,
                embedding_model=cfg.embedding.model_name,
                distance_metric=cfg.vector_db.distance_metric,
            )

            # Check if collection exists; if not, try to create it
            try:
                from src.vector_store.factory import VectorStoreFactory
                store = VectorStoreFactory.from_config(cfg.vector_db)
                if not store.collection_exists(col_name):
                    svc.create_and_index(
                        dataset_name=cfg.dataset.dataset_name,
                        collection_name=col_name,
                        backend=cfg.vector_db.backend,
                        chunk_size=cfg.chunking.chunk_size,
                        chunk_overlap=cfg.chunking.chunk_overlap,
                        embedding_model=cfg.embedding.model_name,
                        embedding_dimension=cfg.embedding.dimension,
                        distance_metric=cfg.vector_db.distance_metric,
                    )
            except Exception as exc:
                logger.warning(f"Collection check/create failed, proceeding anyway: {exc}")

            # 5. Update config with resolved values
            merged_dump = cfg.model_dump()
            merged_dump["dataset"]["collection_name"] = col_name
            merged_dump["dataset"]["questions_file"] = str(dataset_path)
            # Disable highlight_chunks for benchmark runs
            merged_dump["generation"]["highlight_chunks"] = False
            cfg = BenchmarkConfig.model_validate(merged_dump)

            # 6. Load questions to get count
            with open(dataset_path, "r", encoding="utf-8") as f:
                ds_data = json.load(f)
            questions = ds_data.get("questions", [])
            total_questions = len(questions)

            yield _sse("started", {
                "total_questions": total_questions,
                "config_hash": cfg.config_hash(),
            })

            # 7. Run benchmark with progress callback
            from src.benchmarks.runner import ParameterizedBenchmarkRunner

            runner = ParameterizedBenchmarkRunner(config=cfg)

            def progress_cb(idx: int, total: int, evaluation: dict):
                yield_data = {
                    "question_index": idx + 1,
                    "total_questions": total,
                    "question_id": evaluation.get("question_id", ""),
                    "question_type": evaluation.get("type", ""),
                    "retrieval_time_ms": round(evaluation.get("retrieval_time_ms", 0), 1),
                }
                progress_events.append(_sse("progress", yield_data))

            # Since progress_callback is sync and we need to collect events,
            # use a list to accumulate them
            progress_events: list[str] = []

            wall_start = time.time()

            # We can't yield from inside the callback, so we accumulate and yield after
            # Instead, run the benchmark and periodically yield progress
            # Use a threading approach similar to datasets
            import queue
            import threading

            progress_queue: queue.Queue[str | None] = queue.Queue()

            def _progress_callback(idx: int, total: int, evaluation: dict):
                event = _sse("progress", {
                    "question_index": idx + 1,
                    "total_questions": total,
                    "question_id": evaluation.get("question_id", ""),
                    "question_type": evaluation.get("type", ""),
                    "retrieval_time_ms": round(evaluation.get("retrieval_time_ms", 0), 1),
                })
                progress_queue.put(event)

            result_holder: list = []
            error_holder: list = []

            def _worker():
                try:
                    result = runner.run(
                        output_dir=RESULTS_DIR,
                        progress_callback=_progress_callback,
                    )
                    result_holder.append(result)
                except Exception as exc:
                    error_holder.append(str(exc))
                finally:
                    progress_queue.put(None)  # sentinel

            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()

            # Yield progress events as they come
            while True:
                try:
                    item = progress_queue.get(timeout=300)
                except queue.Empty:
                    yield _sse("error", {"message": "Benchmark timed out"})
                    return
                if item is None:
                    break
                yield item

            thread.join(timeout=5)

            if error_holder:
                yield _sse("error", {"message": error_holder[0]})
                return

            if not result_holder:
                yield _sse("error", {"message": "Benchmark completed without result"})
                return

            result = result_holder[0]
            wall_time = result.total_wall_time_s

            # Build the filename from the saved result
            from src.benchmarks.runner import _save_result
            # Result was already saved in run() because we passed output_dir
            # Get the filename
            technique = result.config.retrieval.technique
            ts = result.timestamp.replace(":", "").replace("-", "")
            hash8 = result.config_hash[:8]
            rel_filename = f"{technique}/{result.phase_name}_{hash8}_{ts}.json"

            yield _sse("complete", {
                "filename": rel_filename,
                "phase_name": result.phase_name,
                "total_questions": result.evaluation.total_questions,
                "avg_mrr": round(result.evaluation.avg_mrr, 4),
                "avg_recall_at_5": round(result.evaluation.avg_recall_at_k.get(5, 0.0), 4),
                "avg_recall_at_10": round(result.evaluation.avg_recall_at_k.get(10, 0.0), 4),
                "total_wall_time_s": round(wall_time, 2),
            })

        except Exception as exc:
            logger.error(f"Benchmark run failed: {exc}", exc_info=True)
            yield _sse("error", {"message": str(exc)})
