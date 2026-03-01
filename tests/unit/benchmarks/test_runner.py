"""Unit tests for src/benchmarks/runner.py (E1-F1-T2).

Covers BenchmarkResult serialization, ParameterizedBenchmarkRunner instantiation,
_build_rag_pipeline dispatch, run() orchestration, _run_generation_pass behaviour,
and run_sweep() batching — all external I/O is mocked.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.benchmarks.config import BenchmarkConfig, GenerationConfig, RetrievalConfig
from src.benchmarks.runner import BenchmarkResult, ParameterizedBenchmarkRunner
from src.evaluation.metrics import RetrievalMetrics


# ---------------------------------------------------------------------------
# File-scoped fixture: fully mocked run() context
# ---------------------------------------------------------------------------


@pytest.fixture
def run_ctx(vanilla_config_no_gen, fake_evaluation_results):
    """Yield a pre-wired runner with BenchmarkRunner and _build_rag_pipeline mocked.

    Produces a dict with keys:
        runner     — ParameterizedBenchmarkRunner instance
        mock_br    — mock BenchmarkRunner class
        eval_results — the fake EvaluationResults returned by run_benchmark()
    """
    runner = ParameterizedBenchmarkRunner(
        config=vanilla_config_no_gen,
        qdrant_manager=MagicMock(),
        embedding_generator=MagicMock(),
    )
    runner._rag_pipeline = MagicMock()

    with patch("src.benchmarks.runner.BenchmarkRunner") as mock_br, patch.object(
        ParameterizedBenchmarkRunner, "_build_rag_pipeline"
    ):
        mock_br.return_value.run_benchmark.return_value = fake_evaluation_results
        yield {
            "runner": runner,
            "mock_br": mock_br,
            "eval_results": fake_evaluation_results,
        }


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    def test_to_dict_has_required_keys(self, fake_benchmark_result):
        d = fake_benchmark_result.to_dict()
        for key in (
            "config",
            "config_hash",
            "phase_name",
            "timestamp",
            "evaluation",
            "per_question_full",
            "total_wall_time_s",
        ):
            assert key in d, f"missing key: {key}"

    def test_to_json_string_is_valid_json(self, fake_benchmark_result):
        content = fake_benchmark_result.to_json()
        parsed = json.loads(content)
        assert isinstance(parsed, dict)
        assert "config_hash" in parsed

    def test_to_json_writes_file(self, tmp_path, fake_benchmark_result):
        path = tmp_path / "result.json"
        fake_benchmark_result.to_json(path)
        assert path.exists()
        parsed = json.loads(path.read_text())
        assert "phase_name" in parsed

    def test_print_summary_contains_phase_and_hash(self, fake_benchmark_result, capsys):
        fake_benchmark_result.print_summary()
        out = capsys.readouterr().out
        assert fake_benchmark_result.phase_name in out
        assert fake_benchmark_result.config_hash in out


# ---------------------------------------------------------------------------
# ParameterizedBenchmarkRunner — instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_instantiation_stores_config(self, vanilla_config):
        runner = ParameterizedBenchmarkRunner(config=vanilla_config)
        assert runner.config is vanilla_config


# ---------------------------------------------------------------------------
# _build_rag_pipeline
# ---------------------------------------------------------------------------


class TestBuildRagPipeline:
    def test_build_pipeline_vanilla_imports_correctly(self, vanilla_config):
        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.VanillaRetriever = mock_cls

        runner = ParameterizedBenchmarkRunner(
            config=vanilla_config,
            qdrant_manager=MagicMock(),
            embedding_generator=MagicMock(),
        )

        with patch(
            "src.benchmarks.runner.importlib.import_module",
            return_value=mock_module,
        ) as mock_import:
            runner._build_rag_pipeline()
            mock_import.assert_called_once_with("src.rag.phase1_vanilla.retriever")

        mock_cls.assert_called_once_with(
            collection_name=vanilla_config.dataset.collection_name,
            qdrant_manager=runner._vector_store,
            embedding_generator=runner._embedding_gen,
        )
        assert runner._rag_pipeline is mock_cls.return_value

    def test_build_pipeline_hybrid_raises_not_implemented(self, hybrid_config):
        runner = ParameterizedBenchmarkRunner(
            config=hybrid_config,
            qdrant_manager=MagicMock(),
            embedding_generator=MagicMock(),
        )
        with pytest.raises(NotImplementedError):
            runner._build_rag_pipeline()

    def test_build_pipeline_temporal_raises_not_implemented(self):
        cfg = BenchmarkConfig(retrieval=RetrievalConfig(technique="temporal"))
        runner = ParameterizedBenchmarkRunner(
            config=cfg,
            qdrant_manager=MagicMock(),
            embedding_generator=MagicMock(),
        )
        with pytest.raises(NotImplementedError):
            runner._build_rag_pipeline()


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_returns_benchmark_result(self, run_ctx):
        result = run_ctx["runner"].run()
        assert isinstance(result, BenchmarkResult)

    def test_run_result_has_correct_phase_name(self, run_ctx):
        result = run_ctx["runner"].run()
        assert result.phase_name == run_ctx["runner"].config.name

    def test_run_result_has_config_hash(self, run_ctx):
        result = run_ctx["runner"].run()
        assert len(result.config_hash) == 16
        assert all(c in "0123456789abcdef" for c in result.config_hash)

    def test_run_result_timestamp_is_utc_iso(self, run_ctx):
        result = run_ctx["runner"].run()
        assert result.timestamp.endswith("Z")
        # Basic ISO-8601 shape: YYYY-MM-DDTHH:MM:SSZ
        assert "T" in result.timestamp

    def test_run_wall_time_is_positive(self, run_ctx):
        result = run_ctx["runner"].run()
        assert result.total_wall_time_s >= 0

    def test_run_per_question_full_matches_eval_results(self, run_ctx):
        result = run_ctx["runner"].run()
        expected_len = len(run_ctx["eval_results"].per_question_metrics)
        assert len(result.per_question_full) == expected_len

    def test_run_generation_disabled_skips_pass(self, run_ctx):
        result = run_ctx["runner"].run()
        # generation.enabled is False on vanilla_config_no_gen — no generated_answer keys
        for entry in result.per_question_full:
            assert "generated_answer" not in entry

    def test_run_with_max_questions(self, run_ctx):
        run_ctx["runner"].run(max_questions=5)
        run_ctx["mock_br"].return_value.run_benchmark.assert_called_once_with(
            collection_name=run_ctx["runner"].config.dataset.collection_name,
            phase_name=run_ctx["runner"].config.name,
            max_questions=5,
        )

    def test_run_with_questions_file_override(
        self, tmp_path, vanilla_config_no_gen, fake_evaluation_results
    ):
        override_path = tmp_path / "custom_questions.json"
        override_path.write_text('{"questions": []}', encoding="utf-8")

        runner = ParameterizedBenchmarkRunner(
            config=vanilla_config_no_gen,
            qdrant_manager=MagicMock(),
            embedding_generator=MagicMock(),
        )
        runner._rag_pipeline = MagicMock()

        with patch("src.benchmarks.runner.BenchmarkRunner") as mock_br, patch.object(
            ParameterizedBenchmarkRunner, "_build_rag_pipeline"
        ):
            mock_br.return_value.run_benchmark.return_value = fake_evaluation_results
            runner.run(questions_file=override_path)

        mock_br.assert_called_once_with(
            questions_file=override_path,
            qdrant_manager=runner._vector_store,
            embedding_generator=runner._embedding_gen,
            k_values=vanilla_config_no_gen.evaluation.k_values,
        )


# ---------------------------------------------------------------------------
# _run_generation_pass
# ---------------------------------------------------------------------------


class TestGenerationPass:
    def test_generation_pass_adds_answer_field(
        self, vanilla_config, tmp_questions_file
    ):
        runner = ParameterizedBenchmarkRunner(config=vanilla_config)
        mock_response = MagicMock()
        mock_response.answer = "The Normandy landings on June 6, 1944."
        runner._rag_pipeline = MagicMock()
        runner._rag_pipeline.query.return_value = mock_response

        per_q = [{"question_id": "q1", "question": "What was D-Day?"}]
        runner._run_generation_pass(tmp_questions_file, per_q, None)

        assert per_q[0]["generated_answer"] == "The Normandy landings on June 6, 1944."

    def test_generation_pass_adds_generation_time(
        self, vanilla_config, tmp_questions_file
    ):
        runner = ParameterizedBenchmarkRunner(config=vanilla_config)
        runner._rag_pipeline = MagicMock()
        runner._rag_pipeline.query.return_value = MagicMock(answer="Some answer")

        per_q = [{"question_id": "q1", "question": "What was D-Day?"}]
        runner._run_generation_pass(tmp_questions_file, per_q, None)

        assert "generation_time_ms" in per_q[0]
        assert per_q[0]["generation_time_ms"] >= 0

    def test_generation_pass_handles_query_failure(
        self, vanilla_config, tmp_questions_file
    ):
        runner = ParameterizedBenchmarkRunner(config=vanilla_config)
        runner._rag_pipeline = MagicMock()
        runner._rag_pipeline.query.side_effect = RuntimeError("model unavailable")

        per_q = [{"question_id": "q1", "question": "What was D-Day?"}]
        runner._run_generation_pass(tmp_questions_file, per_q, None)  # must not raise

        assert per_q[0]["generated_answer"] is None
        assert per_q[0]["generation_time_ms"] == 0.0


# ---------------------------------------------------------------------------
# run_sweep()
# ---------------------------------------------------------------------------


class TestRunSweep:
    @patch.object(ParameterizedBenchmarkRunner, "run")
    def test_run_sweep_returns_list_of_results(
        self, mock_run, vanilla_config, fake_benchmark_result
    ):
        mock_run.return_value = fake_benchmark_result
        results = ParameterizedBenchmarkRunner.run_sweep(
            [vanilla_config, vanilla_config]
        )
        assert len(results) == 2
        assert all(isinstance(r, BenchmarkResult) for r in results)

    @patch.object(
        ParameterizedBenchmarkRunner,
        "run",
        side_effect=NotImplementedError("not yet implemented"),
    )
    def test_run_sweep_skips_not_implemented_by_default(
        self, mock_run, hybrid_config
    ):
        results = ParameterizedBenchmarkRunner.run_sweep([hybrid_config])
        assert results == []

    @patch.object(
        ParameterizedBenchmarkRunner,
        "run",
        side_effect=NotImplementedError("not yet implemented"),
    )
    def test_run_sweep_stop_on_error_reraises(self, mock_run, hybrid_config):
        with pytest.raises(NotImplementedError):
            ParameterizedBenchmarkRunner.run_sweep(
                [hybrid_config], stop_on_error=True
            )

    @patch.object(ParameterizedBenchmarkRunner, "run")
    def test_run_sweep_saves_json_files(
        self, mock_run, vanilla_config, fake_benchmark_result, tmp_path
    ):
        mock_run.return_value = fake_benchmark_result
        ParameterizedBenchmarkRunner.run_sweep(
            [vanilla_config], output_dir=tmp_path
        )
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1
        # Verify the file is valid JSON
        parsed = json.loads(json_files[0].read_text())
        assert isinstance(parsed, dict)
