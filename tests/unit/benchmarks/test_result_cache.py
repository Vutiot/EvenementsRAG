"""Tests for the file-based result cache (E4-F1-T2 / E4-F1-T3)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.benchmarks.result_cache import INDEX_FILENAME, ResultCache, cache_key
from src.benchmarks.runner import ParameterizedBenchmarkRunner


@pytest.fixture
def questions(tmp_path):
    p = tmp_path / "eval.json"
    p.write_text(json.dumps({"questions": [{"id": "q1"}]}))
    return p


@pytest.fixture
def results_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return d


def _runner(cfg, fake_evaluation_results):
    runner = ParameterizedBenchmarkRunner(
        config=cfg, qdrant_manager=MagicMock(), embedding_generator=MagicMock()
    )
    runner._rag_pipeline = MagicMock()
    return runner


@pytest.fixture
def run_env(fake_evaluation_results):
    with (
        patch("src.benchmarks.runner.BenchmarkRunner") as mock_br,
        patch.object(ParameterizedBenchmarkRunner, "_build_rag_pipeline"),
        patch("src.benchmarks.dataset_manager.DatasetManager"),
    ):
        mock_br.return_value.run_benchmark.return_value = fake_evaluation_results
        yield mock_br


class TestCacheKey:
    def test_stable(self, questions):
        assert cache_key("abc", questions) == cache_key("abc", questions)

    def test_changes_with_config_dataset_contents_and_limits(self, questions):
        base = cache_key("abc", questions)
        assert cache_key("abd", questions) != base
        assert cache_key("abc", questions, max_questions=10) != base
        assert cache_key("abc", questions, evaluation_mode="mapped") != base
        questions.write_text(json.dumps({"questions": [{"id": "q2"}]}))
        assert cache_key("abc", questions) != base

    def test_missing_dataset_file_does_not_crash(self, tmp_path):
        assert cache_key("abc", tmp_path / "nope.json")


class TestResultCacheIndex:
    def test_register_and_lookup(self, results_dir):
        f = results_dir / "vanilla" / "run.json"
        f.parent.mkdir()
        f.write_text("{}")
        cache = ResultCache(results_dir)
        assert cache.lookup("k") is None
        cache.register("k", f)
        assert cache.lookup("k") == f
        index = json.loads((results_dir / INDEX_FILENAME).read_text())
        assert index == {"k": "vanilla/run.json"}

    def test_stale_entry_dropped(self, results_dir):
        f = results_dir / "run.json"
        f.write_text("{}")
        cache = ResultCache(results_dir)
        cache.register("k", f)
        f.unlink()
        assert cache.lookup("k") is None
        assert json.loads((results_dir / INDEX_FILENAME).read_text()) == {}

    def test_corrupt_index_is_ignored(self, results_dir):
        (results_dir / INDEX_FILENAME).write_text("not json")
        assert ResultCache(results_dir).lookup("k") is None


class TestRunnerCache:
    def test_second_identical_run_hits_cache(
        self, vanilla_config_no_gen, fake_evaluation_results, run_env, questions, results_dir
    ):
        first = _runner(vanilla_config_no_gen, fake_evaluation_results).run(
            questions_file=questions, output_dir=results_dir
        )
        assert first.cached_from is None
        assert run_env.return_value.run_benchmark.call_count == 1

        cfg2 = vanilla_config_no_gen.model_copy(deep=True)
        cfg2.name = "renamed_run"
        second = _runner(cfg2, fake_evaluation_results).run(
            questions_file=questions, output_dir=results_dir
        )
        # Nothing recomputed, metrics copied, new file under the new name
        assert run_env.return_value.run_benchmark.call_count == 1
        assert second.cached_from == first.relative_filename(results_dir)
        assert second.phase_name == "renamed_run"
        assert second.evaluation.avg_mrr == pytest.approx(first.evaluation.avg_mrr)
        assert second.evaluation.avg_recall_at_k == first.evaluation.avg_recall_at_k
        assert second.result_path.exists() and second.result_path != first.result_path
        saved = json.loads(second.result_path.read_text())
        assert saved["cached_from"] == second.cached_from
        assert saved["phase_name"] == "renamed_run"

    def test_use_cache_false_reruns(
        self, vanilla_config_no_gen, fake_evaluation_results, run_env, questions, results_dir
    ):
        for _ in range(2):
            _runner(vanilla_config_no_gen, fake_evaluation_results).run(
                questions_file=questions, output_dir=results_dir, use_cache=False
            )
        assert run_env.return_value.run_benchmark.call_count == 2

    def test_regenerated_dataset_misses_cache(
        self, vanilla_config_no_gen, fake_evaluation_results, run_env, questions, results_dir
    ):
        _runner(vanilla_config_no_gen, fake_evaluation_results).run(
            questions_file=questions, output_dir=results_dir
        )
        questions.write_text(json.dumps({"questions": [{"id": "new"}]}))
        result = _runner(vanilla_config_no_gen, fake_evaluation_results).run(
            questions_file=questions, output_dir=results_dir
        )
        assert result.cached_from is None
        assert run_env.return_value.run_benchmark.call_count == 2

    def test_no_output_dir_never_caches(
        self, vanilla_config_no_gen, fake_evaluation_results, run_env, questions
    ):
        for _ in range(2):
            _runner(vanilla_config_no_gen, fake_evaluation_results).run(questions_file=questions)
        assert run_env.return_value.run_benchmark.call_count == 2
