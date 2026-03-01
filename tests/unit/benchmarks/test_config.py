"""Unit tests for src/benchmarks/config.py (E1-F1-T1).

Covers:
- BenchmarkConfig defaults and named presets
- config_hash stability and sensitivity
- YAML round-trip, file I/O, and preset YAML file loading
- Pydantic validators (ValidationError cases)
- Warning on non-standard k_values
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.benchmarks.config import (
    BenchmarkConfig,
    ChunkingConfig,
    DatasetConfig,
    EmbeddingConfig,
    EvaluationConfig,
    GenerationConfig,
    RerankerConfig,
    RetrievalConfig,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_defaults(self):
        cfg = BenchmarkConfig()
        assert cfg.name == "unnamed_benchmark"
        assert cfg.description == ""
        assert isinstance(cfg.dataset, DatasetConfig)
        assert isinstance(cfg.embedding, EmbeddingConfig)
        assert isinstance(cfg.chunking, ChunkingConfig)
        assert isinstance(cfg.retrieval, RetrievalConfig)
        assert isinstance(cfg.reranker, RerankerConfig)
        assert isinstance(cfg.generation, GenerationConfig)
        assert isinstance(cfg.evaluation, EvaluationConfig)

    def test_phase1_vanilla_preset(self, vanilla_config):
        assert vanilla_config.retrieval.technique == "vanilla"
        assert "all-MiniLM-L6-v2" in vanilla_config.embedding.model_name
        assert vanilla_config.name == "phase1_vanilla"

    def test_phase2_hybrid_preset(self, hybrid_config):
        assert hybrid_config.retrieval.technique == "hybrid"
        assert hybrid_config.retrieval.sparse_weight == 0.3
        assert hybrid_config.retrieval.dense_weight == 0.7
        assert hybrid_config.name == "phase2_hybrid"


# ---------------------------------------------------------------------------
# config_hash
# ---------------------------------------------------------------------------


class TestConfigHash:
    def test_config_hash_is_16_hex_chars(self, vanilla_config):
        h = vanilla_config.config_hash()
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_config_hash_excludes_name_description(self, vanilla_config):
        h1 = vanilla_config.config_hash()
        renamed = vanilla_config.model_copy(
            update={"name": "different_name", "description": "totally different desc"}
        )
        assert renamed.config_hash() == h1

    def test_config_hash_changes_on_param_change(self, vanilla_config):
        h1 = vanilla_config.config_hash()
        modified = vanilla_config.model_copy(
            update={"chunking": ChunkingConfig(chunk_size=256, chunk_overlap=50)}
        )
        assert modified.config_hash() != h1


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------


class TestYAMLIO:
    def test_yaml_round_trip(self, vanilla_config):
        yaml_str = vanilla_config.to_yaml()
        reloaded = BenchmarkConfig.from_yaml_string(yaml_str)
        assert reloaded.config_hash() == vanilla_config.config_hash()

    def test_from_yaml_file_phase1(self, tmp_path, vanilla_config):
        path = tmp_path / "phase1.yaml"
        vanilla_config.to_yaml(path)
        reloaded = BenchmarkConfig.from_yaml(path)
        assert reloaded.retrieval.technique == "vanilla"

    def test_from_yaml_file_phase2(self, tmp_path, hybrid_config):
        path = tmp_path / "phase2.yaml"
        hybrid_config.to_yaml(path)
        reloaded = BenchmarkConfig.from_yaml(path)
        assert reloaded.retrieval.technique == "hybrid"

    def test_to_yaml_writes_file(self, tmp_path, vanilla_config):
        path = tmp_path / "output.yaml"
        assert not path.exists()
        vanilla_config.to_yaml(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_preset_yaml_files_load(self):
        phase1_path = Path("config/benchmarks/phase1_vanilla.yaml")
        phase2_path = Path("config/benchmarks/phase2_hybrid.yaml")

        cfg1 = BenchmarkConfig.from_yaml(phase1_path)
        assert cfg1.retrieval.technique == "vanilla"

        cfg2 = BenchmarkConfig.from_yaml(phase2_path)
        assert cfg2.retrieval.technique == "hybrid"


# ---------------------------------------------------------------------------
# Validators (ValidationError expected)
# ---------------------------------------------------------------------------


class TestValidators:
    def test_chunk_overlap_gte_chunk_size_raises(self):
        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_size=512, chunk_overlap=512)

    def test_hybrid_weights_not_summing_to_one_raises(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(technique="hybrid", sparse_weight=0.4, dense_weight=0.4)

    def test_reranker_model_name_required_raises(self):
        with pytest.raises(ValidationError):
            RerankerConfig(type="cohere", model_name=None)


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


class TestWarnings:
    def test_nonstandard_k_values_warns(self):
        with pytest.warns(UserWarning):
            EvaluationConfig(k_values=[2, 4])


# ---------------------------------------------------------------------------
# Distance Metric Sweep (E2-F2-T2)
# ---------------------------------------------------------------------------


class TestDistanceMetricSweep:
    def test_returns_three_configs(self):
        configs = BenchmarkConfig.distance_metric_sweep()
        assert len(configs) == 3

    def test_metrics_covered(self):
        configs = BenchmarkConfig.distance_metric_sweep()
        metrics = {c.vector_db.distance_metric for c in configs}
        assert metrics == {"cosine", "euclidean", "dot_product"}

    def test_collection_names(self):
        configs = BenchmarkConfig.distance_metric_sweep()
        names = {c.dataset.collection_name for c in configs}
        assert names == {"ww2_dm_cosine", "ww2_dm_euclidean", "ww2_dm_dot_product"}

    def test_hashes_differ(self):
        configs = BenchmarkConfig.distance_metric_sweep()
        hashes = {c.config_hash() for c in configs}
        assert len(hashes) == 3

    def test_manhattan_excluded(self):
        configs = BenchmarkConfig.distance_metric_sweep()
        metrics = [c.vector_db.distance_metric for c in configs]
        assert "manhattan" not in metrics

    def test_yaml_presets_load(self):
        from pathlib import Path

        for metric in ("cosine", "euclidean", "dot_product"):
            path = Path(f"config/benchmarks/wiki_dm_{metric}.yaml")
            cfg = BenchmarkConfig.from_yaml(path)
            assert cfg.vector_db.distance_metric == metric
            assert cfg.dataset.collection_name == f"ww2_dm_{metric}"

    def test_all_use_vanilla_baseline(self):
        configs = BenchmarkConfig.distance_metric_sweep()
        for cfg in configs:
            assert cfg.retrieval.technique == "vanilla"
            assert cfg.chunking.chunk_size == 512
            assert cfg.chunking.chunk_overlap == 50
            assert "all-MiniLM-L6-v2" in cfg.embedding.model_name
