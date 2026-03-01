"""Unit tests for BenchmarkConfig.hybrid_weight_sweep() and YAML presets."""

import pytest
from pathlib import Path

from src.benchmarks.config import BenchmarkConfig


class TestHybridWeightSweep:
    """~9 tests covering sweep output and YAML preset correctness."""

    def test_returns_6_configs(self):
        configs = BenchmarkConfig.hybrid_weight_sweep()
        assert len(configs) == 6

    def test_all_technique_hybrid(self):
        configs = BenchmarkConfig.hybrid_weight_sweep()
        for cfg in configs:
            assert cfg.retrieval.technique == "hybrid"

    def test_collection_names(self):
        configs = BenchmarkConfig.hybrid_weight_sweep()
        expected = [
            "ww2_hybrid_w0",
            "ww2_hybrid_w10",
            "ww2_hybrid_w15",
            "ww2_hybrid_w20",
            "ww2_hybrid_w30",
            "ww2_hybrid_w50",
        ]
        actual = [cfg.dataset.collection_name for cfg in configs]
        assert actual == expected

    def test_sparse_plus_dense_equals_one(self):
        configs = BenchmarkConfig.hybrid_weight_sweep()
        for cfg in configs:
            total = cfg.retrieval.sparse_weight + cfg.retrieval.dense_weight
            assert abs(total - 1.0) < 0.01, (
                f"Weights don't sum to 1.0: {cfg.retrieval.sparse_weight} + "
                f"{cfg.retrieval.dense_weight} = {total}"
            )

    def test_unique_config_hashes(self):
        configs = BenchmarkConfig.hybrid_weight_sweep()
        hashes = [cfg.config_hash() for cfg in configs]
        assert len(set(hashes)) == 6, "Duplicate config hashes found"

    def test_yaml_round_trip(self, tmp_path):
        configs = BenchmarkConfig.hybrid_weight_sweep()
        for cfg in configs:
            yaml_str = cfg.to_yaml()
            restored = BenchmarkConfig.from_yaml_string(yaml_str)
            assert restored.retrieval.sparse_weight == cfg.retrieval.sparse_weight
            assert restored.retrieval.dense_weight == cfg.retrieval.dense_weight
            assert restored.dataset.collection_name == cfg.dataset.collection_name
            assert restored.retrieval.technique == "hybrid"

    def test_preset_files_exist(self):
        preset_dir = Path("config/benchmarks")
        for pct in [0, 10, 15, 20, 30, 50]:
            path = preset_dir / f"wiki_hybrid_w{pct}.yaml"
            assert path.exists(), f"Missing YAML preset: {path}"

    def test_preset_files_load_without_error(self):
        preset_dir = Path("config/benchmarks")
        for pct in [0, 10, 15, 20, 30, 50]:
            path = preset_dir / f"wiki_hybrid_w{pct}.yaml"
            cfg = BenchmarkConfig.from_yaml(path)
            assert cfg.retrieval.technique == "hybrid"

    def test_custom_base_config(self):
        base = BenchmarkConfig.phase2_hybrid()
        base = base.model_copy(update={"name": "custom_base"})
        configs = BenchmarkConfig.hybrid_weight_sweep(base=base)
        assert len(configs) == 6
        for cfg in configs:
            assert cfg.retrieval.technique == "hybrid"

    def test_custom_weights(self):
        custom_weights = [(0.0, 1.0), (0.5, 0.5)]
        configs = BenchmarkConfig.hybrid_weight_sweep(weights=custom_weights)
        assert len(configs) == 2
        assert configs[0].retrieval.sparse_weight == 0.0
        assert configs[1].retrieval.sparse_weight == 0.5
