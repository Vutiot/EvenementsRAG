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
# Sweep methods
# ---------------------------------------------------------------------------


class TestSweeps:
    def test_chunk_size_sweep_default_length(self):
        assert len(BenchmarkConfig.chunk_size_sweep()) == 3

    def test_chunk_size_sweep_sizes(self):
        cfgs = BenchmarkConfig.chunk_size_sweep()
        assert [c.chunking.chunk_size for c in cfgs] == [256, 512, 1024]

    def test_chunk_size_sweep_unique_collection_names(self):
        names = [c.dataset.collection_name for c in BenchmarkConfig.chunk_size_sweep()]
        assert len(names) == len(set(names))

    def test_chunk_size_sweep_unique_benchmark_names(self):
        names = [c.name for c in BenchmarkConfig.chunk_size_sweep()]
        assert len(names) == len(set(names))

    def test_chunk_size_sweep_overlap_unchanged(self):
        base = BenchmarkConfig.phase1_vanilla()
        for cfg in BenchmarkConfig.chunk_size_sweep(base=base):
            assert cfg.chunking.chunk_overlap == base.chunking.chunk_overlap

    def test_chunk_size_sweep_custom_sizes(self):
        cfgs = BenchmarkConfig.chunk_size_sweep(sizes=[128, 256])
        assert [c.chunking.chunk_size for c in cfgs] == [128, 256]

    def test_chunk_size_sweep_preserves_retrieval_technique(self):
        base = BenchmarkConfig.phase2_hybrid()
        for cfg in BenchmarkConfig.chunk_size_sweep(base=base):
            assert cfg.retrieval.technique == "hybrid"

    def test_chunk_overlap_sweep_default_length(self):
        assert len(BenchmarkConfig.chunk_overlap_sweep()) == 4

    def test_chunk_overlap_sweep_overlaps(self):
        cfgs = BenchmarkConfig.chunk_overlap_sweep()
        assert [c.chunking.chunk_overlap for c in cfgs] == [0, 50, 128, 256]

    def test_chunk_overlap_sweep_unique_collection_names(self):
        names = [c.dataset.collection_name for c in BenchmarkConfig.chunk_overlap_sweep()]
        assert len(names) == len(set(names))

    def test_chunk_overlap_sweep_chunk_size_unchanged(self):
        base = BenchmarkConfig.phase1_vanilla()
        for cfg in BenchmarkConfig.chunk_overlap_sweep(base=base):
            assert cfg.chunking.chunk_size == base.chunking.chunk_size

    def test_chunk_overlap_sweep_skips_invalid_overlap(self):
        base = BenchmarkConfig.phase1_vanilla()   # chunk_size=512
        with pytest.warns(UserWarning):
            cfgs = BenchmarkConfig.chunk_overlap_sweep(base=base, overlaps=[0, 512, 600])
        assert len(cfgs) == 1  # only overlap=0 is valid

    def test_sweep_yaml_files_loadable(self):
        sweep_files = sorted(Path("config/benchmarks").glob("sweep_*.yaml"))
        assert len(sweep_files) == 6
        for f in sweep_files:
            cfg = BenchmarkConfig.from_yaml(f)
            assert cfg.chunking.chunk_size >= 64
