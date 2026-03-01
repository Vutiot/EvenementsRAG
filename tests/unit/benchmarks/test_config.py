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
    OPENROUTER_FREE_MODELS,
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
# GenerationConfig extensions (E2-F4-T1)
# ---------------------------------------------------------------------------


class TestGenerationConfigExtensions:
    def test_top_k_articles_defaults_to_none(self):
        cfg = GenerationConfig()
        assert cfg.top_k_articles is None

    def test_top_k_articles_accepts_valid_value(self):
        cfg = GenerationConfig(top_k_articles=3)
        assert cfg.top_k_articles == 3

    def test_top_k_articles_ge1_raises(self):
        with pytest.raises(Exception):
            GenerationConfig(top_k_articles=0)

    def test_top_k_articles_le20_raises(self):
        with pytest.raises(Exception):
            GenerationConfig(top_k_articles=21)

    def test_prompt_template_defaults_to_none(self):
        cfg = GenerationConfig()
        assert cfg.prompt_template is None

    def test_prompt_template_accepts_string(self):
        tpl = "Answer: {context}\nQ: {query}"
        cfg = GenerationConfig(prompt_template=tpl)
        assert cfg.prompt_template == tpl

    def test_config_hash_changes_on_top_k_articles(self, vanilla_config):
        h1 = vanilla_config.config_hash()
        modified = vanilla_config.model_copy(
            update={"generation": GenerationConfig(top_k_articles=3)}
        )
        assert modified.config_hash() != h1

    def test_config_hash_changes_on_prompt_template(self, vanilla_config):
        h1 = vanilla_config.config_hash()
        modified = vanilla_config.model_copy(
            update={"generation": GenerationConfig(prompt_template="Custom: {query}")}
        )
        assert modified.config_hash() != h1


# ---------------------------------------------------------------------------
# Generation parameter sweeps (E2-F4-T1)
# ---------------------------------------------------------------------------


class TestGenerationSweeps:
    def test_temperature_sweep_returns_3_configs(self, vanilla_config):
        configs = vanilla_config.temperature_sweep()
        assert len(configs) == 3

    def test_temperature_sweep_values(self, vanilla_config):
        configs = vanilla_config.temperature_sweep()
        temps = [c.generation.temperature for c in configs]
        assert temps == [0.0, 0.3, 0.7]

    def test_temperature_sweep_names(self, vanilla_config):
        configs = vanilla_config.temperature_sweep()
        for cfg, expected_temp in zip(configs, [0.0, 0.3, 0.7]):
            assert cfg.name == f"{vanilla_config.name}_temp{expected_temp}"

    def test_temperature_sweep_preserves_other_params(self, vanilla_config):
        configs = vanilla_config.temperature_sweep()
        for cfg in configs:
            assert cfg.generation.top_k_chunks == vanilla_config.generation.top_k_chunks
            assert cfg.retrieval.technique == vanilla_config.retrieval.technique

    def test_top_k_chunks_sweep_returns_3_configs(self, vanilla_config):
        configs = vanilla_config.top_k_chunks_sweep()
        assert len(configs) == 3

    def test_top_k_chunks_sweep_values(self, vanilla_config):
        configs = vanilla_config.top_k_chunks_sweep()
        ks = [c.generation.top_k_chunks for c in configs]
        assert ks == [3, 5, 10]

    def test_top_k_chunks_sweep_names(self, vanilla_config):
        configs = vanilla_config.top_k_chunks_sweep()
        for cfg, expected_k in zip(configs, [3, 5, 10]):
            assert cfg.name == f"{vanilla_config.name}_topk{expected_k}"

    def test_model_sweep_returns_3_configs(self, vanilla_config):
        configs = vanilla_config.model_sweep()
        assert len(configs) == 3

    def test_model_sweep_uses_openrouter_free_models(self, vanilla_config):
        configs = vanilla_config.model_sweep()
        models = [c.generation.model for c in configs]
        assert models == OPENROUTER_FREE_MODELS

    def test_model_sweep_forces_openrouter_provider(self, vanilla_config):
        configs = vanilla_config.model_sweep()
        for cfg in configs:
            assert cfg.generation.llm_provider == "openrouter"

    def test_model_sweep_names_contain_model_prefix(self, vanilla_config):
        configs = vanilla_config.model_sweep()
        for cfg in configs:
            assert cfg.name.startswith(f"{vanilla_config.name}_model_")

    def test_model_sweep_configs_have_different_hashes(self, vanilla_config):
        configs = vanilla_config.model_sweep()
        hashes = [c.config_hash() for c in configs]
        assert len(set(hashes)) == 3  # all distinct
