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


# ---------------------------------------------------------------------------
# EvaluationConfig RAGAS fields (E1-F2-T2)
# ---------------------------------------------------------------------------


class TestEvaluationConfigRagas:
    def test_ragas_defaults(self):
        cfg = EvaluationConfig()
        assert cfg.compute_ragas is False
        assert len(cfg.ragas_metrics) == 12
        assert "faithfulness" in cfg.ragas_metrics
        assert "answer_relevancy" in cfg.ragas_metrics
        assert cfg.ragas_evaluator_model == "mistralai/mistral-small-3.1-24b-instruct:free"
        assert cfg.ragas_max_workers == 1
        assert cfg.ragas_timeout == 180

    def test_ragas_custom_metrics(self):
        cfg = EvaluationConfig(ragas_metrics=["faithfulness", "coherence"])
        assert cfg.ragas_metrics == ["faithfulness", "coherence"]

    def test_ragas_unknown_metric_warns(self):
        with pytest.warns(UserWarning, match="unknown ragas_metrics"):
            EvaluationConfig(ragas_metrics=["faithfulness", "nonexistent_metric"])

    def test_ragas_config_hash_changes(self, vanilla_config):
        h1 = vanilla_config.config_hash()
        modified = vanilla_config.model_copy(
            update={"evaluation": EvaluationConfig(compute_ragas=True)}
        )
        assert modified.config_hash() != h1


# ---------------------------------------------------------------------------
# User config merging (E3-F1-T3)
# ---------------------------------------------------------------------------


class TestLoadWithUserOverrides:
    def test_load_with_empty_user_config(self, tmp_path):
        """Loading with empty/non-existent user config returns base config."""
        base_path = Path("config/benchmarks/default.yaml")
        cfg = BenchmarkConfig.load_with_user_overrides(base_path)
        assert cfg.name == "default"
        assert cfg.generation.model == "mistralai/mistral-small-3.1-24b-instruct:free"

    def test_load_with_empty_user_config_file(self, tmp_path):
        """Loading with empty user config file (no content) returns base config."""
        base_path = Path("config/benchmarks/default.yaml")
        user_path = tmp_path / "empty.yaml"
        user_path.write_text("")

        cfg = BenchmarkConfig.load_with_user_overrides(base_path, user_path)
        assert cfg.name == "default"
        assert cfg.generation.model == "mistralai/mistral-small-3.1-24b-instruct:free"

    def test_load_with_single_field_override(self, tmp_path):
        """User config overrides a single field in base config."""
        base_path = Path("config/benchmarks/default.yaml")
        user_path = tmp_path / "user.yaml"
        user_path.write_text("generation:\n  model: custom-model:free\n")

        cfg = BenchmarkConfig.load_with_user_overrides(base_path, user_path)
        assert cfg.generation.model == "custom-model:free"
        # Other fields should remain unchanged
        assert cfg.generation.temperature == 0.0
        assert cfg.chunking.chunk_size == 512

    def test_load_with_multiple_field_overrides(self, tmp_path):
        """User config overrides multiple fields."""
        base_path = Path("config/benchmarks/default.yaml")
        user_path = tmp_path / "user.yaml"
        user_path.write_text(
            "generation:\n"
            "  model: custom-model:free\n"
            "  temperature: 0.5\n"
            "  max_tokens: 1000\n"
        )

        cfg = BenchmarkConfig.load_with_user_overrides(base_path, user_path)
        assert cfg.generation.model == "custom-model:free"
        assert cfg.generation.temperature == 0.5
        assert cfg.generation.max_tokens == 1000
        # Other generation fields should remain unchanged
        assert cfg.generation.top_k_chunks == 5

    def test_load_with_nested_override(self, tmp_path):
        """User config can override nested fields across different sections."""
        base_path = Path("config/benchmarks/default.yaml")
        user_path = tmp_path / "user.yaml"
        user_path.write_text(
            "generation:\n"
            "  model: llama:free\n"
            "chunking:\n"
            "  chunk_size: 1024\n"
        )

        cfg = BenchmarkConfig.load_with_user_overrides(base_path, user_path)
        assert cfg.generation.model == "llama:free"
        assert cfg.chunking.chunk_size == 1024
        assert cfg.chunking.chunk_overlap == 50  # Unchanged

    def test_load_with_nonexistent_user_path(self, tmp_path):
        """If user_config_path doesn't exist, returns base config."""
        base_path = Path("config/benchmarks/default.yaml")
        user_path = tmp_path / "nonexistent.yaml"

        cfg = BenchmarkConfig.load_with_user_overrides(base_path, user_path)
        assert cfg.name == "default"
        assert cfg.generation.model == "mistralai/mistral-small-3.1-24b-instruct:free"

    def test_load_with_none_user_path(self):
        """If user_config_path is None, returns base config."""
        base_path = Path("config/benchmarks/default.yaml")
        cfg = BenchmarkConfig.load_with_user_overrides(base_path, None)
        assert cfg.name == "default"
        assert cfg.generation.model == "mistralai/mistral-small-3.1-24b-instruct:free"

    def test_load_preserves_config_hash_on_no_override(self, tmp_path):
        """Config hash should be the same if user config doesn't override anything."""
        base_path = Path("config/benchmarks/default.yaml")
        base_cfg = BenchmarkConfig.from_yaml(base_path)

        user_path = tmp_path / "empty_override.yaml"
        user_path.write_text("# Empty override file\n")

        cfg = BenchmarkConfig.load_with_user_overrides(base_path, user_path)
        assert cfg.config_hash() == base_cfg.config_hash()

    def test_default_yaml_exists_and_loads(self):
        """The default.yaml preset file exists and loads correctly."""
        default_path = Path("config/benchmarks/default.yaml")
        assert default_path.exists()

        cfg = BenchmarkConfig.from_yaml(default_path)
        assert cfg.name == "default"
        assert cfg.description == "Default preset - base configuration for interactive queries"

    def test_user_config_yaml_exists(self):
        """The user-config.yaml template file exists."""
        user_config_path = Path("config/benchmarks/user-config.yaml")
        assert user_config_path.exists()

    def test_root_symlinks_exist(self):
        """Root-level symlinks exist and point to the correct files."""
        default_link = Path("default.yaml")
        user_config_link = Path("user-config.yaml")

        assert default_link.exists() or default_link.is_symlink()
        assert user_config_link.exists() or user_config_link.is_symlink()
