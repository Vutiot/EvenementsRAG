"""Unit tests for E2-F2-T3: embedding model sweep, known-model validator,
runner model_name plumbing, and cache-hash model isolation.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.benchmarks.config import (
    BenchmarkConfig,
    EmbeddingConfig,
    _EMBEDDING_SWEEP_MODELS,
)


# ---------------------------------------------------------------------------
# Embedding Model Sweep
# ---------------------------------------------------------------------------


class TestEmbeddingModelSweep:
    def test_returns_four_configs(self):
        configs = BenchmarkConfig.embedding_model_sweep()
        assert len(configs) == 4

    def test_models_covered(self):
        configs = BenchmarkConfig.embedding_model_sweep()
        models = {c.embedding.model_name for c in configs}
        assert models == set(_EMBEDDING_SWEEP_MODELS.keys())

    def test_collection_names(self):
        configs = BenchmarkConfig.embedding_model_sweep()
        names = {c.dataset.collection_name for c in configs}
        assert names == {
            "ww2_em_minilm_l6",
            "ww2_em_minilm_l12",
            "ww2_em_bge_small",
            "ww2_em_bge_base",
        }

    def test_dimensions_match_registry(self):
        configs = BenchmarkConfig.embedding_model_sweep()
        for cfg in configs:
            _, expected_dim = _EMBEDDING_SWEEP_MODELS[cfg.embedding.model_name]
            assert cfg.embedding.dimension == expected_dim

    def test_hashes_differ(self):
        configs = BenchmarkConfig.embedding_model_sweep()
        hashes = {c.config_hash() for c in configs}
        assert len(hashes) == 4

    def test_all_use_vanilla_baseline(self):
        configs = BenchmarkConfig.embedding_model_sweep()
        for cfg in configs:
            assert cfg.retrieval.technique == "vanilla"
            assert cfg.chunking.chunk_size == 512
            assert cfg.chunking.chunk_overlap == 50

    def test_yaml_round_trip(self):
        configs = BenchmarkConfig.embedding_model_sweep()
        for cfg in configs:
            yaml_str = cfg.to_yaml()
            reloaded = BenchmarkConfig.from_yaml_string(yaml_str)
            assert reloaded.config_hash() == cfg.config_hash()
            assert reloaded.embedding.model_name == cfg.embedding.model_name
            assert reloaded.embedding.dimension == cfg.embedding.dimension

    def test_yaml_presets_load(self):
        short_names = ["minilm_l6", "minilm_l12", "bge_small", "bge_base"]
        for short in short_names:
            path = Path(f"config/benchmarks/wiki_em_{short}.yaml")
            cfg = BenchmarkConfig.from_yaml(path)
            assert cfg.dataset.collection_name == f"ww2_em_{short}"
            assert cfg.name == f"wiki_em_{short}"


# ---------------------------------------------------------------------------
# Known Model Dimension Validator
# ---------------------------------------------------------------------------


class TestKnownModelDimension:
    def test_minilm_l12_correct_dimension(self):
        cfg = EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L12-v2", dimension=384
        )
        assert cfg.dimension == 384

    def test_bge_small_correct_dimension(self):
        cfg = EmbeddingConfig(model_name="BAAI/bge-small-en-v1.5", dimension=384)
        assert cfg.dimension == 384

    def test_bge_base_correct_dimension(self):
        cfg = EmbeddingConfig(model_name="BAAI/bge-base-en-v1.5", dimension=768)
        assert cfg.dimension == 768

    def test_bge_base_wrong_dimension_warns(self):
        with pytest.warns(UserWarning, match="dimension"):
            EmbeddingConfig(model_name="BAAI/bge-base-en-v1.5", dimension=384)

    def test_minilm_l12_wrong_dimension_warns(self):
        with pytest.warns(UserWarning, match="dimension"):
            EmbeddingConfig(
                model_name="sentence-transformers/all-MiniLM-L12-v2", dimension=768
            )

    def test_unknown_model_no_warning(self):
        # Unknown models should not trigger a warning
        cfg = EmbeddingConfig(model_name="custom/my-model", dimension=256)
        assert cfg.dimension == 256


# ---------------------------------------------------------------------------
# Runner Embedding Model Plumbing
# ---------------------------------------------------------------------------


class TestRunnerEmbeddingModelPlumbing:
    def test_runner_passes_model_name_to_embedding_generator(self):
        """Verify EmbeddingGenerator is called with model_name from config."""
        from src.benchmarks.config import GenerationConfig
        from src.benchmarks.runner import ParameterizedBenchmarkRunner

        cfg = BenchmarkConfig(
            embedding=EmbeddingConfig(
                model_name="BAAI/bge-base-en-v1.5", dimension=768
            ),
            generation=GenerationConfig(enabled=False),
        )
        runner = ParameterizedBenchmarkRunner(
            config=cfg,
            vector_store=MagicMock(),
        )

        mock_eval_results = MagicMock()
        mock_eval_results.per_question_metrics = []

        with (
            patch(
                "src.embeddings.embedding_generator.EmbeddingGenerator",
            ) as mock_eg,
            patch("src.benchmarks.runner.BenchmarkRunner") as mock_br,
            patch.object(ParameterizedBenchmarkRunner, "_build_rag_pipeline"),
        ):
            mock_br.return_value.run_benchmark.return_value = mock_eval_results
            runner.run()

            mock_eg.assert_called_once_with(model_name="BAAI/bge-base-en-v1.5")

    def test_runner_skips_init_when_embedding_gen_injected(self):
        """When embedding_generator is injected, EmbeddingGenerator is not called."""
        from src.benchmarks.config import GenerationConfig
        from src.benchmarks.runner import ParameterizedBenchmarkRunner

        cfg = BenchmarkConfig(
            generation=GenerationConfig(enabled=False),
        )
        mock_eg = MagicMock()
        runner = ParameterizedBenchmarkRunner(
            config=cfg,
            vector_store=MagicMock(),
            embedding_generator=mock_eg,
        )

        mock_eval_results = MagicMock()
        mock_eval_results.per_question_metrics = []

        with (
            patch(
                "src.embeddings.embedding_generator.EmbeddingGenerator",
            ) as eg_cls,
            patch("src.benchmarks.runner.BenchmarkRunner") as mock_br,
            patch.object(ParameterizedBenchmarkRunner, "_build_rag_pipeline"),
        ):
            mock_br.return_value.run_benchmark.return_value = mock_eval_results
            runner.run()

            eg_cls.assert_not_called()
        assert runner._embedding_gen is mock_eg


# ---------------------------------------------------------------------------
# Cache Hash Includes Model Name
# ---------------------------------------------------------------------------


class TestCacheHashIncludesModel:
    def test_same_text_different_model_different_hash(self):
        """_hash_text must produce different hashes for different model names."""
        from src.embeddings.embedding_generator import EmbeddingGenerator

        text = "The Battle of Stalingrad was a turning point."

        with patch(
            "src.embeddings.embedding_generator.SentenceTransformer"
        ) as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model

            gen_a = EmbeddingGenerator(model_name="model-a")
            gen_b = EmbeddingGenerator(model_name="model-b")

        hash_a = gen_a._hash_text(text)
        hash_b = gen_b._hash_text(text)

        assert hash_a != hash_b

    def test_same_model_same_text_same_hash(self):
        """Same model + same text must produce the same hash."""
        from src.embeddings.embedding_generator import EmbeddingGenerator

        text = "D-Day was June 6, 1944."

        with patch(
            "src.embeddings.embedding_generator.SentenceTransformer"
        ) as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model

            gen = EmbeddingGenerator(model_name="model-x")

        assert gen._hash_text(text) == gen._hash_text(text)
