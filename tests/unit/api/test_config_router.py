"""Unit tests for src/api/routers/config.py - preset configuration endpoints.

Covers:
- Listing available presets
- Loading preset configs
- User config merging for default preset
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api.routers.config import router
from src.benchmarks.config import BenchmarkConfig


@pytest.fixture
def client():
    """Create a test client for the config router."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestListPresets:
    def test_list_presets_returns_list(self, client):
        """GET /presets returns a list of preset info."""
        response = client.get("/presets")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_list_presets_includes_default(self, client):
        """GET /presets should include the default preset."""
        response = client.get("/presets")
        assert response.status_code == 200
        data = response.json()
        filenames = [p["filename"] for p in data]
        assert "default.yaml" in filenames

    def test_list_presets_includes_phase1(self, client):
        """GET /presets should include phase1_vanilla preset."""
        response = client.get("/presets")
        assert response.status_code == 200
        data = response.json()
        filenames = [p["filename"] for p in data]
        assert "phase1_vanilla.yaml" in filenames

    def test_list_presets_excludes_user_config(self, client):
        """GET /presets should exclude user-config.yaml (it's a template)."""
        response = client.get("/presets")
        assert response.status_code == 200
        data = response.json()
        filenames = [p["filename"] for p in data]
        assert "user-config.yaml" not in filenames

    def test_preset_info_has_required_fields(self, client):
        """Each preset info should have filename, name, and description."""
        response = client.get("/presets")
        assert response.status_code == 200
        data = response.json()
        for preset in data:
            assert "filename" in preset
            assert "name" in preset
            assert "description" in preset


class TestGetPreset:
    def test_get_preset_default(self, client):
        """GET /presets/default.yaml returns the default config."""
        response = client.get("/presets/default.yaml")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "default"
        assert data["generation"]["model"] == "mistralai/mistral-small-3.1-24b-instruct:free"

    def test_get_preset_phase1(self, client):
        """GET /presets/phase1_vanilla.yaml returns phase1 config."""
        response = client.get("/presets/phase1_vanilla.yaml")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "phase1_vanilla"
        assert data["retrieval"]["technique"] == "vanilla"

    def test_get_preset_phase2(self, client):
        """GET /presets/phase2_hybrid.yaml returns phase2 config."""
        response = client.get("/presets/phase2_hybrid.yaml")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "phase2_hybrid"
        assert data["retrieval"]["technique"] == "hybrid"

    def test_get_preset_nonexistent(self, client):
        """GET /presets/nonexistent.yaml returns 404."""
        response = client.get("/presets/nonexistent.yaml")
        assert response.status_code == 404

    def test_get_preset_no_extension(self, client):
        """GET /presets/default (no .yaml) returns 404."""
        response = client.get("/presets/default")
        assert response.status_code == 404

    def test_get_preset_path_traversal_blocked(self, client):
        """GET /presets/../foo.yaml is blocked (normalized to 404)."""
        response = client.get("/presets/../foo.yaml")
        # FastAPI normalizes the path, so ../foo.yaml becomes foo.yaml which doesn't exist
        assert response.status_code == 404

    def test_get_preset_with_user_config_merge(self, tmp_path, client):
        """GET /presets/default.yaml merges user-config.yaml if it exists."""
        # Create a temporary user-config with a model override
        user_config_path = Path("config/benchmarks/user-config.yaml")

        # Save original content
        original_content = user_config_path.read_text()

        try:
            # Write test content
            user_config_path.write_text(
                "generation:\n  model: test-llama:free\n"
            )

            response = client.get("/presets/default.yaml")
            assert response.status_code == 200
            data = response.json()
            assert data["generation"]["model"] == "test-llama:free"

        finally:
            # Restore original content
            user_config_path.write_text(original_content)

    def test_get_preset_non_default_ignores_user_config(self, client):
        """GET /presets/phase1_vanilla.yaml does NOT merge user-config.yaml."""
        # Create a temporary user-config with a model override
        user_config_path = Path("config/benchmarks/user-config.yaml")

        # Save original content
        original_content = user_config_path.read_text()

        try:
            # Write test content
            user_config_path.write_text(
                "generation:\n  model: test-llama:free\n"
            )

            response = client.get("/presets/phase1_vanilla.yaml")
            assert response.status_code == 200
            data = response.json()
            # phase1_vanilla should use its own model, not the user override
            assert data["generation"]["model"] == "mistralai/mistral-small-3.1-24b-instruct:free"

        finally:
            # Restore original content
            user_config_path.write_text(original_content)

    def test_get_preset_returns_complete_config(self, client):
        """GET /presets/default.yaml returns all config sections."""
        response = client.get("/presets/default.yaml")
        assert response.status_code == 200
        data = response.json()

        # Check all major sections are present
        assert "name" in data
        assert "description" in data
        assert "dataset" in data
        assert "embedding" in data
        assert "chunking" in data
        assert "retrieval" in data
        assert "reranker" in data
        assert "generation" in data
        assert "evaluation" in data
        assert "vector_db" in data
