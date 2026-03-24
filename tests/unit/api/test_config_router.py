"""Unit tests for src/api/routers/config.py - preset configuration endpoints.

Covers:
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


class TestGetPreset:
    def test_get_preset_default(self, client):
        """GET /presets/default.yaml returns the default config."""
        response = client.get("/presets/default.yaml")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "default"

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
