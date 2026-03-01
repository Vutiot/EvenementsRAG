"""Tests for preset config endpoints."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestListPresets:
    def test_returns_list(self):
        response = client.get("/api/presets")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 24

    def test_preset_has_required_fields(self):
        response = client.get("/api/presets")
        first = response.json()[0]
        assert "filename" in first
        assert "name" in first
        assert "description" in first

    def test_all_presets_have_yaml_extension(self):
        response = client.get("/api/presets")
        for preset in response.json():
            assert preset["filename"].endswith(".yaml")

    def test_phase1_vanilla_present(self):
        response = client.get("/api/presets")
        filenames = [p["filename"] for p in response.json()]
        assert "phase1_vanilla.yaml" in filenames


class TestGetPreset:
    def test_returns_full_config(self):
        response = client.get("/api/presets/phase1_vanilla.yaml")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "phase1_vanilla"
        assert "dataset" in data
        assert "embedding" in data
        assert "chunking" in data
        assert "retrieval" in data
        assert "generation" in data
        assert "evaluation" in data
        assert "vector_db" in data

    def test_config_values_match_preset(self):
        response = client.get("/api/presets/phase1_vanilla.yaml")
        data = response.json()
        assert data["retrieval"]["technique"] == "vanilla"
        assert data["chunking"]["chunk_size"] == 512
        assert data["embedding"]["dimension"] == 384

    def test_not_found(self):
        response = client.get("/api/presets/nonexistent.yaml")
        assert response.status_code == 404

    def test_invalid_extension(self):
        response = client.get("/api/presets/something.json")
        assert response.status_code == 404

    def test_path_traversal_blocked(self):
        response = client.get("/api/presets/..%2F..%2Fetc%2Fpasswd")
        assert response.status_code in (400, 404)
