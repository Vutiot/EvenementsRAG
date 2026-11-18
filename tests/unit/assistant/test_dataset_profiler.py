"""Tests for the deterministic dataset profiling harness (E8-F1-T1)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.assistant.dataset_profiler import (
    DatasetProfile,
    build_profile,
    detect_language,
    extract_date_range,
    get_profile,
    kmeans,
)

THEMES = {
    "naval": (
        "Battle of Midway",
        "The Imperial Japanese Navy lost four aircraft carriers near Midway Atoll in June 1942. "
        "Admiral Chester Nimitz directed the carrier fleet and the dive bombers.",
    ),
    "eastern": (
        "Battle of Stalingrad",
        "The Red Army encircled the German Sixth Army at Stalingrad in the winter of 1942 "
        "and 1943. General Friedrich Paulus surrendered the encircled army.",
    ),
    "diplomacy": (
        "Yalta Conference",
        "Winston Churchill, Franklin Roosevelt and Joseph Stalin met at Yalta in February "
        "1945 to discuss the postwar reorganisation of Europe and the United Nations.",
    ),
}


def _fixture_records(per_theme: int = 6, dim: int = 16):
    rng = np.random.default_rng(42)
    records = []
    for t_idx, (theme, (title, text)) in enumerate(THEMES.items()):
        base = np.zeros(dim)
        base[t_idx] = 1.0
        for j in range(per_theme):
            vec = base + rng.normal(0, 0.05, dim)
            records.append(
                {
                    "id": f"{theme}-{j}",
                    "vector": vec.tolist(),
                    "payload": {
                        "content": f"{text} Paragraph {j}.",
                        "article_title": title if j < per_theme - 1 else f"{title} (aftermath)",
                        "chunk_id": f"{theme}_{j}",
                        "token_count": 40 + j,
                        "categories": ["World War II", f"Category {theme}"],
                    },
                }
            )
    return records


@pytest.fixture
def records():
    return _fixture_records()


def test_build_profile_basic_fields(records):
    p = build_profile("ww2_test", records, n_clusters=3, use_spacy=False)
    assert p.n_chunks == 18
    assert p.n_documents == 6
    assert p.chunk_tokens.min == 40 and p.chunk_tokens.max == 45
    assert p.language == "en"
    assert p.date_range.min_year == 1942 and p.date_range.max_year == 1945
    assert p.date_range.chunks_with_year == 1.0
    assert p.top_categories[0].text == "World War II"
    assert p.entity_method == "heuristic"
    names = {e.text for e in p.top_entities}
    assert "Winston Churchill" in names or "Chester Nimitz" in names


def test_clusters_recover_the_three_themes(records):
    p = build_profile("ww2_test", records, n_clusters=3, use_spacy=False)
    assert len(p.clusters) == 3
    assert sorted(c.size for c in p.clusters) == [6, 6, 6]
    for c in p.clusters:
        # every representative of a cluster comes from the same theme
        prefixes = {r.chunk_id.split("_")[0] for r in c.representatives}
        assert len(prefixes) == 1
        assert len(c.representatives) == 3
        assert c.top_terms


def test_profile_is_deterministic_and_order_independent(records):
    a = build_profile("ww2_test", records, n_clusters=3, use_spacy=False)
    b = build_profile("ww2_test", list(reversed(records)), n_clusters=3, use_spacy=False)
    assert a.model_dump(exclude={"created_at"}) == b.model_dump(exclude={"created_at"})


def test_profile_is_compact(records):
    p = build_profile("ww2_test", records * 50, n_clusters=8, use_spacy=False)
    assert p.approx_tokens() < 4000


def test_no_vectors_means_no_clusters(records):
    for r in records:
        r.pop("vector")
    p = build_profile("ww2_test", records, use_spacy=False)
    assert p.clusters == []
    assert p.n_chunks == 18


def test_empty_collection():
    p = build_profile("empty", [], use_spacy=False)
    assert p.n_chunks == 0 and p.date_range is None and p.clusters == []


def test_detect_language():
    assert detect_language(["The army was moved to the front and the city fell."]) == "en"
    assert detect_language(["La ville est tombée et les troupes sont parties pour le front."]) == "fr"
    assert detect_language(["12345 67890"]) == "unknown"


def test_extract_date_range_ignores_texts_without_years():
    dr = extract_date_range(["in 1939 and 1945", "no year here"])
    assert (dr.min_year, dr.max_year, dr.chunks_with_year) == (1939, 1945, 0.5)
    assert extract_date_range(["nothing"]) is None


def test_kmeans_k_larger_than_points():
    x = np.eye(3, dtype=np.float32)
    labels, centroids = kmeans(x, 8)
    assert centroids.shape[0] == 3 and len(set(labels.tolist())) == 3


def _fake_store(records):
    store = MagicMock()
    store.collection_exists.return_value = True
    half = len(records) // 2
    store.scroll.side_effect = [(records[:half], "next"), (records[half:], None)]
    return store


def test_get_profile_caches_on_disk(records, tmp_path):
    store = _fake_store(records)
    p1 = get_profile("ww2_test", store=store, profiles_dir=tmp_path, use_spacy=False, n_clusters=3)
    assert (tmp_path / "ww2_test.json").exists()
    assert store.scroll.call_count == 2
    p2 = get_profile("ww2_test", store=store, profiles_dir=tmp_path)
    assert store.scroll.call_count == 2  # served from disk
    assert p2 == p1


def test_get_profile_refresh_and_errors(records, tmp_path):
    store = _fake_store(records)
    get_profile("ww2_test", store=store, profiles_dir=tmp_path, use_spacy=False)
    store.scroll.side_effect = [(records, None)]
    get_profile("ww2_test", refresh=True, store=store, profiles_dir=tmp_path, use_spacy=False)
    assert store.scroll.call_count == 3

    with pytest.raises(ValueError):
        get_profile("../etc/passwd", store=store, profiles_dir=tmp_path)
    store.collection_exists.return_value = False
    with pytest.raises(KeyError):
        get_profile("missing", store=store, profiles_dir=tmp_path)


def test_profile_endpoint(monkeypatch, records):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.routers import assistant

    profile = build_profile("ww2_test", records, n_clusters=3, use_spacy=False)

    def fake_get_profile(collection, refresh=False):
        if collection == "missing":
            raise KeyError(collection)
        return profile

    monkeypatch.setattr(assistant, "get_profile", fake_get_profile)
    app = FastAPI()
    app.include_router(assistant.router, prefix="/api")
    client = TestClient(app)
    r = client.get("/api/assistant/profile/ww2_test")
    assert r.status_code == 200
    assert DatasetProfile.model_validate(r.json()).n_chunks == 18
    assert client.get("/api/assistant/profile/missing").status_code == 404


def test_spacy_entities_when_a_model_is_installed(records):
    from src.assistant.dataset_profiler import _load_spacy

    if _load_spacy() is None:
        pytest.skip("no spaCy English model installed")
    p = build_profile("ww2_test", records, n_clusters=3)
    assert p.entity_method.startswith("spacy:")
    labels = {e.text: e.label for e in p.top_entities}
    assert labels.get("Winston Churchill") == "PERSON"
    assert "United Nations" in labels  # leading "the" stripped
