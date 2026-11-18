"""Tests for the model pricing catalog (E9-F1-T1)."""

from datetime import date

import pytest
from pydantic import ValidationError

from src.automl.pricing import DEFAULT_PRICING_PATH, ModelPrice, PricingCatalog

YAML = """\
# hand-written comment
currency: USD
models:
  priced-llm:
    kind: llm
    roles: [generation]
    input_per_1m: 0.5
    output_per_1m: 2.0
    checked: 2026-11-18
  free-llm:
    input_per_1m: 0
    output_per_1m: 0
  unknown-llm:
    roles: [judge]
    input_per_1m: null
    output_per_1m: null
  local-emb:
    kind: embedding
    seconds_per_1k_items: 4.0
"""


@pytest.fixture
def catalog_path(tmp_path):
    p = tmp_path / "pricing.yaml"
    p.write_text(YAML)
    return p


def test_shipped_catalog_is_valid_and_has_no_fake_prices():
    catalog = PricingCatalog.load(DEFAULT_PRICING_PATH)
    assert "z-ai/glm-5.3-flash" in catalog.models
    assert all(not p.is_priced for p in catalog.models.values())


def test_load_and_llm_cost(catalog_path):
    catalog = PricingCatalog.load(catalog_path)
    assert catalog.get("priced-llm").checked == date(2026, 11, 18)
    assert catalog.llm_cost("priced-llm", 1_000_000, 500_000) == pytest.approx(1.5)
    assert catalog.llm_cost("free-llm", 10_000, 10_000) == 0
    assert catalog.llm_cost("unknown-llm", 10, 10) is None
    assert catalog.llm_cost("not-in-catalog", 10, 10) is None


def test_local_seconds(catalog_path):
    catalog = PricingCatalog.load(catalog_path)
    assert catalog.local_seconds("local-emb", 2500) == pytest.approx(10.0)
    assert catalog.local_seconds("priced-llm", 100) is None


def test_unpriced_and_roles(catalog_path):
    catalog = PricingCatalog.load(catalog_path)
    assert catalog.unpriced(["priced-llm", "unknown-llm", "x", "x"]) == ["unknown-llm", "x"]
    assert catalog.models_for_role("judge") == ["unknown-llm"]


def test_missing_file_gives_empty_catalog(tmp_path):
    assert PricingCatalog.load(tmp_path / "nope.yaml").models == {}


@pytest.mark.parametrize(
    "entry",
    [
        {"input_per_1m": 1.0},  # output missing
        {"input_per_1m": -1, "output_per_1m": 1},  # negative
        {"kind": "embedding", "input_per_1m": 1, "output_per_1m": 1},
        {"kind": "llm", "seconds_per_1k_items": 3},
        {"roles": ["cooking"]},
    ],
)
def test_invalid_entries_rejected(entry):
    with pytest.raises(ValidationError):
        ModelPrice.model_validate(entry)


def test_add_model_appends_and_keeps_comments(catalog_path):
    catalog = PricingCatalog.load(catalog_path)
    catalog.add_model(
        "new/model",
        ModelPrice(roles=["assistant"], input_per_1m=1.0, output_per_1m=3.0),
        path=catalog_path,
    )
    text = catalog_path.read_text()
    assert text.startswith("# hand-written comment")
    reloaded = PricingCatalog.load(catalog_path)
    assert reloaded.llm_cost("new/model", 1_000_000, 0) == pytest.approx(1.0)
    assert reloaded.get("priced-llm").input_per_1m == 0.5
    with pytest.raises(ValueError):
        catalog.add_model("new/model", ModelPrice(), path=catalog_path)
