"""Model pricing catalog (E9-F1-T1).

Loads ``config/pricing.yaml``, a hand-edited catalog of API prices per model
(and local compute-time estimates for embedding and reranker models), and
turns token counts into costs. Unknown prices stay ``None``: callers must
report a model as unpriced rather than assume it is free.

    catalog = PricingCatalog.load()
    catalog.llm_cost("z-ai/glm-5.3-flash", input_tokens=1200, output_tokens=300)
    catalog.unpriced(["z-ai/glm-5.3-flash", "openai/gpt-5.4-mini"])
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_PRICING_PATH = Path(__file__).resolve().parents[2] / "config" / "pricing.yaml"

ModelKind = Literal["llm", "embedding", "reranker"]
ModelRole = Literal[
    "generation",
    "judge",
    "ragas",
    "entity_extraction",
    "dataset_generation",
    "assistant",
]


class ModelPrice(BaseModel):
    """Price entry for one model."""

    kind: ModelKind = "llm"
    provider: Optional[str] = None
    roles: List[ModelRole] = Field(default_factory=list)
    input_per_1m: Optional[float] = Field(None, ge=0)
    output_per_1m: Optional[float] = Field(None, ge=0)
    seconds_per_1k_items: Optional[float] = Field(None, ge=0)
    checked: Optional[date] = None

    @model_validator(mode="after")
    def _fields_match_kind(self) -> "ModelPrice":
        if self.kind == "llm":
            if self.seconds_per_1k_items is not None:
                raise ValueError("seconds_per_1k_items is for local models only")
            if (self.input_per_1m is None) != (self.output_per_1m is None):
                raise ValueError("set both input_per_1m and output_per_1m, or neither")
        elif self.input_per_1m is not None or self.output_per_1m is not None:
            raise ValueError(f"{self.kind} models are local: use seconds_per_1k_items")
        return self

    @property
    def is_priced(self) -> bool:
        if self.kind == "llm":
            return self.input_per_1m is not None
        return self.seconds_per_1k_items is not None


class PricingCatalog(BaseModel):
    """All model prices, keyed by model id as used in configs."""

    currency: str = "USD"
    models: Dict[str, ModelPrice] = Field(default_factory=dict)

    # -- loading / saving ---------------------------------------------------

    @classmethod
    def load(cls, path: Union[str, Path, None] = None) -> "PricingCatalog":
        """Load and validate the catalog. A missing file gives an empty catalog."""
        p = Path(path) if path else DEFAULT_PRICING_PATH
        if not p.exists():
            return cls()
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def add_model(
        self,
        model: str,
        price: ModelPrice,
        path: Union[str, Path, None] = None,
    ) -> None:
        """Add a missing model and append it to the YAML file.

        Appends instead of rewriting so the hand-written comments survive.
        Used by the AutoML interview to record a price the user gave. Existing
        entries are never changed from code: edit the file by hand for that.
        """
        if model in self.models:
            raise ValueError(f"{model!r} is already in the pricing catalog")
        self.models[model] = price
        p = Path(path) if path else DEFAULT_PRICING_PATH
        entry = price.model_dump(mode="json", exclude_defaults=True)
        entry.setdefault("kind", price.kind)
        block = yaml.safe_dump({model: entry}, sort_keys=False, allow_unicode=True)
        indented = "".join(f"  {line}\n" for line in block.splitlines())
        text = p.read_text(encoding="utf-8") if p.exists() else "models:\n"
        if not text.endswith("\n"):
            text += "\n"
        p.write_text(text + indented, encoding="utf-8")

    # -- lookups -----------------------------------------------------------

    def get(self, model: str) -> Optional[ModelPrice]:
        return self.models.get(model)

    def llm_cost(self, model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
        """Cost of one or more calls, or ``None`` if the model has no price."""
        price = self.models.get(model)
        if price is None or price.kind != "llm" or not price.is_priced:
            return None
        return (input_tokens * price.input_per_1m + output_tokens * price.output_per_1m) / 1_000_000

    def local_seconds(self, model: str, n_items: int) -> Optional[float]:
        """Estimated local compute time for ``n_items`` (chunks or pairs)."""
        price = self.models.get(model)
        if price is None or price.seconds_per_1k_items is None:
            return None
        return price.seconds_per_1k_items * n_items / 1000

    def unpriced(self, models: Iterable[str]) -> List[str]:
        """Models (in input order, deduplicated) with no usable price."""
        seen: Dict[str, None] = {}
        for m in models:
            price = self.models.get(m)
            if price is None or not price.is_priced:
                seen.setdefault(m, None)
        return list(seen)

    def models_for_role(self, role: ModelRole) -> List[str]:
        return [m for m, p in self.models.items() if role in p.roles]
