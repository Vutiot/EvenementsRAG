"""Preset configuration endpoints."""

from fastapi import APIRouter, HTTPException

from src.api.dependencies import PRESETS_DIR
from src.api.schemas import PresetInfo
from src.benchmarks.config import BenchmarkConfig

router = APIRouter()


@router.get("/presets", response_model=list[PresetInfo])
def list_presets():
    """List all YAML preset files from config/benchmarks/."""
    presets = []
    for path in sorted(PRESETS_DIR.glob("*.yaml")):
        try:
            cfg = BenchmarkConfig.from_yaml(path)
            presets.append(PresetInfo(
                filename=path.name,
                name=cfg.name,
                description=cfg.description,
            ))
        except Exception:
            presets.append(PresetInfo(
                filename=path.name,
                name=path.stem,
                description="(failed to parse)",
            ))
    return presets


@router.get("/presets/{filename}")
def get_preset(filename: str):
    """Return full config as JSON for a specific preset."""
    path = PRESETS_DIR / filename
    if not path.exists() or not path.suffix == ".yaml":
        raise HTTPException(status_code=404, detail=f"Preset '{filename}' not found")
    # Prevent path traversal
    if not path.resolve().parent == PRESETS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid filename")
    cfg = BenchmarkConfig.from_yaml(path)
    return cfg.model_dump()
