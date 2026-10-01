"""On-disk cache for trained models, so the daily prediction job
(`backend/predict/log_predictions.py`) doesn't retrain all 5 models from
scratch every single day — spec Section 5.5: "Nightly: ingest, rebuild
features, re-score upcoming games. Weekly retrain." A cached model older
than `max_age_days` (default 7, matching "weekly retrain") is retrained and
overwritten; a fresh one is just loaded from disk, which takes a fraction
of a second instead of the ~30-60s a LightGBM training run costs.

Persisted under `data/models/` — already inside the Docker volume mount
(`./data:/app/data` in docker-compose.yml), so the cache survives between
container runs without any extra setup.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import lightgbm as lgb

CACHE_DIR = Path("data/models")


def get_or_train_booster(
    name: str, train_fn: Callable[[], tuple[lgb.Booster, dict]], max_age_days: int = 7
) -> tuple[lgb.Booster, dict]:
    """`train_fn()` returns `(booster, extra)` where `extra` is a small
    JSON-serializable dict of whatever else the model needs (e.g. `{"sigma":
    ...}` for the points model, `{"r": ...}` for a Negative Binomial
    market). Returns `(booster, extra)`, from cache if fresh enough, else
    retrains and overwrites the cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CACHE_DIR / f"{name}.txt"
    meta_path = CACHE_DIR / f"{name}.json"

    if model_path.exists() and meta_path.exists():
        age_days = (time.time() - model_path.stat().st_mtime) / 86400
        if age_days <= max_age_days:
            booster = lgb.Booster(model_file=str(model_path))
            extra = json.loads(meta_path.read_text())
            return booster, extra

    booster, extra = train_fn()
    booster.save_model(str(model_path))
    meta_path.write_text(json.dumps(extra))
    return booster, extra
