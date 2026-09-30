"""Small shared helpers for ingestors."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

logger = logging.getLogger("ingest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

T = TypeVar("T")


@dataclass
class IngestResult:
    rows_written: int
    notes: dict[str, Any] = field(default_factory=dict)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def retry(fn: Callable[[], T], attempts: int = 4, base_delay: float = 2.0, what: str = "call") -> T:
    """Retry a flaky network call with exponential backoff.

    nba_api / stats.nba.com and similar unofficial endpoints intermittently
    time out or 403 under normal use; a short retry loop is cheap insurance.
    """
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a generic retry wrapper
            last_exc = exc
            if attempt == attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning("%s failed (attempt %d/%d): %s — retrying in %.1fs", what, attempt, attempts, exc, delay)
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc
