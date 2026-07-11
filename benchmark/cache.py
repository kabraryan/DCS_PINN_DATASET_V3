"""Content-addressed cache for per-dive algorithm scalars.

Two boundaries: profile reconstruction is invariant to the CV protocol, and the
CV protocol is invariant to algorithm internals. Caching at both means adding an
algorithm recomputes only that algorithm's column, and re-running statistics
recomputes nothing.

JSON only. `joblib.load` on an untrusted pickle is arbitrary code execution, and
these are single floats.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Optional


def cache_key(*parts: object) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path(root: Path, key: str) -> Path:
    return Path(root) / key[:2] / f"{key}.json"


def cached(root: Path, key: str, compute: Callable[[], Optional[float]]) -> Optional[float]:
    p = _path(root, key)
    if p.exists():
        return json.loads(p.read_text())["v"]
    value = compute()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"v": value}))
    return value


def clear(root: Path) -> None:
    for p in Path(root).rglob("*.json"):
        p.unlink()
