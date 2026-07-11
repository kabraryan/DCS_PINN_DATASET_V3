"""Guard the docs' load-bearing numbers against code drift.

This project's recurring failure is prose diverging from code: a typo across four
files, fabricated statistics, and — even after a "Provenance discipline" rule was
written — correction tables left carrying pre-solver-fix numbers while their own
amendment headers were corrected. `--check` on RESULTS.md guarded exactly one
file; every statistic hand-typed into the Corrections prose sat outside the guard,
and three drifted.

This extends the guard to those inline numbers. Each CLAIM pairs:
  - the exact fragment that must appear verbatim in a named doc, and
  - a `regen` callable that recomputes that fragment from the code/data.

The check asserts BOTH directions:
  1. `regen()` still equals the registered fragment  (else the CODE drifted; the
     doc is stale — regenerate it), and
  2. the fragment still appears in the doc            (else the DOC was edited away
     from the registry, or the registry is stale).

Expensive regens (ODE integration) are cached by the SHA-256 of their producing
source files, so they re-run only when the physics that produces them changes —
which is exactly when you want them re-verified. Cheap regens (reading the CSV)
always run.

Known limitation: matching is by substring, so if a number appears in several
places in a doc and only ONE instance drifts, the check still passes (the value is
found elsewhere). This is deliberate — the drift that actually happened in this
project replaced every instance consistently (the whole doc said 3.2%), which the
check does catch. Guarding per-occurrence would need line anchors and is not worth
the brittleness. Register the distinctive number, not its prose.

Usage:
    python scripts/check_doc_numbers.py            # list every claim + status
    python scripts/check_doc_numbers.py --check    # exit non-zero on any drift
    python scripts/check_doc_numbers.py --slow      # include the ODE-derived claims

Interpreter: /opt/miniconda3/bin/python3 (3.13). /usr/bin/python3 (3.9) will fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REAL_CSV = os.path.expanduser("~/Desktop/FINAL DIVE/datasets/real/dcs_all_dives.csv")
REAL_JSONL = os.path.expanduser("~/Desktop/FINAL DIVE/datasets/real/dcs_real_cases.jsonl")
CACHE = ROOT / ".doc_numbers_cache.json"

SPEC = "docs/specs/2026-06-23-pinn-bubble-dynamics-design.md"
RESULTS = "RESULTS.md"
MAX_BOUNCE = 300.0


@dataclass
class Claim:
    id: str
    docs: List[str]                       # every fragment must appear in each of these
    regen: Callable[[], List[str]]        # recompute the distinctive fragment(s)
    deps: List[str] = field(default_factory=list)  # content-hash gate for slow claims
    slow: bool = False                    # cached by deps-hash; skipped unless --slow


# ── regen helpers (cheap: read the real CSV directly) ────────────────────────

def _cohort():
    df = pd.read_csv(REAL_CSV)
    keep = ((df.depth_fsw <= MAX_BOUNCE) & (df.bottom_time_min <= MAX_BOUNCE)
            & (df.ascent_time_min <= MAX_BOUNCE))
    kept, dropped = df[keep], df[~keep]
    return {
        "total": len(df), "kept": len(kept), "dropped": len(dropped),
        "kept_pct": len(kept) / len(df) * 100,
        "kept_dcs": (kept.outcome == 1.0).mean() * 100,
        "dropped_dcs": (dropped.outcome == 1.0).mean() * 100,
    }


def _exclude_cohort():
    df = pd.read_csv(REAL_CSV)
    keep = ((df.depth_fsw <= MAX_BOUNCE) & (df.bottom_time_min <= MAX_BOUNCE)
            & (df.ascent_time_min <= MAX_BOUNCE))
    df = df[keep]
    df = df[df.outcome != 0.5]
    return {"n": len(df), "pos": int((df.outcome >= 1.0).sum()),
            "pct": (df.outcome >= 1.0).mean() * 100, "trials": df.data_set.nunique()}


def _reconstruction_rmse():
    """Median RMSE of staged vs linear reconstruction against the 428 gold curves.

    Replicates validate_reconstruction.py's core; ~15 s. Cheap enough to run live.
    """
    import json as _json
    from benchmark.profile import Dive, reconstruct
    recs = [_json.loads(l) for l in open(REAL_JSONL) if l.strip()]
    lin, stg = [], []
    for r in recs:
        s = r.get("depth_time_series") or []
        if len(s) < 8 or (r.get("quality_flags") or ""):
            continue
        if not all(r.get(k) and 0 < r[k] <= 300 for k in
                   ("max_depth_fsw", "bottom_time_min", "ascent_time_min")):
            continue
        t = np.array([q["t_min"] for q in s], float)
        d = np.array([q["depth_fsw"] for q in s], float)
        if np.any(np.diff(t) <= 0):
            continue
        dive = Dive(r.get("profile_number", "x"), r["max_depth_fsw"],
                    r["bottom_time_min"], r["ascent_time_min"], 1.0, r["data_set"])
        for recon, acc in (("linear", lin), ("staged", stg)):
            p = reconstruct(dive, recon)
            hi = min(t[-1], p.t_min[-1]); m = t <= hi
            if m.sum() < 5:
                continue
            acc.append(np.sqrt(np.mean((np.interp(t[m], p.t_min, p.depth_fsw) - d[m]) ** 2)))
    stg, lin = np.array(stg), np.array(lin)
    better = np.mean(stg < lin) * 100
    return {"staged": float(np.median(stg)), "linear": float(np.median(lin)),
            "better": float(better), "n": len(stg)}


def _c12_growth_r07():
    """OPT-1 (VPM skin) growth fraction at R0=0.7um — Correction 12's headline.

    SLOW: runs the EP ODE over the synthetic sampler via verify_nucleation_options.py.
    Cached by the SHA of ep_bubble.py + verify_nucleation_options.py so it re-runs
    only when the solver or the harness changes.
    """
    out = subprocess.check_output(
        ["/opt/miniconda3/bin/python3", "scripts/verify_nucleation_options.py"],
        cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
    for line in out.splitlines():
        if "OPT 1" in line and "R0=0.7um, skin, no ceiling" in line:
            # "... grew  16.4%  std ..."
            pct = float(line.split("grew")[1].split("%")[0].strip())
            return {"pct": pct}
    raise RuntimeError("could not parse OPT 1 R0=0.7um from verify_nucleation_options output")


# ── the registry ─────────────────────────────────────────────────────────────

# Each regen returns distinctive number-fragments that must appear VERBATIM in the
# doc. Bare 3-4-sig-fig numbers (36.08, 24.3%, 16.4%) are specific enough that a
# collision is implausible; keeping them minimal avoids the brittleness of matching
# full prose. If the code recomputes a different value, the fragment stops matching
# and the check fails — pointing at exactly which number drifted.
CLAIMS: List[Claim] = [
    # RESULTS.md is also --check-guarded; these cross-checks confirm the cohort
    # figures regenerate from the raw CSV, not just from the run that wrote RESULTS.
    Claim("cohort.kept_pct", [RESULTS],
          lambda: [f"kept {_cohort()['kept']} bounce dives ({_cohort()['kept_pct']:.1f}%)"]),
    Claim("cohort.dropped_rates", [RESULTS],
          lambda: [f"{_cohort()['dropped_dcs']:.1f}% vs {_cohort()['kept_dcs']:.1f}% kept"]),
    # SPEC prose — the un-gated numbers that drifted before.
    Claim("recon.rmse", [SPEC],
          lambda: [f"{_reconstruction_rmse()['staged']:.2f}",
                   f"{_reconstruction_rmse()['linear']:.2f}"]),
    Claim("recon.better_pct", [SPEC],
          lambda: [f"{_reconstruction_rmse()['better']:.1f}%"]),
    Claim("c12.growth_r07", [SPEC],
          lambda: [f"**{_c12_growth_r07()['pct']:.1f}%**"],
          deps=["benchmark/algorithms/ep_bubble.py", "scripts/verify_nucleation_options.py"],
          slow=True),
]


# ── check engine ─────────────────────────────────────────────────────────────

def _deps_hash(deps: List[str]) -> str:
    h = hashlib.sha256()
    for d in sorted(deps):
        h.update((ROOT / d).read_bytes())
    return h.hexdigest()[:16]


def _load_cache() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def _save_cache(c: dict) -> None:
    CACHE.write_text(json.dumps(c, indent=2, sort_keys=True))


def run(check: bool, include_slow: bool) -> int:
    cache = _load_cache()
    doc_text = {p: (ROOT / p).read_text() for p in {d for c in CLAIMS for d in c.docs}}
    failures = []

    for claim in CLAIMS:
        if claim.slow and not include_slow:
            print(f"  SKIP (slow)  {claim.id}  — run with --slow")
            continue

        # regen (cached for slow claims, keyed by producing-source hash)
        if claim.slow:
            key = f"{claim.id}:{_deps_hash(claim.deps)}"
            if key in cache:
                fragments, source = cache[key], "cached"
            else:
                fragments, source = claim.regen(), "regenerated"
                cache[key] = fragments
        else:
            fragments, source = claim.regen(), "live"

        # every regenerated fragment must appear verbatim in every listed doc
        missing = [(d, f) for d in claim.docs for f in fragments if f not in doc_text[d]]
        ok = not missing
        status = "OK  " if ok else "DRIFT"
        print(f"  {status} {claim.id:22s} [{source:11s}] {fragments}")
        for d, f in missing:
            failures.append((claim.id, d, f))
            print(f"        ↳ {f!r} not found in {d}")

    _save_cache(cache)

    if failures:
        print(f"\n{len(failures)} doc-number drift(s): the code now produces a value the doc "
              f"does not contain.\nRegenerate the doc text (or fix the code), then re-run.")
        return 1 if check else 0
    print("\nAll checked doc numbers match the code." + ("" if include_slow else
          "  (slow ODE claims skipped — add --slow to include.)"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero on drift (for CI)")
    ap.add_argument("--slow", action="store_true",
                    help="include ODE-derived claims (cached by source hash)")
    args = ap.parse_args()
    return run(check=args.check, include_slow=args.slow)


if __name__ == "__main__":
    raise SystemExit(main())
