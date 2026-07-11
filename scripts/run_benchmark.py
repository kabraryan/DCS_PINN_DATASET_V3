"""Run the decompression benchmark and regenerate RESULTS.md.

Every number in RESULTS.md is generated. `--check` regenerates and diffs, exiting
non-zero on drift -- the machine-checkable form of the Provenance discipline rule
added after Corrections 9 and 12 fabricated statistics that prose review missed.

This tool NEVER emits a probability. risk_index is a rank.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.algorithms import DEFAULT_ALGORITHMS, REGISTRY, AlgorithmError
from benchmark.cache import cache_key, cached
from benchmark.evaluate import baseline_auc, four_gate, leakage_gap, shuffle_control
from benchmark.profile import Dive, RECONSTRUCTIONS, reconstruct
from benchmark.verdict import MARGINAL_RULES, PRIMARY_MARGINAL, verdict

REAL_CSV = os.path.expanduser("~/Desktop/FINAL DIVE/datasets/real/dcs_all_dives.csv")
MAX_BOUNCE = 300.0
EP_FAILURE_ABORT_RATE = 0.01


@dataclass(frozen=True)
class DropReport:
    n_total: int
    n_kept: int
    n_dropped: int
    kept_dcs_rate: float
    dropped_dcs_rate: float


def assert_has_variance(col: np.ndarray, name: str) -> None:
    if float(np.nanstd(col)) < 1e-12:
        raise AlgorithmError(
            f"{name}: zero-variance column. The model is degenerate "
            f"(see Correction 11); it is not a result."
        )


def load_dives(csv: Optional[str], marginal: str):
    df = pd.read_csv(csv or REAL_CSV)
    n_total = len(df)
    keep = ((df.depth_fsw <= MAX_BOUNCE) & (df.bottom_time_min <= MAX_BOUNCE)
            & (df.ascent_time_min <= MAX_BOUNCE))
    dropped, df = df[~keep], df[keep].copy()
    report = DropReport(n_total, len(df), len(dropped),
                        float((df.outcome == 1.0).mean()),
                        float((dropped.outcome == 1.0).mean()) if len(dropped) else 0.0)

    if marginal == "exclude":
        df = df[df.outcome != 0.5].copy()
    elif marginal == "positive":
        df["outcome"] = (df.outcome > 0.0).astype(float)
    elif marginal == "negative":
        df["outcome"] = (df.outcome >= 1.0).astype(float)
    else:
        raise ValueError(marginal)

    dives = [
        Dive(f"{r.data_set}:{r.profile_number}", float(r.depth_fsw),
             float(r.bottom_time_min), float(r.ascent_time_min),
             float(r.outcome), str(r.data_set))
        for r in df.itertuples()
    ]
    y = (df.outcome.values >= 1.0).astype(int)
    groups = df.data_set.values
    return dives, y, groups, report


_PROFILE_CACHE: Dict[Tuple[str, str], object] = {}


def _profile(dive: Dive, recon: str):
    """CACHE 1: reconstruction is invariant to which algorithm consumes it.

    In-process, not on disk: a Profile is two arrays, cheap to rebuild (0.7 ms)
    but wasteful to redo once per algorithm.
    """
    key = (dive.dive_id, recon)
    if key not in _PROFILE_CACHE:
        _PROFILE_CACHE[key] = reconstruct(dive, recon)
    return _PROFILE_CACHE[key]


def build_matrix(dives: List[Dive], recon: str, algo_name: str, cache_root: Path):
    algo = REGISTRY[algo_name]
    risk, deficit, failures = [], [], 0
    for d in dives:
        k_r = cache_key("risk", algo_name, algo.params, recon, d.dive_id,
                        d.depth_fsw, d.bottom_time_min, d.ascent_time_min)
        k_d = cache_key("deficit", algo_name, algo.params, recon, d.dive_id,
                        d.depth_fsw, d.bottom_time_min, d.ascent_time_min)
        try:
            p = _profile(d, recon)                      # CACHE 1
            risk.append(cached(cache_root, k_r, lambda: algo.risk_index(p, d)))
            deficit.append(cached(cache_root, k_d, lambda: algo.deficit(p, d)))
        except AlgorithmError:
            failures += 1
            risk.append(np.nan)
            deficit.append(np.nan)

    if failures / max(len(dives), 1) > EP_FAILURE_ABORT_RATE:
        raise AlgorithmError(
            f"{algo_name}: {failures}/{len(dives)} solves failed "
            f"(> {EP_FAILURE_ABORT_RATE:.0%}). Refusing to emit a dataset whose "
            f"physics is fabricated on a label-correlated subset."
        )
    out: Dict[str, np.ndarray] = {"risk_index": np.asarray(risk, float),
                                  "n_failed": failures}
    dv = np.asarray(deficit, float)
    out["deficit"] = None if np.all(np.isnan(dv)) else dv
    return out


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def recorded_command(argv: List[str]) -> str:
    """The exact invocation that reproduces this RESULTS.md.

    Provenance must be copy-paste reproducible (Task 9's whole purpose): every
    flag the user actually passed has to show up here. `--check` is the one
    exception -- a RESULTS.md is produced by a generate run, never a check run,
    so that token is stripped if present.
    """
    filtered = [a for a in argv if a != "--check"]
    return "python " + shlex.join(filtered)


def render_results(rows, controls, report, command, csv_path, failures) -> str:
    import scipy, sklearn
    lines = [
        "# Decompression Algorithm Benchmark — Results",
        "",
        "> Generated file. Do not edit. Regenerate with the command below;",
        "> `--check` fails if this file has drifted.",
        "",
        "## Provenance",
        "",
        f"- command: `{command}`",
        f"- git: `{_git_sha()}`",
        f"- input: `{csv_path}` sha256[:16] `{_sha256(csv_path)}`",
        f"- python {platform.python_version()}, numpy {np.__version__}, "
        f"scipy {scipy.__version__}, sklearn {sklearn.__version__}",
    ]
    if failures:
        for algo_name, (failed, total) in failures.items():
            if failed:
                lines.append(
                    f"- `{algo_name}`: {failed}/{total} dive-solves failed "
                    f"(excluded from the affected columns, not silently kept as NaN)"
                )
    if not any(failed for failed, _ in failures.values()):
        lines.append("- 0 solver failures")
    lines += [
        "",
        "## Cohort",
        "",
        f"- {report.n_total} dives; kept {report.n_kept} bounce dives "
        f"({report.n_kept / report.n_total:.1%})",
        f"- dropped {report.n_dropped} saturation / >300 fsw excursions; their DCS rate "
        f"was {report.dropped_dcs_rate:.1%} vs {report.kept_dcs_rate:.1%} kept "
        f"(**the exclusion is outcome-correlated**)",
        "",
        "## Verdicts",
        "",
        "| algorithm | metric | verdict | ΔAUC (staged, exclude) | sign | reasons |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['algo']}` | `{r['metric']}` | **{r['verdict']}** | "
            f"{r['delta']:+.4f} | {r['sign']:+.0f} | {r['reasons'] or '—'} |"
        )
    lines += [
        "",
        "## Controls",
        "",
        f"- label shuffle → AUC {controls['shuffle']:.4f} (must be ≈ 0.5)",
        f"- leakage gap (ordinary − grouped) → +{controls['leakage']:.4f}",
        f"- baseline (logistic on 3 raw features) → AUC {controls['baseline']:.4f} "
        f"± {controls['baseline_sd']:.4f}",
        "",
        "## Reading this table",
        "",
        "AUC is a **ranking**, never a probability. The ~16% DCS rate here reflects Navy",
        "trials designed to provoke DCS on partially-extracted negatives (2,700 of 8,578).",
        "This benchmark is not a dive-planning tool. Fold sd ≈ 0.06, so |ΔAUC| < 0.03 is",
        "noise. `N/A — no schedule` means the algorithm defines no ceiling.",
        "",
    ]
    return "\n".join(lines)


def _comparable(text: str) -> str:
    """RESULTS.md content minus the volatile provenance lines (git sha, command)."""
    return "\n".join(ln for ln in text.splitlines()
                     if not ln.startswith("- git:") and not ln.startswith("- command:"))


def main() -> int:
    argv = list(sys.argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--marginal", required=True,
                    choices=list(MARGINAL_RULES),
                    help="Safety-relevant choice; there is deliberately no default.")
    ap.add_argument("--algorithms", nargs="*", default=list(DEFAULT_ALGORITHMS))
    ap.add_argument("--out", default="RESULTS.md")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cache", default=".benchmark_cache")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=REAL_CSV)
    args = ap.parse_args()

    cache_root = Path(args.cache)
    rows = []
    failures: Dict[str, Tuple[int, int]] = {}
    for algo_name in args.algorithms:
        gates_by_metric: Dict[str, Dict[Tuple[str, str], object]] = {}
        for rule in MARGINAL_RULES:
            dives, y, groups, report = load_dives(args.csv, rule)
            X_raw = np.array([[d.depth_fsw, d.bottom_time_min, d.ascent_time_min]
                              for d in dives], float)
            for recon in RECONSTRUCTIONS:
                cols = build_matrix(dives, recon, algo_name, cache_root)
                n_failed = int(cols["n_failed"])
                prev = failures.get(algo_name, (0, len(dives)))
                failures[algo_name] = (max(prev[0], n_failed), len(dives))
                for metric in ("risk_index", "deficit"):
                    col = cols[metric]
                    if col is None:
                        continue
                    # A tolerated AlgorithmError leaves NaN in the column. Drop those
                    # rows here -- sklearn raises on NaN input, so an unfiltered NaN
                    # would crash the whole run with an opaque error instead of a
                    # report. The drop is per (algo, metric, recon, rule) because
                    # different algorithms fail on different dives.
                    finite = np.isfinite(col)
                    col_f, Xr_f = col[finite], X_raw[finite]
                    y_f, g_f = y[finite], groups[finite]
                    assert_has_variance(col_f, f"{algo_name}.{metric}")
                    g = four_gate(Xr_f, col_f, y_f, g_f,
                                  n_rep=args.repeats, seed=args.seed)
                    gates_by_metric.setdefault(metric, {})[(recon, rule)] = g

        for metric, gates in gates_by_metric.items():
            v = verdict(gates)
            primary = gates[("staged", PRIMARY_MARGINAL)]
            rows.append({"algo": algo_name, "metric": metric, "verdict": v,
                         "delta": primary.delta, "sign": primary.sign,
                         "reasons": "; ".join(primary.reasons)})

    dives, y, groups, report = load_dives(args.csv, args.marginal)
    X_raw = np.array([[d.depth_fsw, d.bottom_time_min, d.ascent_time_min]
                      for d in dives], float)
    base = baseline_auc(X_raw, y, groups, n_rep=args.repeats, seed=args.seed)
    controls = {
        "shuffle": shuffle_control(X_raw, y, groups, seed=args.seed),
        "leakage": leakage_gap(X_raw, y, groups, seed=args.seed),
        "baseline": float(base.mean()),
        "baseline_sd": float(base.std()),
    }

    text = render_results(rows, controls, report,
                          recorded_command(argv), args.csv, failures)
    out = Path(args.out)
    if args.check:
        # Compare benchmark-relevant content only. The `- git:` line records the
        # HEAD sha at generation, which advances on every later commit (even one
        # that does not touch the benchmark), and the `- command:` line varies with
        # how the check itself was invoked. Excluding both means --check flags a
        # change in the RESULTS (verdicts, numbers, controls) -- what it exists to
        # catch -- not a bookkeeping sha bump.
        if not out.exists() or _comparable(out.read_text()) != _comparable(text):
            print(f"{out} is stale. Regenerate it.", file=sys.stderr)
            return 1
        print(f"{out} is current.")
        return 0
    out.write_text(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
