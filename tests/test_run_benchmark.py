import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmark.algorithms import REGISTRY
from benchmark.algorithms.base import AlgorithmError
from benchmark.profile import Dive, reconstruct

PY = "/opt/miniconda3/bin/python3"
ROOT = Path(__file__).resolve().parents[1]


def test_noise_algorithm_is_registered_and_has_variance():
    algo = REGISTRY["noise"]
    vals = [algo.risk_index(reconstruct(Dive(f"d{i}", 100, 40, 20, 0.0, "T"), "linear"),
                            Dive(f"d{i}", 100, 40, 20, 0.0, "T")) for i in range(20)]
    assert np.std(vals) > 1e-9


def test_constant_column_raises_rather_than_scoring():
    """A zero-variance column is a degenerate model (Correction 11), not a result."""
    from scripts.run_benchmark import assert_has_variance
    with pytest.raises(AlgorithmError):
        assert_has_variance(np.full(20, 3.0), "constant")


def test_bounce_window_drops_saturation_dives_and_reports_them():
    from scripts.run_benchmark import load_dives
    dives, y, g, report = load_dives(None, "exclude")
    assert report.n_dropped > 0
    assert report.dropped_dcs_rate > report.kept_dcs_rate, \
        "the exclusion is outcome-correlated and must be reported, not hidden"


def test_marginal_flag_is_required():
    r = subprocess.run([PY, "scripts/run_benchmark.py"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "--marginal" in (r.stderr + r.stdout)


def test_check_fails_on_a_stale_results_file(tmp_path):
    stale = tmp_path / "RESULTS.md"
    stale.write_text("# stale\n")
    r = subprocess.run(
        [PY, "scripts/run_benchmark.py", "--marginal", "exclude",
         "--out", str(stale), "--check", "--repeats", "1"],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0
