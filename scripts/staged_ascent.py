"""Staged-decompression profile reconstruction for the real NMRC 99-02 dives.

`dcs_all_dives.csv` records only (depth_fsw, bottom_time_min, ascent_time_min).
The earlier reconstruction modelled a LINEAR ascent, but these were staged
decompression dives: travel between stops, then hold. Linear ascent overstates
supersaturation early and understates late, and plausibly depresses Buhlmann's
discrimination (AUC ~0.60).

Reconstruction:
  1. Descend at 60 fsw/min, hold bottom time.
  2. Generate a schedule from the Buhlmann ceiling: ascend at 30 fsw/min toward
     the shallowest depth the tissues currently tolerate, rounded up to a 10-fsw
     stop; hold there until the ceiling clears; repeat to the surface.
  3. Rescale the *stop* durations (never the travel legs) so total ascent time
     equals the dive's RECORDED ascent_time_min.

Step 3 is what keeps this honest. A dive whose recorded ascent is shorter than
the schedule Buhlmann would demand gets its stops compressed, so it still
violates M-values and still produces prs > 1. The reconstruction supplies the
*shape* of a staged ascent; the data supplies its *duration*. Without the
rescale, every profile would respect the ceiling by construction, prs would be
pinned at 1.0, and the feature would carry no variance -- a circularity.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Shared Bühlmann primitives, single source of truth. Duplicating these locally is
# how compartment 16's b-coefficient once diverged across four files (Correction 10).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.buhlmann import (  # noqa: E402
    ASCENT_FSW_PER_MIN, DESCENT_FSW_PER_MIN, FSW_TO_BAR, F_N2_AIR, P_SURFACE,
    STOP_INCREMENT_FSW, haldane_step,
)

DT_MIN = 0.5
MAX_STOP_ITERS = 20000

_step = haldane_step  # local alias; identical to benchmark.buhlmann.haldane_step


def _ceiling_fsw(P_t, a, b):
    """Shallowest ambient pressure the tissues tolerate, in fsw (0 = surface ok)."""
    tol_bar = np.max((P_t - a) * b)
    return max(0.0, (tol_bar - P_SURFACE) / FSW_TO_BAR)


def _schedule(P_t, depth_fsw, a, b, k):
    """Buhlmann-ceiling ascent schedule. Returns (segments, travel_min, stop_min).

    segments: list of ('travel', from_fsw, to_fsw, minutes) | ('stop', d, d, minutes)
    Tissue state is advanced along the way (on a copy).
    """
    P = P_t.copy()
    d = float(depth_fsw)
    segs, travel, stop = [], 0.0, 0.0

    for _ in range(MAX_STOP_ITERS):
        if d <= 0.0:
            break
        ceil_fsw = _ceiling_fsw(P, a, b)
        target = np.ceil(ceil_fsw / STOP_INCREMENT_FSW) * STOP_INCREMENT_FSW
        target = min(target, d)

        if target < d:                      # can ascend
            dz = d - target
            dt = dz / ASCENT_FSW_PER_MIN
            mid = (d + target) / 2.0        # midpoint ambient over the leg
            P = _step(P, (P_SURFACE + mid * FSW_TO_BAR) * F_N2_AIR, k, dt)
            segs.append(('travel', d, target, dt))
            travel += dt
            d = target
        else:                               # blocked: hold a stop
            P = _step(P, (P_SURFACE + d * FSW_TO_BAR) * F_N2_AIR, k, DT_MIN)
            if segs and segs[-1][0] == 'stop' and segs[-1][1] == d:
                s = segs[-1]
                segs[-1] = ('stop', d, d, s[3] + DT_MIN)
            else:
                segs.append(('stop', d, d, DT_MIN))
            stop += DT_MIN
    return segs, travel, stop


def build_profile(depth_fsw, bt_min, at_min, table, surface_watch_min=120.0):
    """Return (t_min, depth_fsw) for a staged-ascent reconstruction."""
    a, b = table[:, 1], table[:, 2]
    k = np.log(2.0) / table[:, 0]

    # descent + bottom, tracking tissues so the schedule starts from the right state
    desc_min = max(depth_fsw / DESCENT_FSW_PER_MIN, DT_MIN)
    P = np.full(16, P_SURFACE * F_N2_AIR)
    n_desc = max(int(np.ceil(desc_min / DT_MIN)), 1)
    for i in range(n_desc):
        z = depth_fsw * (i + 1) / n_desc
        P = _step(P, (P_SURFACE + z * FSW_TO_BAR) * F_N2_AIR, k, desc_min / n_desc)
    n_bot = max(int(np.ceil(bt_min / DT_MIN)), 1)
    for _ in range(n_bot):
        P = _step(P, (P_SURFACE + depth_fsw * FSW_TO_BAR) * F_N2_AIR, k, bt_min / n_bot)

    segs, travel, stop = _schedule(P, depth_fsw, a, b, k)

    # Rescale stop time so the ascent lasts exactly as long as recorded.
    if at_min <= travel or stop <= 0.0:
        # Recorded ascent is faster than the travel legs alone (or no stops needed):
        # fall back to a straight ascent over the recorded time.
        segs = [('travel', depth_fsw, 0.0, max(at_min, DT_MIN))]
        scale = 0.0
    else:
        scale = (at_min - travel) / stop

    # materialise the profile on a uniform grid
    times, depths = [0.0], [0.0]

    def extend(dur, z0, z1):
        if dur <= 0:
            return
        n = max(int(np.ceil(dur / DT_MIN)), 1)
        for i in range(1, n + 1):
            times.append(times[-1] + dur / n)
            depths.append(z0 + (z1 - z0) * i / n)

    extend(desc_min, 0.0, depth_fsw)
    extend(bt_min, depth_fsw, depth_fsw)
    for kind, z0, z1, dur in segs:
        extend(dur * (scale if kind == 'stop' else 1.0), z0, z1)
    extend(surface_watch_min, 0.0, 0.0)

    t = np.asarray(times)
    z = np.maximum(np.asarray(depths), 0.0)
    # Degenerate travel legs (dz ~ 1e-16) can emit duplicate timestamps, which
    # solve_ivp rejects as unsorted t_eval. Keep strictly increasing samples.
    keep = np.concatenate([[True], np.diff(t) > 1e-12])
    return t[keep], z[keep]
