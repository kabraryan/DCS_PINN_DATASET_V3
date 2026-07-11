from __future__ import annotations

import numpy as np
import pytest

from benchmark.buhlmann import (
    zhl16c_table, half_time_k, haldane_step, amb_bar, m_value, ceiling_fsw,
    P_SURFACE, F_N2_AIR, FSW_TO_BAR,
)

CANONICAL_B = [0.5050, 0.6514, 0.7222, 0.7825, 0.8126, 0.8434, 0.8693, 0.8910,
               0.9092, 0.9222, 0.9319, 0.9403, 0.9477, 0.9544, 0.9602, 0.9653]


def test_table_shape():
    assert zhl16c_table().shape == (16, 3)


def test_b_column_matches_canonical_buhlmann():
    """Compartment 16 b was 0.8693 (compartment 7's value) until 2026-07-09."""
    np.testing.assert_allclose(zhl16c_table()[:, 2], CANONICAL_B)


def test_b_strictly_increasing_and_a_strictly_decreasing():
    t = zhl16c_table()
    assert np.all(np.diff(t[:, 2]) > 0), "b must rise monotonically toward 1"
    assert np.all(np.diff(t[:, 1]) < 0), "a must fall monotonically"
    assert t[15, 2] < 1.0


def test_haldane_halves_the_gap_after_one_half_time():
    """Analytic property: after t_half, the gas gap to the alveolar pressure halves."""
    table = zhl16c_table()
    k = half_time_k(table)
    P_t = np.full(16, 0.79)
    P_alv = np.full(16, 2.79)          # gap of 2.0 bar
    out = haldane_step(P_t, P_alv, k, table[:, 0])   # step each by its own half-time
    np.testing.assert_allclose(P_alv - out, 1.0, rtol=1e-12)


def test_amb_bar_surface_and_33fsw():
    assert amb_bar(0.0) == pytest.approx(P_SURFACE)
    assert amb_bar(33.0) == pytest.approx(P_SURFACE + 33 * 0.030643)


def test_ceiling_zero_when_tissues_at_surface_equilibrium():
    table = zhl16c_table()
    a, b = table[:, 1], table[:, 2]
    P_t = np.full(16, P_SURFACE * F_N2_AIR)
    assert ceiling_fsw(P_t, a, b) == 0.0


def test_ceiling_rises_with_tissue_loading():
    table = zhl16c_table()
    a, b = table[:, 1], table[:, 2]
    low = ceiling_fsw(np.full(16, 1.5), a, b)
    high = ceiling_fsw(np.full(16, 3.0), a, b)
    assert high > low >= 0.0


def test_gradient_factor_one_reduces_to_plain_ceiling():
    table = zhl16c_table()
    a, b = table[:, 1], table[:, 2]
    P_t = np.full(16, 2.5)
    assert ceiling_fsw(P_t, a, b, gf=1.0) == pytest.approx(ceiling_fsw(P_t, a, b))


def test_gradient_factor_one_equals_the_plain_buhlmann_formula():
    """gf=1.0 must reduce to the classic ceiling (P_t - a) * b, computed independently.

    The neighbouring test compares ceiling_fsw against its own default argument, so it
    passes whatever the formula does. This one recomputes the target from first
    principles. Task 4's gradient-factor interpolation relies on this degeneracy.
    """
    table = zhl16c_table()
    a, b = table[:, 1], table[:, 2]
    P_t = np.full(16, 2.5)
    tol_bar = np.max((P_t - a) * b)                      # classic Bühlmann, no gf term
    expected = max(0.0, float((tol_bar - P_SURFACE) / FSW_TO_BAR))
    assert ceiling_fsw(P_t, a, b, gf=1.0) == pytest.approx(expected)


def test_smaller_gradient_factor_is_strictly_more_conservative():
    """Monotone in gf: every decrease in gf must deepen the ceiling."""
    table = zhl16c_table()
    a, b = table[:, 1], table[:, 2]
    P_t = np.full(16, 2.5)
    ceilings = [ceiling_fsw(P_t, a, b, gf=g) for g in (0.1, 0.3, 0.5, 0.7, 1.0)]
    assert all(x > y for x, y in zip(ceilings, ceilings[1:]))


def test_smaller_gradient_factor_gives_deeper_ceiling():
    """gf < 1 is more conservative: you must stop deeper."""
    table = zhl16c_table()
    a, b = table[:, 1], table[:, 2]
    P_t = np.full(16, 2.5)
    assert ceiling_fsw(P_t, a, b, gf=0.3) > ceiling_fsw(P_t, a, b, gf=1.0)


def test_m_value_exceeds_ambient():
    table = zhl16c_table()
    mv = m_value(2.0, table[:, 1], table[:, 2])
    assert np.all(mv > 2.0)
