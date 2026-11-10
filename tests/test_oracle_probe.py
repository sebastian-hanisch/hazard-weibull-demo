"""Orakel des Weibull-Plots: Kaplan–Meier-Punkte per exakter Bruchrechnung und R² per scipy bzw. Korrelation, unabhängig vom Kern.

* Die Punkte (ln t, ln(−ln Ŝ)) entstehen aus `fractions` (Produkt-Grenzwert mit Brüchen, Risikomenge von Hand gezählt), nicht aus hzw_core.
* R² der Ausgleichsgeraden: Quadrat des Korrelationskoeffizienten (bei einfacher linearer Regression gleich dem Bestimmtheitsmaß) und `scipy.stats.linregress`.
"""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import hzw_core as K
import hzw_model as M
import hzw_scenario as S


def km_fractions(t, e):
    """Produkt-Grenzwert in Brüchen: je Ereigniszeit x ist n = #(t ≥ x), d = #(Ereignisse bei x); Ŝ = Π (n − d)/n."""
    s = F(1)
    out = []
    for x in sorted({ti for ti, ei in zip(t, e) if ei}):
        n = sum(1 for ti in t if ti >= x)
        d = sum(1 for ti, ei in zip(t, e) if ti == x and ei)
        s *= F(n - d, n)
        out.append((x, s))
    return out


def plot_points_oracle(t, e):
    return [(math.log(x), math.log(-math.log(float(s)))) for x, s in km_fractions(t, e) if 0 < s < 1]


def test_handbeispiel_punkte_des_weibull_plots():
    """Sechs Container: Ŝ = 5/6, 5/8, 5/12 bei t = 2, 4, 5 (der letzte Punkt t = 8 hat Ŝ = 0 und fällt heraus)."""
    t, e = [2, 3, 4, 5, 5, 8], [1, 0, 1, 1, 0, 1]
    pts = plot_points_oracle(t, e)
    assert len(pts) == 3
    assert pts[0] == pytest.approx((math.log(2), math.log(-math.log(5 / 6))), abs=1e-15)
    x, y = M.weibull_plot_points(K.kaplan_meier(t, e))
    assert x.tolist() == pytest.approx([p[0] for p in pts], abs=1e-14) and y.tolist() == pytest.approx([p[1] for p in pts], abs=1e-14)


def test_punkte_und_r2_gegen_bruchrechnung_und_scipy():
    stats = pytest.importorskip("scipy.stats")
    checked = 0
    for seed in range(40):
        world = list(S.WORLDS.values())[seed % 4]
        sample = S.simulate(world, 12 + seed, (8, 15, 30)[seed % 3], seed)
        t, e = sample.time.tolist(), sample.event.tolist()
        pts = plot_points_oracle(t, e)
        res = M.plot_r2(t, e)
        if len(pts) < M.MIN_PLOT_POINTS:
            assert res is None
            continue
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        r2, slope, intercept, xs, ys = res
        assert xs == pytest.approx(x, abs=1e-12) and ys == pytest.approx(y, abs=1e-12), seed
        lin = stats.linregress(x, y)
        assert r2 == pytest.approx(lin.rvalue**2, abs=1e-12), seed
        assert r2 == pytest.approx(float(np.corrcoef(x, y)[0, 1] ** 2), abs=1e-12), seed
        assert slope == pytest.approx(lin.slope, abs=1e-10) and intercept == pytest.approx(lin.intercept, abs=1e-10), seed
        checked += 1
    assert checked >= 35


def test_randfaelle_ohne_punkte():
    assert M.plot_r2([1, 2, 3], [0, 0, 0]) is None            # nichts beobachtet
    assert M.plot_r2([2, 2, 2, 2, 2, 2], [1] * 6) is None     # ein Zeitpunkt: Ŝ springt auf 0, keine Punkte
    with pytest.raises(K.SurvivalError):
        M.plot_r2([0, 1], [1, 1])
