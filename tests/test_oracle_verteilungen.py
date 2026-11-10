"""Orakel der Verteilungen und der Quantile (hzw_model, hzw_scenario): scipy.stats als anderer Rechenweg, dazu numerische Ableitung und Quadratur.

scipy steht nur in requirements-dev.txt (die App rechnet ohne); fehlt es, überspringen sich die Tests (`importorskip`).
"""

import math

import numpy as np
import pytest

import hzw_model as M
import hzw_scenario as S

stats = pytest.importorskip("scipy.stats")
integrate = pytest.importorskip("scipy.integrate")


def test_weibull_gegen_scipy():
    t = np.array([0.3, 1.0, 2.5, 4.0, 9.0, 20.0])
    for k, lam in ((0.6, 3.0), (1.0, 6.0), (1.5, 6.0), (3.0, 2.5)):
        ref = stats.weibull_min(k, scale=lam)
        assert M.weibull_survival(t, k, lam) == pytest.approx(ref.sf(t), rel=1e-12)
        assert M.weibull_hazard(t, k, lam) == pytest.approx(ref.pdf(t) / ref.sf(t), rel=1e-12)
        for q in (0.1, 0.5, 0.9, 0.95):
            assert M.weibull_quantile(q, k, lam) == pytest.approx(ref.ppf(q), rel=1e-12)
        assert M.weibull_mean(k, lam) == pytest.approx(ref.mean(), rel=1e-12)
        assert M.weibull_median(k, lam) == pytest.approx(ref.median(), rel=1e-12)


def test_lognormal_gegen_scipy():
    t = np.array([0.3, 1.0, 2.5, 4.0, 9.0, 20.0])
    for median, sigma in ((4.0, 0.9), (4.0, 0.5)):
        ref = stats.lognorm(sigma, scale=median)
        assert M.lognormal_survival(t, median, sigma) == pytest.approx(ref.sf(t), rel=1e-10)
        assert M.lognormal_hazard(t, median, sigma) == pytest.approx(ref.pdf(t) / ref.sf(t), rel=1e-10)
        for q in (0.5, 0.75, 0.9, 0.95):
            assert M.lognormal_quantile(q, median, sigma) == pytest.approx(ref.ppf(q), rel=1e-10)
        assert M.lognormal_mean(median, sigma) == pytest.approx(ref.mean(), rel=1e-12)


def test_normal_quantil_gegen_scipy():
    for p in (1e-6, 0.01, 0.2, 0.5, 0.9, 0.975, 0.999999):
        assert M.normal_quantile(p) == pytest.approx(float(stats.norm.ppf(p)), abs=1e-9)
    z = np.array([-5.0, -1.0, 0.0, 2.0, 7.0])
    assert M.normal_sf(z) == pytest.approx(stats.norm.sf(z), rel=1e-12)


def test_hazard_ist_negative_ableitung_des_log_ueberlebens():
    """h(t) = −d/dt ln S(t) per zentraler Differenz, für Weibull und Lognormal."""
    h = 1e-5
    for t in (0.5, 2.0, 6.0, 15.0):
        for k, lam in ((0.7, 4.0), (1.5, 6.0)):
            d = -(math.log(float(M.weibull_survival(t + h, k, lam))) - math.log(float(M.weibull_survival(t - h, k, lam)))) / (2 * h)
            assert float(M.weibull_hazard(t, k, lam)) == pytest.approx(d, rel=1e-6)
        d = -(math.log(float(M.lognormal_survival(t + h, 4.0, 0.9))) - math.log(float(M.lognormal_survival(t - h, 4.0, 0.9)))) / (2 * h)
        assert float(M.lognormal_hazard(t, 4.0, 0.9)) == pytest.approx(d, rel=1e-6)


def test_weibull_hazard_ist_monoton_lognormal_nicht():
    t = np.linspace(0.05, 40, 800)
    assert np.all(np.diff(M.weibull_hazard(t, 0.6, 5.0)) < 0)
    assert np.allclose(M.weibull_hazard(t, 1.0, 5.0), 0.2)
    assert np.all(np.diff(M.weibull_hazard(t, 2.0, 5.0)) > 0)
    h = M.lognormal_hazard(t, 4.0, 0.9)
    peak = int(np.argmax(h))
    assert 0 < peak < t.size - 1 and np.all(np.diff(h[: peak + 1]) > 0) and np.all(np.diff(h[peak:]) < 0)


def test_mittel_gegen_quadratur():
    """E[T] = ∫ S(t) dt für alle vier Welten (Quadratur bis weit in den Tail)."""
    for w in S.WORLDS.values():
        val, _ = integrate.quad(lambda x, w=w: float(w.survival(x)), 0, 4000, limit=400)
        assert w.mean() == pytest.approx(val, rel=1e-4), w.key
