"""Verteilungen in geschlossener Form (hzw_model) an Handwerten, plus Weibull-Plot und Szenario-Erzeuger."""

import math

import numpy as np
import pytest

import hzw_core as K
import hzw_model as M
import hzw_scenario as S


def test_weibull_handwerte():
    # k = 2, λ = 3: S(3) = e^-1, Median = 3·√ln 2, Mittel = 3·Γ(1,5) = 3·√π/2
    assert M.weibull_survival(3.0, 2.0, 3.0) == pytest.approx(math.exp(-1.0), abs=1e-15)
    assert M.weibull_median(2.0, 3.0) == pytest.approx(3.0 * math.sqrt(math.log(2.0)), abs=1e-15)
    assert M.weibull_mean(2.0, 3.0) == pytest.approx(3.0 * math.sqrt(math.pi) / 2.0, abs=1e-12)
    # Hazard bei k = 2: h(t) = 2t/λ²; bei t = 3: 6/9
    assert M.weibull_hazard(3.0, 2.0, 3.0) == pytest.approx(2 / 3, abs=1e-15)
    # Quantil: S(t_q) = 1 − q
    for q in (0.1, 0.5, 0.9):
        assert M.weibull_survival(M.weibull_quantile(q, 1.7, 4.2), 1.7, 4.2) == pytest.approx(1 - q, abs=1e-14)


def test_k_gleich_eins_ist_exponential():
    rate = 4 / 27
    for t in (0.5, 2.0, 7.0):
        assert M.weibull_survival(t, 1.0, 1 / rate) == pytest.approx(math.exp(-rate * t), abs=1e-15)
    assert M.weibull_hazard(5.0, 1.0, 1 / rate) == pytest.approx(rate, abs=1e-15)
    assert M.weibull_quantile(0.5, 1.0, 1 / rate) == pytest.approx(M.exponential_quantile(0.5, rate), abs=1e-14)
    assert M.exponential_quantile(0.5, rate) == pytest.approx(math.log(2) * 27 / 4, abs=1e-14)


def test_skala_fuer_median_und_bedingte_wahrscheinlichkeit():
    for k in (0.7, 1.3, 2.5):
        lam = M.weibull_scale_for_median(k, 4.0)
        assert M.weibull_median(k, lam) == pytest.approx(4.0, abs=1e-13)
    # k = 1: Gedächtnislosigkeit; k = 2: steigende Rate, die Wahrscheinlichkeit, noch 2 Tage zu bleiben, sinkt mit dem Alter
    assert [M.conditional_stay(a, 2.0, 1.0, 5.0) for a in (0, 3, 9)] == pytest.approx([math.exp(-0.4)] * 3, abs=1e-14)
    p = [M.conditional_stay(a, 2.0, 2.0, 5.0) for a in (0, 3, 9)]
    assert p[0] > p[1] > p[2]
    q = [M.conditional_stay(a, 2.0, 0.6, 5.0) for a in (0, 3, 9)]
    assert q[0] < q[1] < q[2]  # fallende Rate: wer lange da ist, bleibt wahrscheinlicher


def test_ungueltiges_niveau_meldet_fehler():
    for q in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(K.SurvivalError):
            M.weibull_quantile(q, 1.0, 1.0)
        with pytest.raises(K.SurvivalError):
            M.exponential_quantile(q, 1.0)
        with pytest.raises(K.SurvivalError):
            M.normal_quantile(q)


def test_normal_und_lognormal_handwerte():
    assert M.normal_quantile(0.5) == pytest.approx(0.0, abs=1e-12)
    assert M.normal_quantile(0.975) == pytest.approx(1.959964, abs=1e-6)
    assert float(M.normal_sf(0.0)) == pytest.approx(0.5, abs=1e-15)
    # Median ist der Median: S(m) = 1/2; Quantil 0,9 der Welt der Messreihe: 4·exp(0,9·z₀,₉) mit z₀,₉ = 1,2815515655
    assert float(M.lognormal_survival(4.0, 4.0, 0.9)) == pytest.approx(0.5, abs=1e-15)
    assert M.lognormal_quantile(0.9, 4.0, 0.9) == pytest.approx(4.0 * math.exp(0.9 * 1.2815515655), abs=1e-6)
    assert M.lognormal_mean(4.0, 0.9) == pytest.approx(4.0 * math.exp(0.405), abs=1e-12)


def test_line_fit_handbeispiel():
    # Punkte (0, 1), (1, 3), (2, 5): exakt y = 1 + 2x, R² = 1; mit Rauschen: (0,0), (1,1), (2,0): Steigung 0, R² = 0
    a, b, r2 = M.line_fit([0, 1, 2], [1, 3, 5])
    assert (a, b, r2) == pytest.approx((1.0, 2.0, 1.0), abs=1e-14)
    a, b, r2 = M.line_fit([0, 1, 2], [0, 1, 0])
    assert b == pytest.approx(0.0, abs=1e-15) and r2 == pytest.approx(0.0, abs=1e-15)


def test_weibull_plot_ist_gerade_bei_weibull_ohne_zensierung():
    rng = K.SplitMix64(3)
    t = rng.weibull(3000, 1.7, 5.0)
    res = M.plot_r2(t, np.ones(3000, bool))
    r2, slope, intercept, x, y = res
    assert r2 > 0.99 and slope == pytest.approx(1.7, abs=0.1) and intercept == pytest.approx(-1.7 * math.log(5.0), abs=0.2)


def test_weibull_plot_ohne_genug_punkte_ist_none():
    assert M.plot_r2([1, 2, 3, 4], [1, 1, 1, 1]) is None  # nur drei Punkte mit 0 < Ŝ < 1 (der letzte hat Ŝ = 0)
    assert M.plot_r2([1, 2, 3], [0, 0, 0]) is None


def test_szenario_ist_deterministisch_und_folgt_der_definition():
    w = S.WORLDS["lognormal_09"]
    a, b = S.simulate(w, 200, 8, 5), S.simulate(w, 200, 8, 5)
    assert np.array_equal(a.time, b.time) and np.array_equal(a.event, b.event)
    assert not np.array_equal(a.time, S.simulate(w, 200, 8, 6).time)
    # Beobachtungszeit: Ereignis → Verweilzeit D ≤ W − A; Zensierung → Zeit = W − A < D; alle Zeiten in (0, W]
    assert a.time.min() > 0 and a.time.max() <= 8.0
    rng = K.SplitMix64(5)
    d = S.draw_durations(rng, w, 200)
    arrival = rng.uniform(200) * 8
    assert np.allclose(a.time, np.minimum(d, 8 - arrival)) and np.array_equal(a.event, d <= 8 - arrival)
    assert a.n == 200 and a.n_events == int(a.event.sum()) and a.censored_share == pytest.approx(1 - a.n_events / 200)


def test_zensierungsanteil_folgt_der_formel():
    """Anteil zensiert = (1/W) ∫₀^W S(u) du (Ankunft gleichverteilt): per Mittelpunktsregel gegen die Simulation über 20 000 Container."""
    for key, window in (("lognormal_09", 8), ("weibull_13", 30)):
        w = S.WORLDS[key]
        u = (np.arange(4000) + 0.5) * window / 4000
        expected = float(np.mean(w.survival(u)))
        got = S.simulate(w, 20000, window, 77).censored_share
        assert got == pytest.approx(expected, abs=0.01), key


def test_welten_haben_median_vier():
    for w in S.WORLDS.values():
        assert float(w.survival(4.0)) == pytest.approx(0.5, abs=1e-9)
        assert w.quantile(0.5) == pytest.approx(4.0, abs=1e-6)
        assert w.mean() > 0
