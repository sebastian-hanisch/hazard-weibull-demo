"""Kern (hzw_core): Handbeispiel mit Brüchen, Eigenschaften, Randfälle. Tests aus ueberlebenszeit-planung/kern/test_uz_core.py übernommen, soweit die Demo die Funktion nutzt."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import hzw_core as K


def test_km_handbeispiel_sechs_container(six):
    km = K.kaplan_meier(*six)
    assert km["t"].tolist() == [2, 4, 5, 8]
    assert km["n"].tolist() == [6, 4, 3, 1]
    assert km["d"].tolist() == [1, 1, 1, 1]
    for got, want in zip(km["s"], [F(5, 6), F(5, 8), F(5, 12), F(0)]):
        assert got == pytest.approx(float(want), abs=1e-15)
    assert K.quantile(km, 0.5) == 5.0
    # Greenwood-Varianz bei t = 5: S² Σ d/(n(n−d)) = 85/1728
    assert km["s"][2] ** 2 * km["gw"][2] == pytest.approx(85 / 1728, abs=1e-12)


def test_km_mit_gleichstaenden_zaehlt_ereignis_vor_zensierung():
    # bei t = 5 ein Ereignis und eine Zensierung: beide sind bei t = 5 im Risiko (n = 3), Ereignisse zählen mit d = 1
    km = K.kaplan_meier([2, 3, 4, 5, 5, 8], [1, 0, 1, 1, 0, 1])
    assert km["n"][2] == 3 and km["d"][2] == 1
    km2 = K.kaplan_meier([1, 1, 2], [1, 1, 1])
    assert km2["d"].tolist() == [2, 1] and km2["n"].tolist() == [3, 1]
    assert km2["s"].tolist() == pytest.approx([1 / 3, 0.0])


def test_km_ohne_zensierung_ist_eins_minus_verteilungsfunktion():
    rng = K.SplitMix64(1)
    t = np.round(rng.exponential(50) * 3 + 0.5, 1)  # mit Gleichständen
    km = K.kaplan_meier(t, np.ones(50, bool))
    for x in (0.5, 1.0, 2.0, 4.0, 8.0):
        assert K.survival_at(km, x) == pytest.approx(float(np.mean(t > x)), abs=1e-12)


def test_km_alle_zensiert_und_quantil_nicht_schaetzbar():
    km = K.kaplan_meier([1, 2, 3], [0, 0, 0])
    assert km["t"].size == 0 and math.isnan(K.quantile(km, 0.5)) and K.survival_at(km, 5.0) == 1.0
    km = K.kaplan_meier([1, 2, 5, 6, 7, 8], [1, 0, 0, 0, 0, 0])
    assert math.isnan(K.quantile(km, 0.5))


def test_greenwood_band_liegt_in_null_eins_und_um_die_kurve(six):
    km = K.kaplan_meier(*six)
    for method in ("loglog", "plain"):
        s, lo, hi = K.greenwood_band(km, 4.5, 0.95, method)
        assert s == pytest.approx(5 / 8) and 0 <= lo < s < hi <= 1
    s, lo, hi = K.greenwood_band(km, 9.0)
    assert s == 0.0 and math.isnan(lo)
    assert math.isnan(K.greenwood_band(km, 1.0)[1])  # vor dem ersten Ereignis: Ŝ = 1, Band nan
    assert K._z_two_sided(0.95) == pytest.approx(1.959964, abs=1e-6)
    with pytest.raises(K.SurvivalError):
        K.greenwood_band(km, 4.5, 0.95, "unbekannt")


def test_eingabepruefung():
    for bad_t, bad_e in (([0, 1], [1, 1]), ([-1, 1], [1, 1]), ([1, 2], [1, 2]), ([1, 2], [1]), ([], []), ([1, float("nan")], [1, 1]), ([float("inf"), 1], [1, 1])):
        with pytest.raises(K.SurvivalError):
            K.kaplan_meier(bad_t, bad_e)
        with pytest.raises(K.SurvivalError):
            K.weibull_fit(bad_t, bad_e)
        with pytest.raises(K.SurvivalError):
            K.exponential_fit(bad_t, bad_e)


def test_exponential_handbeispiel(six):
    r = K.exponential_fit(*six)
    assert r["rate"] == pytest.approx(float(F(4, 27)), abs=1e-15)
    assert r["mean"] == pytest.approx(6.75) and r["median"] == pytest.approx(math.log(2) * 27 / 4)
    assert K.exponential_fit([1, 2], [0, 0])["ok"] is False


def test_weibull_fit_findet_wahrheit():
    rng = K.SplitMix64(11)
    t = rng.weibull(4000, 1.5, 6.0)
    c = rng.uniform(4000) * 14
    obs, ev = np.minimum(t, c), t <= c
    r = K.weibull_fit(obs, ev)
    assert r["ok"] and r["k"] == pytest.approx(1.5, abs=0.07) and r["lam"] == pytest.approx(6.0, abs=0.3)
    e = K.weibull_fit(rng.exponential(3000) * 5, np.ones(3000, bool))
    assert e["k"] == pytest.approx(1.0, abs=0.05)


def test_weibull_ohne_schaetzer_meldet_status():
    for t, e in (([1, 2, 3], [0, 0, 0]), ([2, 2, 2], [1, 1, 1]), ([2, 3], [1, 0])):  # kein Ereignis, nur eine Ereigniszeit, ein Ereignis
        r = K.weibull_fit(t, e)
        assert r["ok"] is False and math.isnan(r["k"]) and r["message"]


def test_splitmix_referenzwert_und_determinismus():
    # erster Ausgabewert von SplitMix64 bei Seed 0 (Referenz: Vigna, splitmix64.c)
    assert int(K.SplitMix64(0).u64(1)[0]) == 0xE220A8397B1DCDAF
    a, b = K.SplitMix64(5), K.SplitMix64(5)
    assert np.array_equal(a.uniform(10), b.uniform(10))
    u = K.SplitMix64(9).uniform(50000)
    assert 0.0 <= u.min() and u.max() < 1.0 and abs(u.mean() - 0.5) < 0.01
    x = K.SplitMix64(9).exponential(50000)
    assert abs(x.mean() - 1.0) < 0.03
    z = K.SplitMix64(9).normal(50000)
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1.0) < 0.02
