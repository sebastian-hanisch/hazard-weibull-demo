"""Eigenschaftstests (metamorph, ORACLE-PLAYBOOK §5.2): Aussagen, die für Weibull- und Exponential-Anpassung gelten müssen, ohne dass man das richtige Ergebnis kennt.

Jede Eigenschaft über 20 bis 30 Instanzen mit festem Seed. Eine Eigenschaft gilt nur unter den hier genannten Voraussetzungen (Rechtszensierung, Zeiten > 0).
"""

import math

import numpy as np
import pytest

import hzw_core as K
import hzw_model as M
import hzw_scenario as S


def _sample(seed, n=60):
    rng = K.SplitMix64(seed)
    t = rng.weibull(n, (0.8, 1.2, 1.8)[seed % 3], 5.0)
    c = rng.uniform(n) * 9 + 1e-9
    return np.minimum(t, c), t <= c


def test_skalierung_der_zeit_aendert_lambda_mit_c_und_k_nicht():
    for seed in range(20):
        t, e = _sample(seed)
        c = (0.25, 3.0, 17.0)[seed % 3]
        a, b = K.weibull_fit(t, e), K.weibull_fit(c * t, e)
        assert a["ok"] and b["ok"]
        assert b["lam"] == pytest.approx(c * a["lam"], rel=1e-6), seed
        assert b["k"] == pytest.approx(a["k"], rel=1e-6), seed
        x, y = K.exponential_fit(t, e), K.exponential_fit(c * t, e)
        assert y["rate"] == pytest.approx(x["rate"] / c, rel=1e-12), seed


def test_reihenfolge_und_verdoppeln_aendern_die_schaetzung_nicht():
    for seed in range(20):
        t, e = _sample(seed)
        perm = np.random.default_rng(seed).permutation(t.size)
        a = K.weibull_fit(t, e)
        b = K.weibull_fit(t[perm], e[perm])
        assert (b["k"], b["lam"]) == pytest.approx((a["k"], a["lam"]), rel=1e-6), seed  # Summationsreihenfolge: Abweichung im Rahmen der BFGS-Toleranz
        # jede Beobachtung zweimal: die Log-Likelihood verdoppelt sich, ihr Maximum bleibt an derselben Stelle
        c = K.weibull_fit(np.r_[t, t], np.r_[e, e])
        assert (c["k"], c["lam"]) == pytest.approx((a["k"], a["lam"]), rel=1e-6), seed
        assert K.exponential_fit(np.r_[t, t], np.r_[e, e])["rate"] == pytest.approx(K.exponential_fit(t, e)["rate"], rel=1e-12)


def test_exponential_ist_gedaechtnislos_und_weibull_mit_k_eins():
    for rate in (0.05, 0.15, 0.8):
        for s, t in ((0.5, 1.0), (2.0, 7.0), (10.0, 0.3)):
            sv = lambda x: float(M.weibull_survival(x, 1.0, 1 / rate))  # noqa: E731
            assert sv(s + t) == pytest.approx(sv(s) * sv(t), rel=1e-12)
            assert M.conditional_stay(s, t, 1.0, 1 / rate) == pytest.approx(sv(t), rel=1e-12)
        assert float(M.weibull_hazard(3.0, 1.0, 1 / rate)) == pytest.approx(rate, rel=1e-12)


def test_quantile_sind_monoton_im_niveau_und_im_scale():
    for k in (0.6, 1.0, 2.0):
        qs = [M.weibull_quantile(q, k, 5.0) for q in (0.1, 0.5, 0.9, 0.95)]
        assert qs == sorted(qs) and len(set(qs)) == 4
        assert M.weibull_quantile(0.9, k, 10.0) == pytest.approx(2 * M.weibull_quantile(0.9, k, 5.0), rel=1e-14)


def test_mehr_daten_naehern_die_schaetzung_der_wahrheit():
    """Mittel über 20 feste Seeds: der mittlere Fehler von k̂ ist bei n = 2000 kleiner als bei n = 100 (Weibull-Wahrheit, Form 1,3, Fenster 30)."""
    w = S.WORLDS["weibull_13"]
    err = {}
    for n in (100, 2000):
        errs = []
        for seed in range(20):
            s = S.simulate(w, n, 30, 900 + seed)
            errs.append(abs(K.weibull_fit(s.time, s.event)["k"] - 1.3))
        err[n] = float(np.mean(errs))
    assert err[2000] < err[100]
    assert err[2000] < 0.05


def test_gleiche_einstellungen_gleiche_zahlen():
    a = S.simulate(S.WORLDS["lognormal_09"], 300, 8, 35)
    b = S.simulate(S.WORLDS["lognormal_09"], 300, 8, 35)
    fa, fb = K.weibull_fit(a.time, a.event), K.weibull_fit(b.time, b.event)
    assert fa == fb


def test_weibull_plot_ist_unabhaengig_von_der_zeiteinheit_im_r2():
    """R² der Geraden in log-log ändert sich nicht, wenn man die Zeit mit c skaliert (nur der Achsenabschnitt verschiebt sich)."""
    for seed in range(20):
        t, e = _sample(seed, n=120)
        a, b = M.plot_r2(t, e), M.plot_r2(7.0 * t, e)
        assert a is not None and b is not None
        assert b[0] == pytest.approx(a[0], abs=1e-12) and b[1] == pytest.approx(a[1], abs=1e-12)
        assert b[2] == pytest.approx(a[2] - a[1] * math.log(7.0), abs=1e-10)
