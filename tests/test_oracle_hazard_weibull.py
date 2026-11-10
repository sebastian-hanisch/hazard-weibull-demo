"""Orakel der Weibull- und Exponential-Anpassung (anderer Rechenweg als der Kern).

* Weibull-ML gegen eine Rastersuche über (k, λ): die Log-Likelihood wird hier in anderer Schreibweise (Summe über Beobachtungen mit Hazard und kumuliertem
  Hazard) auf einem verfeinerten Raster maximiert, ohne Gradient und ohne BFGS.
* Weibull-ML ohne Zensierung gegen die Standardformeln (Stationaritätsbedingungen der Weibull-Likelihood).
* Exponential gegen die geschlossene Formel d / Σt und gegen die Rastersuche über die Rate.
* Gradient am Ergebnis per finiter Differenz (Zielwert und Ableitung am eigenen Ergebnis, nicht nur die Parameter, siehe ORACLE-PLAYBOOK §2).
Der große Lauf (mehr als 200 Instanzen) lief im Scratch-Ordner; hier die kleine, schnelle Fassung.
"""

import math

import numpy as np
import pytest

import hzw_core as K


def loglik(t, e, k, lam):
    """Log-Likelihood mit Rechtszensierung: Σ δ_i ln h(t_i) − Σ H(t_i), h = (k/λ)(t/λ)^(k−1), H = (t/λ)^k. Reine Python-Schleife, math statt numpy."""
    total = 0.0
    for ti, ei in zip(t, e):
        if ei:
            total += math.log(k / lam) + (k - 1.0) * math.log(ti / lam)
        total -= (ti / lam) ** k
    return total


def grid_search(t, e, rounds=7):
    """Maximum der Log-Likelihood über ein Raster, das sich um das beste Paar jeweils um den Faktor 4 verengt (Brute Force, ohne Ableitung)."""
    k_lo, k_hi, l_lo, l_hi = math.log(0.2), math.log(8.0), math.log(0.2), math.log(200.0)
    best = (-math.inf, None, None)
    for _ in range(rounds):
        ks = np.exp(np.linspace(k_lo, k_hi, 15))
        ls = np.exp(np.linspace(l_lo, l_hi, 15))
        for k in ks:
            for lam in ls:
                v = loglik(t, e, float(k), float(lam))
                if v > best[0]:
                    best = (v, float(k), float(lam))
        dk, dl = (k_hi - k_lo) / 14, (l_hi - l_lo) / 14
        k_lo, k_hi = math.log(best[1]) - dk, math.log(best[1]) + dk
        l_lo, l_hi = math.log(best[2]) - dl, math.log(best[2]) + dl
    return best


def _instance(seed):
    rng = K.SplitMix64(seed)
    n = 8 + seed % 25
    k_true = (0.6, 1.0, 1.5, 2.4)[seed % 4]
    t = rng.weibull(n, k_true, 5.0)
    c = rng.uniform(n) * (4.0 + seed % 11) + 1e-9
    return np.minimum(t, c), t <= c


def test_handbeispiel_der_likelihood():
    """k = 1 und λ = 27/4 auf dem Handbeispiel: ℓ = d·ln ρ − ρ·Σt mit ρ = 4/27, d = 4, Σt = 27, also 4·ln(4/27) − 4."""
    t, e = [2, 3, 4, 5, 5, 8], [1, 0, 1, 1, 0, 1]
    assert loglik(t, e, 1.0, 27 / 4) == pytest.approx(4 * math.log(4 / 27) - 4, abs=1e-12)


def test_weibull_ml_stimmt_mit_rastersuche_ueberein():
    checked = 0
    for seed in range(30):
        t, e = _instance(seed)
        if np.unique(t[e]).size < 3:
            continue
        r = K.weibull_fit(t, e)
        assert r["ok"], seed
        best, kb, lb = grid_search(t.tolist(), e.tolist())
        ll_fit = loglik(t.tolist(), e.tolist(), r["k"], r["lam"])
        assert ll_fit >= best - 1e-9, (seed, ll_fit, best)            # der Kern ist mindestens so gut wie das Raster
        assert ll_fit - best < 5e-3, (seed, ll_fit, best)             # und nicht in einem fremden Gipfel: das Raster findet fast denselben Wert
        assert r["k"] == pytest.approx(kb, rel=0.02) and r["lam"] == pytest.approx(lb, rel=0.02), seed
        checked += 1
    assert checked >= 25


def test_gradient_am_ergebnis_ist_null():
    for seed in range(20):
        t, e = _instance(seed)
        if np.unique(t[e]).size < 3:
            continue
        r = K.weibull_fit(t, e)
        lk, ll = math.log(r["k"]), math.log(r["lam"])
        h = 1e-5

        def f(a, b, t=t, e=e):
            return loglik(t.tolist(), e.tolist(), math.exp(a), math.exp(b))

        assert abs((f(lk + h, ll) - f(lk - h, ll)) / (2 * h)) < 1e-4, seed
        assert abs((f(lk, ll + h) - f(lk, ll - h)) / (2 * h)) < 1e-4, seed


def test_weibull_ml_ohne_zensierung_erfuellt_die_standardformeln():
    """Ohne Zensierung gilt für das Maximum: λ^k = Σt^k / n  und  1/k = Σ t^k ln t / Σ t^k − Σ ln t / n."""
    for seed in range(15):
        rng = K.SplitMix64(300 + seed)
        n = 20 + 5 * seed
        t = rng.weibull(n, (0.7, 1.3, 2.0)[seed % 3], 4.0)
        r = K.weibull_fit(t, np.ones(n, bool))
        k, lam = r["k"], r["lam"]
        tk = t**k
        assert lam**k == pytest.approx(tk.sum() / n, rel=1e-6), seed
        assert 1.0 / k == pytest.approx((tk * np.log(t)).sum() / tk.sum() - np.log(t).mean(), rel=1e-6), seed


def test_exponential_gegen_geschlossene_formel_und_rastersuche():
    for seed in range(30):
        t, e = _instance(seed)
        if not e.any():
            continue
        r = K.exponential_fit(t, e)
        assert r["rate"] == pytest.approx(float(e.sum()) / float(t.sum()), rel=1e-15)
        # Rastersuche über die Rate: ℓ(ρ) = d ln ρ − ρ Σt, Maximum bei d/Σt
        rhos = np.exp(np.linspace(math.log(r["rate"]) - 0.3, math.log(r["rate"]) + 0.3, 601))
        ll = e.sum() * np.log(rhos) - rhos * t.sum()
        assert rhos[int(np.argmax(ll))] == pytest.approx(r["rate"], rel=2e-3)


def test_randfaelle_ohne_schaetzer_und_entartung():
    assert K.weibull_fit([3, 4, 5], [0, 0, 0])["ok"] is False                 # kein Ereignis
    assert K.weibull_fit([3, 4, 5], [1, 0, 0])["ok"] is False                 # ein Ereignis
    assert K.weibull_fit([4, 4, 4, 4], [1, 1, 1, 1])["ok"] is False           # alle Ereigniszeiten gleich
    assert K.exponential_fit([3, 4, 5], [0, 0, 0])["ok"] is False
    one = K.exponential_fit([3, 4, 5], [1, 0, 0])                              # ein Ereignis genügt für die Exponentialverteilung: 1/12
    assert one["ok"] and one["rate"] == pytest.approx(1 / 12, abs=1e-15)
    for bad in ([0, 1, 2], [-1, 2, 3]):                                        # Zeiten ≤ 0 sind ungültig
        with pytest.raises(K.SurvivalError):
            K.weibull_fit(bad, [1, 1, 1])
