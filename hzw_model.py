"""Verteilungen der Demo in geschlossener Form: Weibull, Exponential, Lognormal (Wahrheit der Messreihe) und der Weibull-Plot.

Alles reine Funktionen auf numpy-Arrays, nur numpy und Standardbibliothek. Zeitachse: Tage (Verweilzeit eines Containers).

Weibull mit Form k und Skala λ:   S(t) = exp(−(t/λ)^k),   h(t) = (k/λ)·(t/λ)^(k−1),   Quantil t_q = λ·(−ln(1−q))^(1/k).
Exponential mit Rate ρ:           S(t) = exp(−ρ·t),        h(t) = ρ,                      Quantil t_q = −ln(1−q)/ρ   (Weibull mit k = 1 und λ = 1/ρ).
Lognormal mit Median m, σ:        S(t) = 1 − Φ((ln t − ln m)/σ),  h(t) = f(t)/S(t).
Weibull-Plot: log(−log Ŝ(t)) gegen log t an den Ereigniszeiten des Kaplan–Meier-Schätzers; unter einer Weibull-Verteilung liegen die Punkte auf einer
Geraden mit Steigung k und Achsenabschnitt −k·log λ. Das Gütemaß R² ist das Bestimmtheitsmaß dieser Geraden.
"""
from __future__ import annotations

import math

import numpy as np

import hzw_core as K

LN2 = math.log(2.0)


# ------------------------------------------------------------------------------------------------ Weibull und Exponential


def weibull_survival(t, k: float, lam: float):
    return np.exp(-((np.asarray(t, dtype=float) / lam) ** k))


def weibull_hazard(t, k: float, lam: float):
    """Abholrate zum Alter t. Für k < 1 gilt h(0) = ∞, die Funktion wird daher nur für t > 0 ausgewertet."""
    t = np.asarray(t, dtype=float)
    return (k / lam) * (t / lam) ** (k - 1.0)


def weibull_quantile(q: float, k: float, lam: float) -> float:
    if not 0.0 < q < 1.0:
        raise K.SurvivalError("Das Niveau q muss zwischen 0 und 1 liegen.")
    return lam * (-math.log1p(-q)) ** (1.0 / k)


def weibull_median(k: float, lam: float) -> float:
    return lam * LN2 ** (1.0 / k)


def weibull_mean(k: float, lam: float) -> float:
    return lam * math.gamma(1.0 + 1.0 / k)


def weibull_scale_for_median(k: float, median: float) -> float:
    """Skala λ, bei der die Weibull-Verteilung mit Form k den gegebenen Median hat."""
    return median / LN2 ** (1.0 / k)


def exponential_quantile(q: float, rate: float) -> float:
    if not 0.0 < q < 1.0:
        raise K.SurvivalError("Das Niveau q muss zwischen 0 und 1 liegen.")
    return -math.log1p(-q) / rate


def conditional_stay(t_age: float, extra: float, k: float, lam: float) -> float:
    """P(T > a + r | T > a) = S(a + r)/S(a): Wahrscheinlichkeit, noch r Tage zu bleiben, wenn der Container schon a Tage da ist."""
    return float(weibull_survival(t_age + extra, k, lam) / weibull_survival(t_age, k, lam))


# ------------------------------------------------------------------------------------------------ Lognormal (Wahrheit der Messreihe)

_erfc = np.vectorize(math.erfc, otypes=[float])


def normal_sf(z):
    """1 − Φ(z), über die komplementäre Fehlerfunktion (genau auch im Tail)."""
    return 0.5 * _erfc(np.asarray(z, dtype=float) / math.sqrt(2.0))


def normal_quantile(p: float) -> float:
    """Φ⁻¹(p) für 0 < p < 1 durch Bisektion über die Fehlerfunktion (ohne scipy)."""
    if not 0.0 < p < 1.0:
        raise K.SurvivalError("Die Wahrscheinlichkeit muss zwischen 0 und 1 liegen.")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def lognormal_survival(t, median: float, sigma: float):
    t = np.asarray(t, dtype=float)
    return normal_sf((np.log(t) - math.log(median)) / sigma)


def lognormal_hazard(t, median: float, sigma: float):
    """h(t) = f(t)/S(t) mit der Lognormal-Dichte f; steigt zuerst und fällt später (nicht monoton)."""
    t = np.asarray(t, dtype=float)
    z = (np.log(t) - math.log(median)) / sigma
    dens = np.exp(-0.5 * z * z) / (t * sigma * math.sqrt(2.0 * math.pi))
    return dens / lognormal_survival(t, median, sigma)


def lognormal_quantile(q: float, median: float, sigma: float) -> float:
    return median * math.exp(sigma * normal_quantile(q))


def lognormal_mean(median: float, sigma: float) -> float:
    return median * math.exp(sigma * sigma / 2.0)


# ------------------------------------------------------------------------------------------------ Weibull-Plot


def weibull_plot_points(km: dict):
    """Punkte (log t, log(−log Ŝ)) an den Ereigniszeiten mit 0 < Ŝ < 1 (dort ist der doppelte Logarithmus erklärt)."""
    t, s = km["t"], km["s"]
    keep = (s > 0.0) & (s < 1.0) & (t > 0.0)
    return np.log(t[keep]), np.log(-np.log(s[keep]))


def line_fit(x, y):
    """Gerade y = a + b·x nach kleinsten Quadraten (Achsenabschnitt a, Steigung b) und Bestimmtheitsmaß R² = 1 − Var(Rest)/Var(y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm, ym = x.mean(), y.mean()
    b = float(np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2))
    a = float(ym - b * xm)
    res = y - (a + b * x)
    return a, b, float(1.0 - np.var(res) / np.var(y))


MIN_PLOT_POINTS = 5


def plot_r2(time, event):
    """Gütemaß des Weibull-Plots einer Stichprobe: (R², Steigung, Achsenabschnitt, x, y) oder None, wenn weniger als MIN_PLOT_POINTS Punkte da sind."""
    km = K.kaplan_meier(time, event)
    x, y = weibull_plot_points(km)
    if x.size < MIN_PLOT_POINTS or np.var(x) == 0.0 or np.var(y) == 0.0:
        return None
    a, b, r2 = line_fit(x, y)
    return r2, b, a, x, y
