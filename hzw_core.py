"""Kern der Demo: Zufall, Eingabeprüfung, Kaplan–Meier, Weibull- und Exponential-Anpassung mit Rechtszensierung.

Kopie aus ueberlebenszeit-planung/kern/uz_core.py, Stand 2026-11-10. Übernommen sind nur die Funktionen, die diese Demo braucht
(`SplitMix64`, `SurvivalError`, `check_data`, `kaplan_meier`, `survival_at`, `quantile`, `greenwood_band`, `weibull_fit`, `exponential_fit`);
`check_data` und `kaplan_meier` ohne den Eintritts-Parameter (verzögerter Eintritt kommt in dieser Demo nicht vor), `kaplan_meier` zählt die Ereignisse
je Zeit mit `np.unique(..., return_counts=True)` statt mit einer Schleife. Der Rest der Quelle und seine Tests fehlen hier absichtlich.

Nur numpy und die Standardbibliothek. Konventionen: Risikomenge bei t = {i : Zeit_i ≥ t}; bei gleicher Zeit zählt das Ereignis vor der Zensierung
(Zensierte bleiben im Risiko); Zensierung gilt als unabhängig vom Ereignis.
Weibull-Anpassung: BFGS mit analytischem Gradienten und mehreren Startwerten samt Konvergenzprüfung (kein Nelder-Mead: der scheiterte in der Vorprüfung
in 1 von 295 Instanzen). Ohne Schätzer meldet sich die Funktion mit `ok = False` und einer Meldung, nie mit einer stillen Zahl.
"""
from __future__ import annotations

import math

import numpy as np

# ------------------------------------------------------------------------------------------------ Zufall

_MASK = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB


class SplitMix64:
    """SplitMix64-Strom (Steele, Lea, Flood 2014). Ausgabe k hängt nur von (Seed, k) ab: plattform- und versionsunabhängig.
    Alle Verteilungen sind hier selbst aus Gleichverteilten gebaut (nie numpy-Verteilungen, deren Algorithmen sich zwischen Versionen ändern)."""

    def __init__(self, seed: int):
        self.state = int(seed) & _MASK

    def u64(self, size: int) -> np.ndarray:
        idx = np.arange(1, size + 1, dtype=np.uint64)
        with np.errstate(over="ignore"):
            z = np.uint64(self.state) + idx * np.uint64(_GAMMA)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
            z = z ^ (z >> np.uint64(31))
        self.state = (self.state + size * _GAMMA) & _MASK
        return z

    def uniform(self, size: int) -> np.ndarray:
        """Gleichverteilt in [0, 1) mit 53 Bit."""
        return (self.u64(size) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))

    def normal(self, size: int) -> np.ndarray:
        """Standardnormal (Box-Muller, nur der Kosinus-Zweig); verbraucht 2·size Gleichverteilte."""
        u1 = self.uniform(size)
        u2 = self.uniform(size)
        return np.sqrt(-2.0 * np.log1p(-u1)) * np.cos(2.0 * np.pi * u2)

    def exponential(self, size: int) -> np.ndarray:
        """Exponentialverteilt mit Mittel 1 (Inversion)."""
        return -np.log1p(-self.uniform(size))

    def lognormal(self, size: int, median: float, sigma: float) -> np.ndarray:
        """Lognormal mit gegebenem Median und σ der Log-Skala."""
        return median * np.exp(sigma * self.normal(size))

    def weibull(self, size: int, shape: float, scale: float) -> np.ndarray:
        """Weibull mit Form k und Skala λ, S(t) = exp(−(t/λ)^k) (Inversion)."""
        return scale * self.exponential(size) ** (1.0 / shape)


# ------------------------------------------------------------------------------------------------ Eingabeprüfung


class SurvivalError(ValueError):
    """Ungültige Eingabe (verständliche Meldung für die App)."""


def check_data(time, event):
    """Prüft und wandelt in float-/bool-Arrays. Zeiten > 0 und endlich, Ereignis 0/1, gleich viele Einträge."""
    t = np.asarray(time, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise SurvivalError("Es braucht mindestens eine Beobachtung (eindimensionale Liste der Zeiten).")
    e_raw = np.asarray(event)
    if e_raw.shape != t.shape:
        raise SurvivalError("Zeiten und Ereignisse müssen gleich viele Einträge haben.")
    if not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise SurvivalError("Alle Zeiten müssen endlich und größer als 0 sein.")
    if not np.all(np.isin(e_raw, (0, 1, False, True))):
        raise SurvivalError("Das Ereignis muss 0 (zensiert) oder 1 (Ereignis) sein.")
    return t, e_raw.astype(bool)


# ------------------------------------------------------------------------------------------------ Kaplan–Meier


def kaplan_meier(time, event) -> dict:
    """Kaplan–Meier-Schätzer. Gibt Arrays je Ereigniszeit zurück:
    t, n (im Risiko), d (Ereignisse), s (Überleben nach t), gw (Greenwood-Summe Σ d/(n(n−d)), Terme mit n = d fehlen: dort ist s = 0)."""
    t, e = check_data(time, event)
    ts, d = np.unique(t[e], return_counts=True)
    d = d.astype(float)
    n = (t.size - np.searchsorted(np.sort(t), ts, side="left")).astype(float)  # #(Zeit ≥ ts)
    s = np.cumprod(1.0 - d / n) if ts.size else np.zeros(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(n > d, d / (n * (n - d)), 0.0)
    return {"t": ts, "n": n, "d": d, "s": s, "gw": np.cumsum(terms), "n_obs": int(t.size)}


def _index_at(km: dict, x: float) -> int:
    return int(np.searchsorted(km["t"], x, side="right")) - 1


def survival_at(km: dict, x: float) -> float:
    """Ŝ(x); vor der ersten Ereigniszeit 1."""
    i = _index_at(km, x)
    return 1.0 if i < 0 else float(km["s"][i])


def quantile(km: dict, q: float) -> float:
    """Kleinste Ereigniszeit mit Ŝ ≤ 1 − q; nan, wenn die Kurve das Niveau nicht erreicht („nicht schätzbar“)."""
    hit = np.where(km["s"] <= 1.0 - q + 1e-12)[0]
    return float(km["t"][hit[0]]) if hit.size else float("nan")


def greenwood_band(km: dict, x: float, level: float = 0.95, method: str = "loglog"):
    """Punktweises Konfidenzband an der Zeit x: (Ŝ, untere, obere Grenze) oder (Ŝ, nan, nan), wenn Ŝ ∈ {0, 1}.
    method „loglog“ (Standard, bleibt in [0, 1]) oder „plain“ (Ŝ ± z·SE, auf [0, 1] gekappt)."""
    i = _index_at(km, x)
    if i < 0:
        return 1.0, float("nan"), float("nan")
    s, gw = float(km["s"][i]), float(km["gw"][i])
    if s <= 0.0 or s >= 1.0:
        return s, float("nan"), float("nan")
    z = _z_two_sided(level)
    if method == "plain":
        se = s * math.sqrt(gw)
        return s, max(0.0, s - z * se), min(1.0, s + z * se)
    if method != "loglog":
        raise SurvivalError("method muss „loglog“ oder „plain“ sein.")
    se_ll = math.sqrt(gw) / abs(math.log(s))
    return s, s ** math.exp(z * se_ll), s ** math.exp(-z * se_ll)


def _z_two_sided(level: float) -> float:
    """Normalquantil für ein zweiseitiges Niveau (Bisektion über die Fehlerfunktion, ohne scipy)."""
    if not 0.0 < level < 1.0:
        raise SurvivalError("Das Niveau muss zwischen 0 und 1 liegen.")
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if math.erf(mid / math.sqrt(2.0)) < level:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ------------------------------------------------------------------------------------------------ Parametrisch: Weibull und Exponential


def weibull_fit(time, event) -> dict:
    """Weibull-Maximum-Likelihood mit Rechtszensierung, S(t) = exp(−(t/λ)^k). BFGS mit analytischem Gradienten (über log k, log λ), vier Startwerte.
    Gibt k, lam, ok, message. Ohne mindestens zwei verschiedene Ereigniszeiten gibt es keinen Schätzer (ok = False)."""
    t, e = check_data(time, event)
    if np.unique(t[e]).size < 2:
        return {"k": float("nan"), "lam": float("nan"), "ok": False, "message": "Für die Weibull-Anpassung braucht es mindestens zwei verschiedene Ereigniszeiten."}
    logt = np.log(t)
    ef = e.astype(float)
    d = float(ef.sum())

    def nll(p):
        lk, ll = p
        if abs(lk) > 12.0 or abs(ll) > 60.0:  # außerhalb jedes sinnvollen Bereichs: der Line-Search halbiert den Schritt
            return float("inf"), np.zeros(2)
        k = math.exp(lk)
        with np.errstate(over="ignore", invalid="ignore"):
            z = np.exp(np.clip(k * (logt - ll), -700.0, 700.0))
            val = -(d * (lk - ll) + np.sum(ef * (k - 1.0) * (logt - ll)) - np.sum(z))
            g_lk = -(d + k * np.sum(ef * (logt - ll)) - k * np.sum(z * (logt - ll)))
            g_ll = -(-d * k + k * np.sum(z))
        if not (np.isfinite(val) and np.isfinite(g_lk) and np.isfinite(g_ll)):
            return float("inf"), np.zeros(2)
        return val, np.array([g_lk, g_ll])

    best = None
    for k0 in (0.7, 1.0, 1.5, 2.5):
        x = np.array([math.log(k0), math.log(float(np.mean(t)))])
        x, f, ok = _bfgs(nll, x)
        if ok and (best is None or f < best[1]):
            best = (x, f)
    if best is None:
        return {"k": float("nan"), "lam": float("nan"), "ok": False, "message": "Die Weibull-Anpassung ist nicht konvergiert."}
    return {"k": math.exp(best[0][0]), "lam": math.exp(best[0][1]), "ok": True, "message": ""}


def _bfgs(fun, x0, iters: int = 200, gtol: float = 1e-9):
    """Kleines BFGS mit Armijo-Rückverfolgung für fun(x) -> (Wert, Gradient). Gibt (x, Wert, konvergiert) zurück."""
    x = np.asarray(x0, dtype=float)
    f, g = fun(x)
    h = np.eye(x.size)
    for _ in range(iters):
        if np.max(np.abs(g)) < gtol:
            return x, f, True
        p = -h @ g
        if g @ p >= 0:
            h, p = np.eye(x.size), -g
        step, ok = 1.0, False
        for _ in range(60):
            xn = x + step * p
            fn, gn = fun(xn)
            if np.isfinite(fn) and fn <= f + 1e-4 * step * (g @ p):
                ok = True
                break
            step *= 0.5
        if not ok:
            return x, f, bool(np.max(np.abs(g)) < 1e-6)
        s_vec, y = xn - x, gn - g
        sy = float(s_vec @ y)
        if sy > 1e-12:
            rho = 1.0 / sy
            i = np.eye(x.size)
            h = (i - rho * np.outer(s_vec, y)) @ h @ (i - rho * np.outer(y, s_vec)) + rho * np.outer(s_vec, s_vec)
        x, f, g = xn, fn, gn
    return x, f, bool(np.max(np.abs(g)) < 1e-6)


def exponential_fit(time, event) -> dict:
    """Exponential mit Zensierung: λ̂ = Ereignisse / Summe der Zeiten (Rate je Zeiteinheit), Mittel 1/λ̂, Median ln 2/λ̂."""
    t, e = check_data(time, event)
    if not e.any():
        return {"rate": float("nan"), "ok": False, "message": "Ohne Ereignis gibt es keinen Schätzer."}
    rate = float(e.sum()) / float(t.sum())
    return {"rate": rate, "mean": 1.0 / rate, "median": math.log(2.0) / rate, "ok": True, "message": ""}
