"""Szenario: Containerverweilzeit und Beobachtungsfenster.

Ein Container kommt zu einem zufälligen Zeitpunkt A, gleichverteilt im Fenster [0, W), und verweilt D Tage. Beobachtet wird, bis die Abholung geschieht
oder das Fenster endet: Zeit = min(D, W − A), Ereignis = (D ≤ W − A). Die Zensierzeit W − A ist unabhängig von D (Zensierung unabhängig vom Ereignis).
Alle Zufallszahlen der Stichproben kommen aus `SplitMix64` (feste Seeds, auf anderen Plattformen nicht geprüft).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import hzw_constants as C
import hzw_core as K
import hzw_model as M


@dataclass(frozen=True)
class World:
    key: str
    label: str
    family: str            # "lognormal" oder "weibull"
    shape: float           # lognormal: σ; weibull: Form k
    median: float = C.MEDIAN

    @property
    def scale(self) -> float:
        """Nur Weibull: Skala λ bei gegebenem Median."""
        return M.weibull_scale_for_median(self.shape, self.median)

    def survival(self, t):
        return M.lognormal_survival(t, self.median, self.shape) if self.family == "lognormal" else M.weibull_survival(t, self.shape, self.scale)

    def hazard(self, t):
        return M.lognormal_hazard(t, self.median, self.shape) if self.family == "lognormal" else M.weibull_hazard(t, self.shape, self.scale)

    def quantile(self, q: float) -> float:
        return M.lognormal_quantile(q, self.median, self.shape) if self.family == "lognormal" else M.weibull_quantile(q, self.shape, self.scale)

    def mean(self) -> float:
        return M.lognormal_mean(self.median, self.shape) if self.family == "lognormal" else M.weibull_mean(self.shape, self.scale)


WORLDS = {
    "lognormal_09": World("lognormal_09", "Lognormal, σ = 0,9 (Median 4 Tage)", "lognormal", 0.9),
    "lognormal_05": World("lognormal_05", "Lognormal, σ = 0,5 (Median 4 Tage)", "lognormal", 0.5),
    "weibull_13": World("weibull_13", "Weibull, Form 1,3 (Median 4 Tage)", "weibull", 1.3),
    "weibull_07": World("weibull_07", "Weibull, Form 0,7 (Median 4 Tage)", "weibull", 0.7),
}


@dataclass(frozen=True)
class Sample:
    time: np.ndarray
    event: np.ndarray
    window: float

    @property
    def n(self) -> int:
        return int(self.time.size)

    @property
    def n_events(self) -> int:
        return int(self.event.sum())

    @property
    def censored_share(self) -> float:
        return float(1.0 - self.event.mean())


def draw_durations(rng: K.SplitMix64, world: World, n: int) -> np.ndarray:
    if world.family == "lognormal":
        return rng.lognormal(n, world.median, world.shape)
    return rng.weibull(n, world.shape, world.scale)


def observe(durations: np.ndarray, window: float, rng: K.SplitMix64) -> Sample:
    """Ankunft gleichverteilt in [0, W); Beobachtung endet bei Abholung oder am Fensterende."""
    arrival = rng.uniform(durations.size) * window
    left = window - arrival
    return Sample(np.minimum(durations, left), durations <= left, float(window))


def simulate(world: World, n: int, window: float, seed: int) -> Sample:
    """Eine Stichprobe: erst die n Verweilzeiten, dann die n Ankünfte, aus demselben Strom des Zufalls-Seeds."""
    rng = K.SplitMix64(seed)
    d = draw_durations(rng, world, n)
    return observe(d, window, rng)
