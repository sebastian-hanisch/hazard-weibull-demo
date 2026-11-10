"""Messung „Hochrechnen“: Quantile aus Weibull-ML, Exponential und Kaplan–Meier gegen die Wahrheit, über feste Seeds.

Aufruf (Wurzelordner der Demo):  python tools/measure_hochrechnen.py
Schreibt data/messung_hochrechnen.json (eingefroren, von der App und von tests/test_claims.py gelesen).
Zellen: Welt × n × Beobachtungsfenster; je Zelle `RUNS_EXTRAPOLATION` Läufe mit den Seeds SEEDS_EXTRAPOLATION + r (SplitMix64).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hzw_constants as C  # noqa: E402
import hzw_evaluation as E  # noqa: E402
import hzw_scenario as S  # noqa: E402


def main() -> None:
    out = {
        "meta": {
            "runs": C.RUNS_EXTRAPOLATION, "seed_base": C.SEEDS_EXTRAPOLATION, "levels": list(C.QUANTILE_LEVELS), "median": C.MEDIAN,
            "worlds": {k: w.label for k, w in S.WORLDS.items()}, "n": list(C.N_CHOICES), "windows": list(C.WINDOW_CHOICES),
            "erzeuger": "tools/measure_hochrechnen.py (SplitMix64, Weibull-ML mit BFGS)",
        },
        "cells": {},
    }
    for wkey, world in S.WORLDS.items():
        for n in C.N_CHOICES:
            for window in C.WINDOW_CHOICES:
                out["cells"][E.cell_key(wkey, n, window)] = E.extrapolation_cell(world, n, window)
                print("fertig:", wkey, n, window, flush=True)
    target = ROOT / C.DATA_DIR / C.FILE_EXTRAPOLATION
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(out, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("geschrieben:", target)


if __name__ == "__main__":
    main()
