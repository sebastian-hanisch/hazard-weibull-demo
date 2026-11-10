"""Messung „Die Probe und ihre Grenze“: Wie oft erkennt der Weibull-Plot eine falsche Form?

Aufruf (Wurzelordner der Demo):  python tools/measure_probe.py
Schreibt data/messung_probe.json. Gütemaß: R² der Geraden in log(−log Ŝ) gegen log t (Kaplan–Meier-Punkte mit 0 < Ŝ < 1, mindestens 5 Punkte).
Schwelle: 5-%-Quantil von R² bei Weibull-Wahrheit (Form 1,3, Median 4 Tage), gleiches Design (n, Fenster), Seeds SEEDS_PROBE_REFERENCE + r.
Trennschärfe: Anteil der Läufe mit R² unter der Schwelle in jeder Welt, mit anderen Seeds (SEEDS_PROBE_TEST + r).
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
            "runs": C.RUNS_PROBE, "seed_reference": C.SEEDS_PROBE_REFERENCE, "seed_test": C.SEEDS_PROBE_TEST, "level": C.PROBE_LEVEL,
            "reference_shape": C.PROBE_REFERENCE_SHAPE, "stability_seed_sets": [list(p) for p in C.PROBE_STABILITY_SEEDS], "median": C.MEDIAN, "worlds": {k: w.label for k, w in S.WORLDS.items()},
            "erzeuger": "tools/measure_probe.py (SplitMix64)",
        },
        "cells": {},
    }
    for n in C.N_CHOICES:
        for window in C.WINDOW_CHOICES:
            out["cells"][E.probe_key(n, window)] = E.probe_cell(n, window)
            out["cells"][E.probe_key(n, window)]["stability"] = E.probe_stability(n, window)
            print("fertig:", n, window, flush=True)
    target = ROOT / C.DATA_DIR / C.FILE_PROBE
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(out, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("geschrieben:", target)


if __name__ == "__main__":
    main()
