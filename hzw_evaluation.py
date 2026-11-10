"""Kennzahlen, Messungen (Hochrechnen, Weibull-Plot) und Verdikte als reine Funktionen.

Die Messungen laufen hier als Funktionen über feste Seeds; `tools/measure_hochrechnen.py` und `tools/measure_probe.py` rufen sie auf und schreiben das
Ergebnis als JSON nach `data/`. Die App liest nur diese Dateien (keine Rechnung über hunderte Läufe beim Rerun). Jedes Verdikt ist eine Funktion
`verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import hzw_constants as C
import hzw_core as K
import hzw_model as M
import hzw_scenario as S

ROOT = Path(__file__).resolve().parent


def fmt(x: float, digits: int = 2) -> str:
    """Zahl mit Dezimalkomma (die App zeigt deutsche Schreibweise)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float, digits: int = 1) -> str:
    return fmt(x, digits) + " %"


def signed_pct(x: float, digits: int = 0) -> str:
    """Prozent mit Vorzeichen (Minuszeichen „−“); was auf null rundet, steht ohne Vorzeichen."""
    r = round(x, digits)
    return ("+" if r > 0 else "−" if r < 0 else "") + fmt(abs(r), digits) + " %"


# ------------------------------------------------------------------------------------------------ Experiment 1: Gestalt


def shape_kind(k: float) -> str:
    """„fallend“ (k < 1), „konstant“ (k = 1) oder „steigend“ (k > 1): Verlauf der Abholrate mit dem Alter."""
    if abs(k - 1.0) < 1e-9:
        return "konstant"
    return "fallend" if k < 1.0 else "steigend"


def hazard_curve(k: float, lam: float, t_max: float = C.T_PLOT_MAX, points: int = 200):
    t = np.linspace(t_max / points, t_max, points)  # t > 0: für k < 1 ist h(0) unendlich
    return t, M.weibull_hazard(t, k, lam), M.weibull_survival(t, k, lam)


def stay_table(k: float, lam: float, ages=C.AGES, extra: float = C.EXTRA_DAYS) -> list[dict]:
    """Wahrscheinlichkeit, noch `extra` Tage zu bleiben, bei Alter a (Gedächtnis der Verteilung)."""
    return [{"age": a, "p": M.conditional_stay(a, extra, k, lam)} for a in ages]


def lognormal_hazard_peak(median: float, sigma: float) -> float:
    """Alter, bei dem die lognormale Abholrate am größten ist (Raster 0,01 Tage über 0,01 bis 60 Tage)."""
    t = np.arange(1, 6001) / 100.0
    return float(t[int(np.argmax(M.lognormal_hazard(t, median, sigma)))])


def verdict_shape(k: float, lam: float) -> tuple[str, str]:
    kind = shape_kind(k)
    med, mean = M.weibull_median(k, lam), M.weibull_mean(k, lam)
    head = f"Form k = {fmt(k, 1)}, Skala λ = {fmt(lam, 1)} Tage: Median {fmt(med)} Tage, Mittel {fmt(mean)} Tage. "
    if kind == "fallend":
        return "info", head + "Die Abholrate **fällt** mit dem Alter: wer schon lange da ist, wird weniger wahrscheinlich abgeholt (die Verweilzeit hat einen langen Tail, das Mittel liegt über dem Median)."
    if kind == "konstant":
        return "info", head + "Die Abholrate ist **konstant**: der Container hat kein Gedächtnis, die Wahrscheinlichkeit, noch r Tage zu bleiben, hängt nicht vom Alter ab (Exponentialverteilung)."
    return "info", head + "Die Abholrate **steigt** mit dem Alter: je länger ein Container steht, desto wahrscheinlicher wird die Abholung."


# ------------------------------------------------------------------------------------------------ Experiment 2: Anpassung an eine Stichprobe


@dataclass(frozen=True)
class FitView:
    km: dict
    weibull: dict
    exponential: dict
    t_last: float                 # größte Beobachtungszeit (Ende des Bereichs, den die Daten abdecken)
    gap_weibull: float            # größter Abstand |Ŝ_KM − S_Weibull| über die Ereigniszeiten (nan ohne Weibull-Schätzer)
    gap_exponential: float
    km_quantiles: dict            # q -> Zeit oder nan („Kurve erreicht das Niveau nicht“)


def fit_sample(sample: S.Sample, levels=C.QUANTILE_LEVELS) -> FitView:
    km = K.kaplan_meier(sample.time, sample.event)
    w = K.weibull_fit(sample.time, sample.event)
    x = K.exponential_fit(sample.time, sample.event)
    gap_w = gap_e = float("nan")
    if km["t"].size:
        if w["ok"]:
            gap_w = float(np.max(np.abs(km["s"] - M.weibull_survival(km["t"], w["k"], w["lam"]))))
        if x["ok"]:
            gap_e = float(np.max(np.abs(km["s"] - np.exp(-x["rate"] * km["t"]))))
    return FitView(km, w, x, float(sample.time.max()), gap_w, gap_e, {q: K.quantile(km, q) for q in levels})


def verdict_fit(view: FitView, window: float) -> tuple[str, str]:
    """Verdikt der Anpassung: ohne Schätzer eine Meldung, sonst der Abstand zur Kaplan–Meier-Kurve im beobachteten Bereich."""
    if not view.weibull["ok"]:
        return "warning", "Die Weibull-Anpassung hat keinen Schätzer: " + view.weibull["message"]
    if not view.exponential["ok"]:
        return "warning", "Die Exponential-Anpassung hat keinen Schätzer: " + view.exponential["message"]
    return "info", (
        f"Im beobachteten Bereich (bis Tag {fmt(view.t_last, 1)}) liegt die Weibull-Kurve höchstens {fmt(view.gap_weibull, 3)} und die Exponentialkurve "
        f"höchstens {fmt(view.gap_exponential, 3)} von der Kaplan–Meier-Kurve entfernt. Hinter Tag {fmt(view.t_last, 1)} gibt es keine Daten: dort ist die Weibull-Kurve "
        "eine Annahme der Form, kein Messwert."
    )


def level_label(q: float) -> str:
    return f"{fmt(q * 100, 0)} % der Container sind abgeholt (Quantil {fmt(q, 2)})"


# ------------------------------------------------------------------------------------------------ Experiment 3: Hochrechnen (Messung)


def quantile_estimates(sample: S.Sample, levels=C.QUANTILE_LEVELS) -> dict:
    """Quantile einer Stichprobe aus Weibull-ML, Exponential und Kaplan–Meier (nan, wo es keinen Schätzer gibt)."""
    km = K.kaplan_meier(sample.time, sample.event)
    w = K.weibull_fit(sample.time, sample.event)
    x = K.exponential_fit(sample.time, sample.event)
    nan = float("nan")
    return {
        "k": w["k"], "lam": w["lam"], "weibull_ok": bool(w["ok"]),
        "weibull": {q: (M.weibull_quantile(q, w["k"], w["lam"]) if w["ok"] else nan) for q in levels},
        "exponential": {q: (M.exponential_quantile(q, x["rate"]) if x["ok"] else nan) for q in levels},
        "km": {q: K.quantile(km, q) for q in levels},
    }


def summarize(values, truth: float) -> dict:
    """Mittel, Standardfehler und Streuung der definierten Werte, Anteil definierter Läufe, mittlere relative Abweichung von der Wahrheit in Prozent."""
    v = np.asarray(values, dtype=float)
    ok = v[~np.isnan(v)]
    out = {"defined_pct": 100.0 * ok.size / v.size, "defined": int(ok.size), "runs": int(v.size)}
    if ok.size >= 2:
        out.update({"mean": float(ok.mean()), "sd": float(ok.std(ddof=1)), "se": float(ok.std(ddof=1) / math.sqrt(ok.size)), "bias_pct": float(100.0 * (ok.mean() / truth - 1.0))})
    elif ok.size == 1:
        out.update({"mean": float(ok[0]), "sd": float("nan"), "se": float("nan"), "bias_pct": float(100.0 * (ok[0] / truth - 1.0))})
    else:
        out.update({"mean": float("nan"), "sd": float("nan"), "se": float("nan"), "bias_pct": float("nan")})
    return out


def extrapolation_cell(world: S.World, n: int, window: float, runs: int = C.RUNS_EXTRAPOLATION, levels=C.QUANTILE_LEVELS, base: int = C.SEEDS_EXTRAPOLATION) -> dict:
    """Eine Zelle der Messung: `runs` Stichproben (Seed base + r), Quantile aus drei Verfahren gegen die Wahrheit der Welt."""
    rows = []
    censored = []
    for r in range(runs):
        sample = S.simulate(world, n, window, base + r)
        censored.append(sample.censored_share)
        rows.append(quantile_estimates(sample, levels))
    cell = {
        "world": world.key, "n": n, "window": window, "runs": runs,
        "censored_pct": 100.0 * float(np.mean(censored)),
        "weibull_failed": int(sum(not r["weibull_ok"] for r in rows)),
        "k_hat_mean": float(np.nanmean([r["k"] for r in rows])),
        "truth": {}, "weibull": {}, "exponential": {}, "km": {},
    }
    for q in levels:
        truth = world.quantile(q)
        cell["truth"][str(q)] = truth
        for method in ("weibull", "exponential", "km"):
            cell[method][str(q)] = summarize([r[method][q] for r in rows], truth)
    return cell


def cell_key(world: str, n: int, window: float) -> str:
    return f"{world}|{int(n)}|{int(window)}"


def load_extrapolation(path: Path | None = None) -> dict:
    return json.loads((path or ROOT / C.DATA_DIR / C.FILE_EXTRAPOLATION).read_text(encoding="utf-8"))


def verdict_extrapolation(cell: dict, q: float, world_label: str) -> tuple[str, str]:
    """Verdikt des Hochrechnens an einer Zelle der Messung. Aussagen gelten für diese Welt, dieses n und dieses Fenster, nicht allgemein."""
    key = str(q)
    truth = cell["truth"][key]
    w, e, k = cell["weibull"][key], cell["exponential"][key], cell["km"][key]
    cond = f"Welt: {world_label}; n = {cell['n']}, Fenster {cell['window']} Tage ({fmt(cell['censored_pct'], 0)} % zensiert), {cell['runs']} Läufe."
    parts = [f"Wahrheit {fmt(truth)} Tage."]
    parts.append(f"Weibull-ML im Mittel {fmt(w['mean'])} ({signed_pct(w['bias_pct'])}), Exponential {fmt(e['mean'])} ({signed_pct(e['bias_pct'])}).")
    if k["defined_pct"] >= 100.0:
        parts.append(f"Kaplan–Meier erreicht das Niveau in allen Läufen und liegt im Mittel bei {fmt(k['mean'])} ({signed_pct(k['bias_pct'])}).")
    elif k["defined"] == 0:
        parts.append("Kaplan–Meier erreicht das Niveau in keinem Lauf: die Kurve endet am Fenster, ohne unter diese Marke zu fallen.")
    else:
        parts.append(f"Kaplan–Meier erreicht das Niveau nur in {fmt(k['defined_pct'], 0)} % der Läufe (dort im Mittel {fmt(k['mean'])}).")
    wb = abs(w["bias_pct"])
    kind = "success" if wb < 5.0 else "warning" if (wb >= 10.0 or k["defined_pct"] < 50.0) else "info"
    if wb < 5.0:
        tail = (f" Bei diesem Quantil liegt die Weibull-Anpassung unter 5 % daneben (die Exponentialanpassung {signed_pct(e['bias_pct'])}); das heißt nicht, dass die Form stimmt: "
                "bei anderen Quantilen kann die Abweichung größer sein.")
    else:
        tail = " Diese Abweichungen sind die gemessenen Richtungen in dieser Welt, keine allgemeine Regel."
    return kind, " ".join(parts) + tail + " " + cond


# ------------------------------------------------------------------------------------------------ Experiment 4: Weibull-Plot (Messung)


def probe_r2(sample: S.Sample) -> float:
    """R² des Weibull-Plots einer Stichprobe; nan, wenn zu wenige Punkte für die Gerade da sind."""
    res = M.plot_r2(sample.time, sample.event)
    return float("nan") if res is None else float(res[0])


def probe_values(world: S.World, n: int, window: float, runs: int, base: int) -> tuple[np.ndarray, float]:
    """R² über `runs` Läufe (Seed base + r) und mittlerer Anteil zensierter Beobachtungen."""
    vals, cens = [], []
    for r in range(runs):
        sample = S.simulate(world, n, window, base + r)
        vals.append(probe_r2(sample))
        cens.append(sample.censored_share)
    return np.array(vals), float(np.mean(cens))


def probe_cell(n: int, window: float, runs: int = C.RUNS_PROBE, ref_base: int = C.SEEDS_PROBE_REFERENCE, test_base: int = C.SEEDS_PROBE_TEST,
               worlds=None, with_shapes: bool = True) -> dict:
    """Schwelle (5-%-Quantil von R² bei Weibull-Wahrheit, Form 1,3) und Trennschärfe je Welt für ein Design (n, Fenster).
    Schwelle und Trennschärfe nutzen verschiedene Seeds (ref_base, test_base)."""
    ref_world = S.WORLDS["weibull_13"]
    ref, ref_cens = probe_values(ref_world, n, window, runs, ref_base)
    threshold = float(np.nanquantile(ref, C.PROBE_LEVEL))
    cell = {"n": n, "window": window, "runs": runs, "threshold": threshold, "ref_mean": float(np.nanmean(ref)), "ref_undefined": int(np.isnan(ref).sum()),
            "ref_censored_pct": 100.0 * ref_cens, "worlds": {}}
    for key in (worlds or S.WORLDS):
        vals, cens = probe_values(S.WORLDS[key], n, window, runs, test_base)
        cell["worlds"][key] = {"r2_mean": float(np.nanmean(vals)), "detected_pct": float(100.0 * np.mean(vals < threshold)), "undefined": int(np.isnan(vals).sum()),
                               "censored_pct": 100.0 * cens}
    if with_shapes:
        cell["threshold_by_shape"] = {}
        for shape in (0.7, 1.3, 2.5):  # Empfindlichkeit der Schwelle gegen die Form der Weibull-Wahrheit
            w = S.World(f"weibull_{shape}", f"Weibull {shape}", "weibull", shape)
            vals, _ = probe_values(w, n, window, runs, ref_base)
            cell["threshold_by_shape"][str(shape)] = float(np.nanquantile(vals, C.PROBE_LEVEL))
    return cell


def probe_stability(n: int, window: float, seed_sets=C.PROBE_STABILITY_SEEDS, runs: int = C.RUNS_PROBE) -> dict:
    """Gegenprobe der Trennschärfe mit weiteren Seed-Paaren (Schwelle und Test neu gezogen): je Welt die Liste der Anteile in Prozent."""
    keys = ("lognormal_09", "lognormal_05", "weibull_13")
    out = {"seed_sets": [list(p) for p in seed_sets], **{k: [] for k in keys}}
    for ref_base, test_base in seed_sets:
        c = probe_cell(n, window, runs, ref_base, test_base, worlds=keys, with_shapes=False)
        for k in keys:
            out[k].append(c["worlds"][k]["detected_pct"])
    return out


def probe_key(n: int, window: float) -> str:
    return f"{int(n)}|{int(window)}"


def load_probe(path: Path | None = None) -> dict:
    return json.loads((path or ROOT / C.DATA_DIR / C.FILE_PROBE).read_text(encoding="utf-8"))


def probe_for_sample(sample: S.Sample, threshold: float) -> dict:
    """Ergebnis der Probe an einer Stichprobe: R², Steigung (Schätzung von k), Punkte und ob R² unter der Schwelle liegt (nan: nicht berechenbar)."""
    res = M.plot_r2(sample.time, sample.event)
    if res is None:
        return {"ok": False, "r2": float("nan"), "slope": float("nan"), "x": np.zeros(0), "y": np.zeros(0), "alarm": False}
    r2, slope, intercept, x, y = res
    return {"ok": True, "r2": r2, "slope": slope, "intercept": intercept, "x": x, "y": y, "alarm": bool(r2 < threshold)}


def verdict_probe(result: dict, threshold: float, cell: dict) -> tuple[str, str]:
    """Verdikt der Probe an einer Stichprobe, mit der gemessenen Trennschärfe des Designs daneben (Urteil nur aus dem Design, nie aus der Wahrheit der Welt)."""
    power09 = cell["worlds"]["lognormal_09"]["detected_pct"]
    power05 = cell["worlds"]["lognormal_05"]["detected_pct"]
    design = f"Trennschärfe in diesem Design (n = {cell['n']}, Fenster {cell['window']} Tage, {cell['runs']} Läufe je Welt): lognormale Wahrheit mit σ = 0,9 wird in {fmt(power09, 1)} % der Läufe erkannt, mit σ = 0,5 in {fmt(power05, 1)} %."
    if not result["ok"]:
        return "warning", f"Die Probe ist hier nicht berechenbar: weniger als {M.MIN_PLOT_POINTS} Punkte mit 0 < Ŝ < 1. " + design
    if result["alarm"]:
        return "warning", f"R² = {fmt(result['r2'], 3)} liegt unter der Schwelle {fmt(threshold, 3)}: die Probe schlägt an. " + design
    kind = "success" if min(power09, power05) >= C.POWER_HIGH else "warning" if max(power09, power05) < C.POWER_LOW else "info"
    return kind, f"R² = {fmt(result['r2'], 3)} liegt über der Schwelle {fmt(threshold, 3)}: die Probe schlägt nicht an. Dass sie nicht anschlägt, ist nur dort ein Hinweis auf eine passende Form, wo sie trennscharf ist. " + design


# ------------------------------------------------------------------------------------------------ Tabellen für die App


def quantile_rows(view: FitView, world: S.World, levels=C.QUANTILE_LEVELS) -> list[dict]:
    """Quantile einer Stichprobe aus den drei Verfahren neben der Wahrheit der Welt, als fertige Texte (deutsche Schreibweise)."""
    rows = []
    for q in levels:
        w = M.weibull_quantile(q, view.weibull["k"], view.weibull["lam"]) if view.weibull["ok"] else float("nan")
        e = M.exponential_quantile(q, view.exponential["rate"]) if view.exponential["ok"] else float("nan")
        km = view.km_quantiles[q]
        rows.append({
            "Anteil abgeholt": pct(q * 100, 0),
            "Wahrheit (Tage)": fmt(world.quantile(q)),
            "Weibull-ML": fmt(w) if not math.isnan(w) else "kein Schätzer",
            "Exponential": fmt(e) if not math.isnan(e) else "kein Schätzer",
            "Kaplan–Meier": fmt(km) if not math.isnan(km) else "nicht erreicht",
        })
    return rows


def extrapolation_rows(cell: dict, q: float) -> list[dict]:
    """Zeilen der Messtabelle zu Experiment 3 (ein Niveau): Mittel, Standardfehler, Abweichung von der Wahrheit, Anteil der Läufe mit Schätzer."""
    key = str(q)
    rows = []
    for method, label in (("weibull", "Weibull-ML"), ("exponential", "Exponential"), ("km", "Kaplan–Meier")):
        d = cell[method][key]
        rows.append({
            "Verfahren": label,
            "Mittel (Tage)": fmt(d["mean"]),
            "Standardfehler": fmt(d["se"], 2),
            "Abweichung von der Wahrheit": signed_pct(d["bias_pct"]) if not math.isnan(d["bias_pct"]) else "–",
            "Läufe mit Schätzer": pct(d["defined_pct"], 0),
        })
    return rows


def probe_rows(probe: dict, current: tuple[int, int]) -> list[dict]:
    """Trennschärfe-Tabelle aller sechs Designs; das gewählte Design ist mit ▶ markiert."""
    rows = []
    for key, c in sorted(probe["cells"].items(), key=lambda kv: tuple(int(v) for v in kv[0].split("|"))):
        n, window = (int(v) for v in key.split("|"))
        w = c["worlds"]
        rows.append({
            "Design": ("▶ " if (n, window) == current else "") + f"n = {n}, Fenster {window} Tage",
            "zensiert (Weibull-Wahrheit)": pct(c["ref_censored_pct"], 0),
            "Schwelle R²": fmt(c["threshold"], 3),
            "lognormal σ = 0,9 erkannt": pct(w["lognormal_09"]["detected_pct"]),
            "lognormal σ = 0,5 erkannt": pct(w["lognormal_05"]["detected_pct"]),
            "Weibull 0,7 (Fehlalarm)": pct(w["weibull_07"]["detected_pct"]),
            "Weibull 1,3 (Fehlalarm)": pct(w["weibull_13"]["detected_pct"]),
        })
    return rows
