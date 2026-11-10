"""Zahlen und Aussagen der README gegen die eingefrorenen Messdateien (DEMO-PLAYBOOK §4, §11a).

* `test_readme_zahlen_stehen_in_der_messung`: jede in der README genannte Messzahl wird aus `data/*.json` mit denselben Formatfunktionen wie die App gebildet und muss
  als Text in der README stehen (Zahl → Messung).
* `test_messung_*_ist_reproduzierbar`: je eine Zelle wird mit denselben Seeds vollständig neu gerechnet und mit der Messdatei verglichen (Toleranz 1e-6 relativ).
* Die Ordnung der Aussagen (Vorzeichen, „in allen Läufen“, Spannen, „80 % nur bei n = 300“) wird über alle Zellen gezählt, nicht an einer Zelle gelesen.
Exakt nachgerechnet ist nur das Handbeispiel (tests/test_core.py); alle anderen Zahlen sind Zufallsergebnisse und stehen hier als Messdatei-Vergleich.
"""

import math
from pathlib import Path

import pytest

import hzw_constants as C
import hzw_evaluation as E
import hzw_scenario as S

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
EXTRA = E.load_extrapolation()
PROBE = E.load_probe()


def ex(world, n, window, q, method):
    return EXTRA["cells"][E.cell_key(world, n, window)][method][str(q)]


def med(world, n, window, q, method):
    d = ex(world, n, window, q, method)
    return f"{E.fmt(d['mean'])} ({E.signed_pct(d['bias_pct'])})"


def full(world, n, window, q, method):
    d = ex(world, n, window, q, method)
    return f"{E.fmt(d['mean'])} ± {E.fmt(d['se'])} ({E.signed_pct(d['bias_pct'])})"


def power(n, window, world):
    return E.fmt(PROBE["cells"][E.probe_key(n, window)]["worlds"][world]["detected_pct"], 1) + " %"


def span(values):
    return f"{E.fmt(min(values), 1)} bis {E.fmt(max(values), 1)} %"


def stab(n, window, world):
    return PROBE["cells"][E.probe_key(n, window)]["stability"][world]


def test_readme_zahlen_stehen_in_der_messung():
    lead = EXTRA["cells"][E.cell_key("lognormal_09", 300, 8)]
    expected = [
        # Hauptmessung (Zelle lognormal_09|300|8, Quantil 0,9)
        f"Wahrheit {E.fmt(lead['truth']['0.9'])} Tage", f"(Wahrheit {E.fmt(lead['truth']['0.9'])} Tage)",
        "Weibull-ML " + full("lognormal_09", 300, 8, 0.9, "weibull"), "Exponential " + full("lognormal_09", 300, 8, 0.9, "exponential"),
        f"nur in {E.fmt(ex('lognormal_09', 300, 8, 0.9, 'km')['defined_pct'], 0)} % der Läufe ({ex('lognormal_09', 300, 8, 0.9, 'km')['defined']} von {lead['runs']}, dort im Mittel {E.fmt(ex('lognormal_09', 300, 8, 0.9, 'km')['mean'])})",
        f"{E.fmt(lead['censored_pct'], 0)} % zensiert", f"um {E.fmt(abs(ex('lognormal_09', 300, 8, 0.9, 'weibull')['bias_pct']), 0)} % unter der Wahrheit",
        f"um {E.fmt(ex('lognormal_09', 300, 8, 0.9, 'exponential')['bias_pct'], 0)} % darüber", f"in {power(300, 8, 'lognormal_09')[:-2]} % der Läufe",
        # Fenster 30
        "Weibull-ML " + med("lognormal_09", 300, 30, 0.9, "weibull"), "Exponential " + med("lognormal_09", 300, 30, 0.9, "exponential"),
        "bei " + E.fmt(ex("lognormal_09", 300, 30, 0.9, "km")["mean"]) + " (" + E.signed_pct(ex("lognormal_09", 300, 30, 0.9, "km")["bias_pct"]) + ")",
        f"({E.fmt(EXTRA['cells'][E.cell_key('lognormal_09', 300, 30)]['censored_pct'], 0)} % zensiert)",
        # Median innerhalb des Fensters
        "Kaplan–Meier " + full("lognormal_09", 300, 8, 0.5, "km"), "Weibull-ML " + med("lognormal_09", 300, 8, 0.5, "weibull"), "Exponential " + med("lognormal_09", 300, 8, 0.5, "exponential"),
        # Weibull im beobachteten Bereich verzerrt (Fenster 30, Median)
        "Weibull-ML " + med("lognormal_09", 300, 30, 0.5, "weibull"), "Exponential " + med("lognormal_09", 300, 30, 0.5, "exponential"), "Kaplan–Meier " + med("lognormal_09", 300, 30, 0.5, "km"),
        # Richtung hängt an der Welt (Quantil 0,9, Fenster 8, n = 300)
        "lognormal σ = 0,9 " + E.signed_pct(ex("lognormal_09", 300, 8, 0.9, "exponential")["bias_pct"]), "lognormal σ = 0,5 " + E.signed_pct(ex("lognormal_05", 300, 8, 0.9, "exponential")["bias_pct"]),
        "Weibull Form 0,7 " + E.signed_pct(ex("weibull_07", 300, 8, 0.9, "exponential")["bias_pct"]),
        "Weibull-ML: " + E.signed_pct(ex("lognormal_09", 300, 8, 0.9, "weibull")["bias_pct"]) + ", " + E.signed_pct(ex("lognormal_05", 300, 8, 0.9, "weibull")["bias_pct"])
        + " und " + E.signed_pct(ex("weibull_07", 300, 8, 0.9, "weibull")["bias_pct"]),
        # Form richtig
        f"(Wahrheit {E.fmt(EXTRA['cells'][E.cell_key('weibull_13', 300, 8)]['truth']['0.9'])})", "Weibull-ML " + med("weibull_13", 300, 8, 0.9, "weibull"), "Exponential " + med("weibull_13", 300, 8, 0.9, "exponential"),
        f"liegt bei {E.fmt(EXTRA['cells'][E.cell_key('weibull_13', 300, 8)]['k_hat_mean'])}",
        # weniger Daten
        "Weibull-ML " + med("lognormal_09", 100, 8, 0.9, "weibull"), "Exponential " + med("lognormal_09", 100, 8, 0.9, "exponential"),
        f"in {E.fmt(ex('lognormal_09', 100, 8, 0.9, 'km')['defined_pct'], 1)} % der Läufe ({ex('lognormal_09', 100, 8, 0.9, 'km')['defined']} von {lead['runs']}) definiert",
        # Trennschärfe
        f"{power(300, 30, 'lognormal_09')} ({power(300, 30, 'lognormal_05')})", f"{power(300, 8, 'lognormal_09')} ({power(300, 8, 'lognormal_05')})", f"{power(100, 8, 'lognormal_09')} ({power(100, 8, 'lognormal_05')})",
        f"Fenster 30 Tage, *n* = 300 ({E.fmt(PROBE['cells'][E.probe_key(300, 30)]['worlds']['lognormal_09']['censored_pct'], 0)} % zensiert)",
        f"Fenster 8 Tage, *n* = 300 ({E.fmt(PROBE['cells'][E.probe_key(300, 8)]['worlds']['lognormal_09']['censored_pct'], 0)} % zensiert)",
        # Gegenproben
        "Fenster 30, *n* = 300: " + span(stab(300, 30, "lognormal_09")), "Fenster 8, *n* = 300: " + span(stab(300, 8, "lognormal_09")), "Fenster 8, *n* = 100: " + span(stab(100, 8, "lognormal_09")),
        f"zwischen {E.fmt(min(stab(300, 8, 'lognormal_09')), 1)} und {E.fmt(max(stab(300, 8, 'lognormal_09')), 1)} %",
        # Fehlalarme
        span([c["worlds"]["weibull_13"]["detected_pct"] for c in PROBE["cells"].values()]), span([c["worlds"]["weibull_07"]["detected_pct"] for c in PROBE["cells"].values()]),
        # Schwelle nach Form
        " / ".join(E.fmt(PROBE["cells"][E.probe_key(300, 8)]["threshold_by_shape"][k], 3) for k in ("0.7", "1.3", "2.5")),
        # Handbeispiel
        "ρ̂ = 4/27 ≈ 0,148", "6,75 Tage", "4,68 Tage",
    ]
    missing = [x for x in expected if x not in README]
    assert not missing, missing


def test_readme_nennt_die_zahl_der_laeufe_und_die_seeds_der_messung():
    assert f"{EXTRA['meta']['runs']} Läufe je Zelle" in README and f"{PROBE['meta']['runs']} Läufe je Welt" in README
    assert EXTRA["meta"]["runs"] == C.RUNS_EXTRAPOLATION and PROBE["meta"]["runs"] == C.RUNS_PROBE
    assert "drei weiteren Seed-Paaren" in README and len(PROBE["meta"]["stability_seed_sets"]) == 3


def test_aussagen_der_readme_gelten_ueber_alle_zellen():
    lead = "lognormal_09"
    # „Beide Formen sind falsch, in entgegengesetzter Richtung“ (Hauptlage)
    w, e = ex(lead, 300, 8, 0.9, "weibull")["bias_pct"], ex(lead, 300, 8, 0.9, "exponential")["bias_pct"]
    assert w < -20 and e > 10
    # „Mit weniger Zensierung rückt die Form in den Hintergrund“: beim Quantil 0,9 sind beide Abweichungen bei Fenster 30 kleiner als bei Fenster 8 (n = 300 und n = 100)
    for n in (300, 100):
        for method in ("weibull", "exponential"):
            assert abs(ex(lead, n, 30, 0.9, method)["bias_pct"]) < abs(ex(lead, n, 8, 0.9, method)["bias_pct"]), (n, method)
    assert ex(lead, 300, 30, 0.9, "km")["defined_pct"] == 100.0  # „in allen 200 Läufen definiert“
    # „Eine Regel ‚Form falsch, Quantil zu klein‘ oder ‚zu groß‘ gibt es nicht“: das Vorzeichen der Exponential-Abweichung wechselt zwischen den Welten (Quantil 0,9, Fenster 8, n = 300)
    signs = {k: math.copysign(1, ex(k, 300, 8, 0.9, "exponential")["bias_pct"]) for k in S.WORLDS if k != "weibull_13"}
    assert signs["lognormal_09"] > 0 and signs["weibull_07"] < 0
    # „Form richtig“: Weibull-Wahrheit, die Weibull-Abweichung liegt in allen drei Fenstern und beiden n unter 2 % (Quantil 0,9)
    for n in C.N_CHOICES:
        for window in C.WINDOW_CHOICES:
            assert abs(ex("weibull_13", n, window, 0.9, "weibull")["bias_pct"]) < 2.0, (n, window)
    # „Weibull-ML ist auch im beobachteten Bereich verzerrt“: Median in der lognormalen Hauptwelt in allen Fenstern größer als die Wahrheit, bei Fenster 30 mit +13 %
    for window in C.WINDOW_CHOICES:
        assert ex(lead, 300, window, 0.5, "weibull")["bias_pct"] > 4.0, window
    # Weibull-Anpassung nie ohne Schätzer in den Messungen („0 von 200“ steht in der App)
    assert all(c["weibull_failed"] == 0 for c in EXTRA["cells"].values())


def test_trennschaerfe_nur_bei_n_300_und_wenig_zensierung():
    """„80 % erreicht die Trennschärfe nur bei n = 300 (σ = 0,9: nur bei Fenster 30; σ = 0,5: bei Fenster 15 rund 75 bis 81 %, bei Fenster 30 über 94 %); bei n = 100 bleibt sie in allen drei Fenstern unter 25 %.“
    Die Schwelle 80 % gilt für die Hauptmessung; die Gegenproben von σ = 0,5 bei Fenster 15 liegen bei 75,2 bis 80,6 %."""
    hit = {(n, w): {k: c["worlds"][k]["detected_pct"] for k in ("lognormal_09", "lognormal_05")} for (n, w), c in
           ((tuple(int(v) for v in key.split("|")), c) for key, c in PROBE["cells"].items())}
    assert {nw for nw, v in hit.items() if v["lognormal_09"] >= 80.0} == {(300, 30)}
    assert {nw for nw, v in hit.items() if v["lognormal_05"] >= 80.0} == {(300, 15), (300, 30)}
    assert all(max(v.values()) < 25.0 for (n, w), v in hit.items() if n == 100)
    # monoton: bei gleichem n wächst die Trennschärfe mit dem Fenster, bei gleichem Fenster mit n (σ = 0,9)
    for n in (100, 300):
        vals = [hit[(n, w)]["lognormal_09"] for w in C.WINDOW_CHOICES]
        assert vals == sorted(vals)
    for w in C.WINDOW_CHOICES:
        assert hit[(300, w)]["lognormal_09"] > hit[(100, w)]["lognormal_09"]


def test_gegenprobe_spannen():
    """Hauptmessung und drei Gegenproben (weitere Seed-Paare) einer Zelle liegen höchstens 12 Prozentpunkte auseinander (σ = 0,9 und σ = 0,5; die größte Spanne gibt es in den Designs mit
    mittlerer Trennschärfe); der Fehlalarm bei Weibull 1,3 bleibt unter 8 %, bei Weibull 0,7 unter 9 %, und kein Lauf ist unberechenbar."""
    for key, c in PROBE["cells"].items():
        for world in ("lognormal_09", "lognormal_05"):
            values = [c["worlds"][world]["detected_pct"], *c["stability"][world]]
            assert max(values) - min(values) < 12.0, (key, world, values)
        assert all(v < 8.0 for v in c["stability"]["weibull_13"]), key
        assert c["worlds"]["weibull_13"]["detected_pct"] < 8.0 and c["worlds"]["weibull_07"]["detected_pct"] < 9.0, key
        assert c["ref_undefined"] == 0 and all(v["undefined"] == 0 for v in c["worlds"].values()), key


def _same(a, b, path=""):
    """Rekursiver Vergleich zweier JSON-Strukturen: Zahlen relativ 1e-6 (nan gleich nan), alles andere exakt."""
    if isinstance(a, dict):
        assert set(a) <= set(b), path
        for k in a:
            _same(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        if math.isnan(a):
            assert isinstance(b, float) and math.isnan(b), path
        else:
            assert a == pytest.approx(b, rel=1e-6, abs=1e-9), path
    else:
        assert a == b, path


def test_messung_hochrechnen_ist_reproduzierbar():
    """Die Hauptzelle (lognormal, n = 300, Fenster 8 Tage, 200 Läufe) wird mit denselben Seeds neu gerechnet."""
    fresh = E.extrapolation_cell(S.WORLDS["lognormal_09"], 300, 8)
    _same(fresh, EXTRA["cells"][E.cell_key("lognormal_09", 300, 8)])


def test_messung_probe_ist_reproduzierbar():
    """Das Design Fenster 8 / n = 300 (Schwelle, vier Welten, Schwellen nach Form, drei Gegenproben) wird mit denselben Seeds neu gerechnet."""
    fresh = E.probe_cell(300, 8)
    fresh["stability"] = E.probe_stability(300, 8)
    _same(fresh, PROBE["cells"][E.probe_key(300, 8)])


def test_messdateien_sind_vollstaendig_und_konsistent():
    assert set(EXTRA["cells"]) == {E.cell_key(w, n, f) for w in S.WORLDS for n in C.N_CHOICES for f in C.WINDOW_CHOICES}
    for key, cell in EXTRA["cells"].items():
        assert cell["runs"] == EXTRA["meta"]["runs"], key
        for q in C.QUANTILE_LEVELS:
            for method in ("weibull", "exponential", "km"):
                d = cell[method][str(q)]
                assert d["runs"] == cell["runs"] and 0.0 <= d["defined_pct"] <= 100.0 and d["defined"] <= d["runs"], (key, q, method)
            assert cell["truth"][str(q)] == pytest.approx(S.WORLDS[cell["world"]].quantile(q), rel=1e-12), (key, q)
        # Quantile wachsen mit dem Niveau (Wahrheit und Mittel der Weibull-Anpassung)
        truths = [cell["truth"][str(q)] for q in C.QUANTILE_LEVELS]
        assert truths == sorted(truths), key
    for key, cell in PROBE["cells"].items():
        assert cell["runs"] == PROBE["meta"]["runs"] and 0.5 < cell["threshold"] < 1.0, key
