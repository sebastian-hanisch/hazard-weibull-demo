"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen aus der Messung, nicht Seeds."""

import pytest

import hzw_evaluation as E


def _method(mean, bias, defined=100.0, sd=0.5):
    return {"mean": mean, "bias_pct": bias, "defined_pct": defined, "defined": int(defined * 2), "sd": sd, "se": 0.1, "runs": 200}


def _cell(w_bias, e_bias, km_defined=100.0, km_mean=10.0, km_bias=0.0):
    return {
        "n": 300, "window": 8, "runs": 200, "censored_pct": 55.7,
        "truth": {"0.9": 12.5},
        "weibull": {"0.9": _method(12.5 * (1 + w_bias / 100), w_bias)},
        "exponential": {"0.9": _method(12.5 * (1 + e_bias / 100), e_bias)},
        "km": {"0.9": _method(km_mean, km_bias, km_defined)},
    }


def test_verdict_extrapolation_nennt_zahlen_und_voraussetzungen():
    kind, text = E.verdict_extrapolation(_cell(-27.0, 16.0, 5.0, 7.8, -38.0), 0.9, "Welt X")
    assert kind == "warning"
    assert "Wahrheit 12,50 Tage" in text and "Weibull-ML im Mittel 9,12 (−27 %)" in text and "Exponential 14,50 (+16 %)" in text
    assert "nur in 5 % der Läufe (dort im Mittel 7,80)" in text and "keine allgemeine Regel" in text
    assert "Welt: Welt X; n = 300, Fenster 8 Tage (56 % zensiert), 200 Läufe." in text


def test_verdict_extrapolation_schwellen():
    """Die Ampel folgt der Weibull-Abweichung (Hauptverfahren der Demo): unter 5 % success, ab 10 % warning, dazwischen info; warning auch, wenn Kaplan–Meier in weniger als 50 % der Läufe definiert ist."""
    assert E.verdict_extrapolation(_cell(-4.9, 40.0), 0.9, "W")[0] == "success"                      # Exponential spielt für die Ampel keine Rolle
    kind, text = E.verdict_extrapolation(_cell(-5.0, 1.0), 0.9, "W")
    assert kind == "info" and "keine allgemeine Regel" in text                                      # genau 5 %: nicht mehr success
    assert E.verdict_extrapolation(_cell(-9.9, 1.0), 0.9, "W")[0] == "info"
    assert E.verdict_extrapolation(_cell(-10.0, 1.0), 0.9, "W")[0] == "warning"
    assert E.verdict_extrapolation(_cell(-6.0, 7.0, km_defined=49.0), 0.9, "W")[0] == "warning"      # KM fast nie definiert
    assert E.verdict_extrapolation(_cell(-6.0, 7.0, km_defined=50.0), 0.9, "W")[0] == "info"
    kind, text = E.verdict_extrapolation(_cell(0.5, 35.0), 0.9, "W")
    assert kind == "success" and "unter 5 % daneben (die Exponentialanpassung +35 %)" in text and "heißt nicht, dass die Form stimmt" in text and "keine allgemeine Regel" not in text


def test_verdict_extrapolation_km_ganz_undefiniert_und_immer_definiert():
    cell = _cell(-12.0, 1.0, km_defined=0.0)
    cell["km"]["0.9"] = {"mean": float("nan"), "bias_pct": float("nan"), "defined_pct": 0.0, "defined": 0, "sd": float("nan"), "se": float("nan"), "runs": 200}
    kind, text = E.verdict_extrapolation(cell, 0.9, "W")
    assert "in keinem Lauf" in text and kind == "warning"
    kind, text = E.verdict_extrapolation(_cell(-1.0, 1.0, km_defined=100.0, km_mean=12.4, km_bias=-0.8), 0.9, "W")
    assert "in allen Läufen" in text and "12,40 (−1 %)" in text


def _probe_cell(p09, p05):
    return {"n": 300, "window": 30, "runs": 1000, "worlds": {"lognormal_09": {"detected_pct": p09}, "lognormal_05": {"detected_pct": p05}}}


def test_verdict_probe_alarm_und_ampel():
    cell = _probe_cell(89.5, 95.4)
    kind, text = E.verdict_probe({"ok": True, "r2": 0.9, "alarm": True}, 0.974, cell)
    assert kind == "warning" and "R² = 0,900 liegt unter der Schwelle 0,974: die Probe schlägt an." in text
    assert "σ = 0,9 wird in 89,5 % der Läufe erkannt, mit σ = 0,5 in 95,4 %" in text and "n = 300, Fenster 30 Tage, 1000 Läufe je Welt" in text
    # kein Alarm: Ampel nach der Trennschärfe des Designs (beide ≥ 80: success; beide < 50: warning; sonst info)
    ok = {"ok": True, "r2": 0.99, "alarm": False}
    assert E.verdict_probe(ok, 0.974, _probe_cell(80.0, 80.0))[0] == "success"
    assert E.verdict_probe(ok, 0.974, _probe_cell(79.9, 95.0))[0] == "info"
    assert E.verdict_probe(ok, 0.974, _probe_cell(49.9, 49.9))[0] == "warning"
    assert E.verdict_probe(ok, 0.974, _probe_cell(10.0, 50.0))[0] == "info"
    kind, text = E.verdict_probe(ok, 0.974, _probe_cell(21.6, 33.2))
    assert kind == "warning" and "schlägt nicht an" in text and "21,6 %" in text and "33,2 %" in text and "nur dort ein Hinweis" in text


def test_verdict_shape_je_zweig():
    kind, text = E.verdict_shape(0.6, 6.0)
    assert "fällt" in text and "Form k = 0,6, Skala λ = 6,0 Tage" in text
    assert "konstant" in E.verdict_shape(1.0, 6.0)[1] and "kein Gedächtnis" in E.verdict_shape(1.0, 6.0)[1]
    assert "steigt" in E.verdict_shape(1.5, 6.0)[1]
    assert E.shape_kind(1.0 + 1e-12) == "konstant" and E.shape_kind(0.99) == "fallend" and E.shape_kind(1.01) == "steigend"
    assert "Median 4,70 Tage, Mittel 5,42 Tage" in E.verdict_shape(1.5, 6.0)[1]


def test_formatierung_mit_dezimalkomma_und_vorzeichen():
    assert E.fmt(1.234) == "1,23" and E.fmt(float("nan")) == "–" and E.pct(21.64) == "21,6 %"
    assert E.signed_pct(-27.2) == "−27 %" and E.signed_pct(16.4) == "+16 %" and E.signed_pct(0.0) == "0 %"
    assert E.signed_pct(2.5, 1) == "+2,5 %"
    assert E.signed_pct(0.4) == "0 %" and E.signed_pct(-0.4) == "0 %" and E.signed_pct(-0.6) == "−1 %"
