"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12), auf die Routinen dieser Demo geschrieben:
kein Ereignis, ein Ereignis, genau zwei Ereignisse, alle Zeiten gleich, Zeiten ≤ 0, kleinste und größte Reglerwerte, Stichprobe ohne Weibull-Schätzer in der App-Logik."""

import math

import numpy as np
import pytest

import hzw_constants as C
import hzw_core as K
import hzw_evaluation as E
import hzw_model as M
import hzw_scenario as S


def test_genau_zwei_ereigniszeiten_reichen_fuer_weibull():
    """Zwei verschiedene Ereigniszeiten sind die Untergrenze: der Schätzer existiert (ob sinnvoll, ist eine andere Frage), ein Ereignis allein reicht nicht."""
    r = K.weibull_fit([2.0, 5.0, 6.0], [1, 1, 0])
    assert r["ok"] and r["k"] > 0 and r["lam"] > 0
    assert K.weibull_fit([2.0, 5.0], [1, 0])["ok"] is False


def test_kein_ereignis_gibt_meldung_statt_zahl():
    view = E.fit_sample(S.Sample(np.array([3.0, 4.0, 5.0]), np.array([False, False, False]), 8.0))
    assert not view.weibull["ok"] and not view.exponential["ok"]
    assert math.isnan(view.gap_weibull) and math.isnan(view.gap_exponential)
    kind, text = E.verdict_fit(view, 8.0)
    assert kind == "warning" and "keinen Schätzer" in text and "mindestens zwei verschiedene Ereigniszeiten" in text
    rows = E.quantile_rows(view, S.WORLDS["lognormal_09"])
    assert all(r["Weibull-ML"] == "kein Schätzer" and r["Exponential"] == "kein Schätzer" and r["Kaplan–Meier"] == "nicht erreicht" for r in rows)


def test_ein_ereignis_exponential_ja_weibull_nein():
    view = E.fit_sample(S.Sample(np.array([3.0, 4.0, 5.0]), np.array([True, False, False]), 8.0))
    assert view.exponential["ok"] and not view.weibull["ok"]
    kind, text = E.verdict_fit(view, 8.0)
    assert kind == "warning" and text.startswith("Die Weibull-Anpassung hat keinen Schätzer")


def test_alle_zeiten_gleich():
    t = np.full(5, 4.0)
    assert K.weibull_fit(t, np.ones(5, bool))["ok"] is False
    assert K.exponential_fit(t, np.ones(5, bool))["rate"] == pytest.approx(0.25, abs=1e-15)
    assert M.plot_r2(t, np.ones(5, bool)) is None


def test_zeiten_nicht_positiv_sind_ungueltig():
    for bad in ([0.0, 1.0, 2.0], [-1.0, 1.0, 2.0]):
        for fit in (K.weibull_fit, K.exponential_fit, K.kaplan_meier, M.plot_r2):
            with pytest.raises(K.SurvivalError):
                fit(bad, [1, 1, 1])


def test_reglergrenzen_der_formen_geben_endliche_kurven():
    for k in (C.FORM_MIN, 1.0, C.FORM_MAX):
        for lam in (C.SKALA_MIN, C.SKALA_MAX):
            t, h, s = E.hazard_curve(k, lam)
            assert np.all(np.isfinite(h)) and np.all(np.isfinite(s)) and np.all((s >= 0) & (s <= 1))
            kind, text = E.verdict_shape(k, lam)
            assert kind == "info" and text
            assert all(0 <= r["p"] <= 1 for r in E.stay_table(k, lam))


def test_kleinste_und_groesste_stichprobe_aller_welten_laufen_durch():
    for world in S.WORLDS.values():
        for n in C.N_CHOICES:
            for window in C.WINDOW_CHOICES:
                view = E.fit_sample(S.simulate(world, n, window, 1))
                assert view.weibull["ok"] and view.exponential["ok"], (world.key, n, window)
                assert len(E.quantile_rows(view, world)) == len(C.QUANTILE_LEVELS)


def test_summarize_ohne_definierte_laeufe():
    d = E.summarize([float("nan")] * 5, 10.0)
    assert d["defined"] == 0 and d["defined_pct"] == 0.0 and math.isnan(d["mean"]) and math.isnan(d["bias_pct"])
    one = E.summarize([12.0, float("nan")], 10.0)
    assert one["defined"] == 1 and one["mean"] == 12.0 and one["bias_pct"] == pytest.approx(20.0)


def test_eintritt_der_probe_bei_zu_wenig_punkten_ist_nicht_berechenbar():
    res = E.probe_for_sample(S.Sample(np.array([1.0, 2.0, 3.0, 4.0]), np.array([True] * 4), 8.0), 0.9)
    assert res["ok"] is False and math.isnan(res["r2"]) and res["alarm"] is False
    cell = {"n": 100, "window": 8, "runs": 10, "worlds": {"lognormal_09": {"detected_pct": 3.0}, "lognormal_05": {"detected_pct": 4.0}}}
    kind, text = E.verdict_probe(res, 0.9, cell)
    assert kind == "warning" and "nicht berechenbar" in text
