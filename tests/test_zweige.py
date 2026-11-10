"""Zweig-Tests (DEMO-PLAYBOOK §0): jede Variante der App weicht auf mindestens einer Stichprobe von der Nachbarvariante ab (eine Variante, die nie etwas ändert, wäre ein toter Zweig),
und jedes Preset zeigt in Experiment 3 und 4 die erwartete Zeile (DEMO-PLAYBOOK §11a Punkt 7)."""

import os

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import hzw_constants as C
import hzw_core as K
import hzw_evaluation as E
import hzw_model as M
import hzw_scenario as S

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def test_weibull_und_exponential_geben_verschiedene_quantile():
    view = E.fit_sample(S.simulate(S.WORLDS["lognormal_09"], 300, 8, 35))
    for q in (0.9, 0.95):
        w = M.weibull_quantile(q, view.weibull["k"], view.weibull["lam"])
        e = M.exponential_quantile(q, view.exponential["rate"])
        assert abs(w / e - 1.0) > 0.05, q


def test_jede_welt_liefert_andere_daten_und_andere_wahrheit():
    samples = {k: S.simulate(w, 300, 8, 35) for k, w in S.WORLDS.items()}
    keys = list(S.WORLDS)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert not np.array_equal(samples[a].time, samples[b].time), (a, b)
            assert S.WORLDS[a].quantile(0.9) != S.WORLDS[b].quantile(0.9), (a, b)


def test_laengeres_fenster_zensiert_weniger_und_mehr_n_aendert_die_stichprobe():
    for world in S.WORLDS.values():
        shares = [S.simulate(world, 300, w, 35).censored_share for w in C.WINDOW_CHOICES]
        assert shares == sorted(shares, reverse=True) and len(set(shares)) == 3, world.key
    assert S.simulate(S.WORLDS["lognormal_09"], 100, 8, 35).n == 100 and S.simulate(S.WORLDS["lognormal_09"], 300, 8, 35).n == 300


def test_greenwood_band_plain_und_loglog_unterscheiden_sich():
    km = K.kaplan_meier([2, 3, 4, 5, 5, 8], [1, 0, 1, 1, 0, 1])
    assert K.greenwood_band(km, 4.5, method="plain")[1:] != pytest.approx(K.greenwood_band(km, 4.5, method="loglog")[1:], abs=1e-3)


def test_anderer_zufalls_seed_gibt_andere_stichprobe_und_andere_probe():
    a, b = S.simulate(S.WORLDS["lognormal_09"], 300, 8, 35), S.simulate(S.WORLDS["lognormal_09"], 300, 8, 36)
    assert not np.array_equal(a.time, b.time)
    assert E.probe_r2(a) != E.probe_r2(b)


# erwartete Ampel und Trennschärfe-Zahl je Preset (aus der README von Hand gelesen, nicht aus der Funktion)
PRESET_ERWARTUNG = {
    "Lage der Hochrechnung (lognormal, Fenster 8, n = 300)": ("warning", "Weibull-ML im Mittel 9,21 (−27 %)", "21,6 %"),
    "Viel Zeit, viele Daten (Fenster 30, n = 300)": ("info", "Weibull-ML im Mittel 11,93 (−6 %)", "89,5 %"),
    "Wenig Daten (Fenster 8, n = 100)": ("warning", "Weibull-ML im Mittel 9,18 (−28 %)", "2,6 %"),
    "Weibull ist wahr (Form 1,3, Fenster 8)": ("success", "Weibull-ML im Mittel 10,09 (0 %)", "21,6 %"),
}


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_zeigt_die_erwartete_zeile_in_experiment_3_und_4(name):
    assert set(PRESET_ERWARTUNG) == set(C.PRESETS)
    kind, phrase, power = PRESET_ERWARTUNG[name]
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception
    preset = C.PRESETS[name]
    elements = {"success": at.success, "info": at.info, "warning": at.warning}
    hits = [el.value for el in elements[kind] if phrase in el.value]
    assert hits, (name, kind, phrase, [(k, [e.value[:80] for e in v]) for k, v in elements.items()])
    assert f"n = {preset['n']}, Fenster {preset['fenster']} Tage" in hits[0] and S.WORLDS[preset["welt"]].label in hits[0]
    # Experiment 4: das Verdikt der Probe nennt die Trennschärfe des Designs (lognormal σ = 0,9)
    probe_texts = [el.value for v in elements.values() for el in v if "Trennschärfe in diesem Design" in el.value]
    assert len(probe_texts) == 1 and f"σ = 0,9 wird in {power} der Läufe erkannt" in probe_texts[0]
