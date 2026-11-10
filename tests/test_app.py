"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button,
fährt jeden Regler an seine Grenzen bzw. durch jede Auswahl und prüft, dass kein Python-Fehler auftritt (insbesondere
`StreamlitDuplicateElementId` bei Diagrammen ohne eindeutiges key= und `StreamlitAPIException` bei Reglern mit berechneten Grenzen)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import hzw_constants as C
import hzw_scenario as S

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    at = _fresh_app()
    assert len(at.metric) == 4 and at.title[0].value


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("world", list(S.WORLDS))
@pytest.mark.parametrize("window", C.WINDOW_CHOICES)
def test_every_world_and_window_runs_with_both_sample_sizes(world, window):
    for n in C.N_CHOICES:
        at = _fresh_app()
        at.selectbox(key="welt_select").set_value(world)
        at.radio(key="fenster_radio").set_value(window)
        at.radio(key="n_radio").set_value(n).run(timeout=TIMEOUT)
        assert not at.exception, (world, window, n, [str(e) for e in at.exception])
        assert [m.value for m in at.metric][0].endswith(f"von {n}")


def test_every_quantile_level_and_extreme_form_with_scale_runs():
    for q in C.QUANTILE_LEVELS:
        at = _fresh_app()
        at.select_slider(key="level_select").set_value(q).run(timeout=TIMEOUT)
        assert not at.exception, q
    at = _fresh_app()
    at.slider(key="form_slider").set_value(C.FORM_MIN)
    at.slider(key="skala_slider").set_value(C.SKALA_MIN).run(timeout=TIMEOUT)
    assert not at.exception
    at.slider(key="form_slider").set_value(C.FORM_MAX)
    at.slider(key="skala_slider").set_value(C.SKALA_MAX).run(timeout=TIMEOUT)
    assert not at.exception


def test_extreme_seeds_run():
    for seed in (0, C.SEED_MAX):
        at = _fresh_app()
        at.number_input(key="seed_input").set_value(seed).run(timeout=TIMEOUT)
        assert not at.exception, seed


def test_permalink_with_invalid_values_is_ignored_and_valid_ones_are_taken():
    at = AppTest.from_file(APP_PATH)
    at.query_params["welt"] = "gibt_es_nicht"
    at.query_params["n"] = "123"
    at.query_params["fenster"] = "15"
    at.query_params["form"] = "99"
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert at.selectbox(key="welt_select").value == C.DEFAULT_WORLD   # ungültig: Standard
    assert at.radio(key="n_radio").value == C.DEFAULT_N               # ungültig: Standard
    assert at.radio(key="fenster_radio").value == 15                  # gültig: übernommen
    assert at.slider(key="form_slider").value == C.FORM_MAX           # außerhalb der Grenzen: auf die Grenze begrenzt


def test_the_assumption_of_independent_censoring_is_visible():
    at = _fresh_app()
    assert any("unabhängig vom Ereignis" in m.value for m in at.markdown)
