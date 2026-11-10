"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Der Test unten stellt
sicher, dass wirklich JEDER Regler der Seitenleiste dort steht: sonst bliebe das Preset trotz Änderung markiert. Die Regler der Experimente
(Form, Skala, Quantil) stehen im Hauptbereich und ebenfalls in SETTING_SPECS."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import hzw_constants as C
import hzw_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for bereich in (at.sidebar, at.main):
        for art in REGLER_ARTEN:
            try:
                return getattr(bereich, art)(key=key)
            except KeyError:
                continue
    raise AssertionError(f"Regler {key} nicht gefunden")


def _anderer_wert(regler, spec):
    """Ein Wert, der vom aktuellen verschieden und gültig ist (Auswahlregler: nächste Option; Zahlenregler: Rand)."""
    if spec.choices is not None:
        return next(c for c in spec.choices if c != regler.value)
    return spec.hi if regler.value != spec.hi else spec.lo


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_preset_setzt_auch_die_experiment_regler_auf_den_standard_zurueck():
    at = _app()
    at.slider(key="form_slider").set_value(2.5).run()
    assert "primary" not in _typen(at).values()
    name = next(iter(C.PRESETS))
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert at.slider(key="form_slider").value == C.DEFAULT_FORM and _typen(at)[name] == "primary"


def test_jeder_regler_der_seitenleiste_steht_in_setting_specs():
    at = _app()
    schluessel = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    fehlt = schluessel - set(P.SETTING_SPECS)
    assert not fehlt, f"Regler {sorted(fehlt)} fehlen in SETTING_SPECS: ihre Änderung hebt das aktive Preset nicht auf"


def test_jeder_regler_in_setting_specs_existiert_in_der_app():
    at = _app()
    for state_key in P.SETTING_SPECS:
        _regler(at, state_key)


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    spec = P.SETTING_SPECS[state_key]
    regler = _regler(at, state_key)
    regler.set_value(_anderer_wert(regler, spec)).run()
    assert not at.exception
    # Die Markierung folgt genau dem Zustand: hervorgehoben ist nur ein Preset, dessen Werte ALLE Regler (auch die der Experimente) entsprechen.
    # Ein Auswahlwert kann ein anderes Preset treffen (n = 100 gehört zu „Wenig Daten“); dann ist genau dieses markiert, sonst keines.
    key_of = {state_key: key for key, state_key in P.PRESET_KEYS.items()}
    zustand = {k: at.session_state[k] for k in P.SETTING_SPECS}
    erwartet = [name for name, preset in C.PRESETS.items() if all(zustand[k] == preset.get(key_of.get(k), P.SETTING_SPECS[k].default) for k in P.SETTING_SPECS)]
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == erwartet
    if state_key not in P.PRESET_KEYS.values():
        assert erwartet == []  # die Experiment-Regler gehören zu keinem Preset: ihre Änderung hebt immer auf
    # erneutes Anklicken markiert wieder
    name = next(iter(C.PRESETS))
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert _typen(at)[name] == "primary"
