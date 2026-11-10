"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Zellen der Messungen decken jede Einstellung ab."""

import hzw_constants as C
import hzw_evaluation as E
import hzw_presets as P
import hzw_scenario as S


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_choices():
    for preset in C.PRESETS.values():
        assert preset["welt"] in S.WORLDS and preset["n"] in C.N_CHOICES and preset["fenster"] in C.WINDOW_CHOICES and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS.values()))
    assert first == {"welt": C.DEFAULT_WORLD, "n": C.DEFAULT_N, "fenster": C.DEFAULT_WINDOW, "seed": C.DEFAULT_SEED}
    for state_key, spec in P.SETTING_SPECS.items():
        if spec.choices is not None:
            assert spec.default in spec.choices, state_key
        else:
            assert spec.lo <= spec.default <= spec.hi, state_key


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)


def test_url_parameters_are_unique():
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_every_setting_of_the_sidebar_has_a_measured_cell():
    """Experiment 3 und 4 lesen eine Zelle der eingefrorenen Messung: jede Kombination aus Welt, n und Fenster muss dort stehen."""
    extra, probe = E.load_extrapolation(), E.load_probe()
    for world in S.WORLDS:
        for n in C.N_CHOICES:
            for window in C.WINDOW_CHOICES:
                assert E.cell_key(world, n, window) in extra["cells"]
                assert E.probe_key(n, window) in probe["cells"]
    assert extra["meta"]["runs"] == C.RUNS_EXTRAPOLATION and probe["meta"]["runs"] == C.RUNS_PROBE
