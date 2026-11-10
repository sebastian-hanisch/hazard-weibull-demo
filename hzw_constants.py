"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

MEDIAN = 4.0                                   # Tage; Median der Verweilzeit in jeder Welt (wie verweilzeit-demo: lognormal, Median 4, σ = 0,9)

# Experiment 1: Gestalt der Weibull-Verteilung
FORM_MIN, FORM_MAX, DEFAULT_FORM = 0.5, 3.0, 1.5
SKALA_MIN, SKALA_MAX, DEFAULT_SKALA = 1.0, 12.0, 6.0   # Tage
T_PLOT_MAX = 20.0                              # Zeitachse der Kurven in Experiment 1 (Tage)
T_FIT_MAX = 30.0                               # Zeitachse in Experiment 2 (Tage): weit genug für das Quantil 0,95 der lognormalen Welt (17,6)
HAZARD_AXIS_CAP = 1.5                          # obere Grenze der Hazard-Achse (je Tag); darüber läuft die Kurve aus dem Bild
AGES = (0.0, 2.0, 4.0, 8.0)                    # Alter a der bedingten Verweilwahrscheinlichkeit
EXTRA_DAYS = 2.0                               # „noch r Tage bleiben“

# Szenario der Experimente 2 bis 4 (Seitenleiste)
N_CHOICES = (100, 300)
WINDOW_CHOICES = (8, 15, 30)                   # Beobachtungsfenster in Tagen
DEFAULT_N, DEFAULT_WINDOW, DEFAULT_WORLD = 300, 8, "lognormal_09"
SEED_MAX, DEFAULT_SEED = 999999, 35

# Experiment 3 und 4: vorgerechnete Messungen (tools/measure_*.py, feste Seeds)
QUANTILE_LEVELS = (0.5, 0.75, 0.9, 0.95)
DEFAULT_LEVEL = 0.9
RUNS_EXTRAPOLATION = 200                       # Läufe je Zelle der Messung „Hochrechnen“
RUNS_PROBE = 1000                              # Läufe je Zelle der Messung „Probe“
SEEDS_EXTRAPOLATION = 7000                     # Seed des Laufs r = Basis + r
SEEDS_PROBE_REFERENCE = 220000                 # Schwelle: Weibull-Wahrheit
SEEDS_PROBE_TEST = 225000                      # Trennschärfe: Welt unter Prüfung (andere Seeds als die Schwelle)
PROBE_STABILITY_SEEDS = ((320000, 325000), (420000, 425000), (520000, 525000))  # weitere Seed-Paare (Schwelle, Test) für die Gegenprobe
PROBE_REFERENCE_SHAPE = 1.3                    # Weibull-Wahrheit, deren 5-%-Quantil von R² die Schwelle bildet
PROBE_LEVEL = 0.05
POWER_HIGH, POWER_LOW = 80.0, 50.0             # Ampel des Verdikts der Probe in Prozent der Läufe
DATA_DIR = "data"
FILE_EXTRAPOLATION = "messung_hochrechnen.json"
FILE_PROBE = "messung_probe.json"

COLOR_WEIBULL = "#2F6B65"
COLOR_EXP = "#c0612b"
COLOR_KM = "#4c78a8"
COLOR_TRUTH = "#555555"
COLOR_BAND = "rgba(76,120,168,0.18)"

LABEL_WELT = "Welt der Daten (Wahrheit)"
PRESETS = {
    "Lage der Hochrechnung (lognormal, Fenster 8, n = 300)": {"welt": DEFAULT_WORLD, "n": DEFAULT_N, "fenster": DEFAULT_WINDOW, "seed": DEFAULT_SEED},
    "Viel Zeit, viele Daten (Fenster 30, n = 300)": {"welt": DEFAULT_WORLD, "n": 300, "fenster": 30, "seed": DEFAULT_SEED},
    "Wenig Daten (Fenster 8, n = 100)": {"welt": DEFAULT_WORLD, "n": 100, "fenster": 8, "seed": DEFAULT_SEED},
    "Weibull ist wahr (Form 1,3, Fenster 8)": {"welt": "weibull_13", "n": DEFAULT_N, "fenster": DEFAULT_WINDOW, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Lage der Hochrechnung (lognormal, Fenster 8, n = 300)": "Verweilzeit lognormal (Median 4 Tage, σ = 0,9), 300 Container, Beobachtungsfenster 8 Tage. Die Anpassung ist falsch, und das Fenster endet lange vor dem Tail.",
    "Viel Zeit, viele Daten (Fenster 30, n = 300)": "Gleiche Welt, aber 30 Tage Fenster: weniger Zensierung, der Kaplan–Meier-Schätzer reicht weit in den Tail.",
    "Wenig Daten (Fenster 8, n = 100)": "Gleiche Welt und gleiches Fenster, nur 100 Container: die Streuung der Schätzungen wächst.",
    "Weibull ist wahr (Form 1,3, Fenster 8)": "Die Daten stammen aus einer Weibull-Verteilung mit Form 1,3: die Weibull-Anpassung hat hier die richtige Form.",
}
