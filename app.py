"""Hazard und Weibull: Form und Hochrechnen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Vier Experimente zur Abholrate (Hazard) von Containern: ihre Gestalt, die Weibull-Anpassung mit Zensierung, das Hochrechnen über das Beobachtungsfenster
und die Grenze des Weibull-Plots als Probe der Form. Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import hzw_constants as C
import hzw_evaluation as E
import hzw_model as M
import hzw_scenario as S
from hzw_layout import dunkles_design, kopf_css
from hzw_presets import (
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from hzw_visualization import build_extrapolation, build_fit, build_hazard, build_plot, build_survival

st.set_page_config(page_title="Hazard und Weibull – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _extrapolation():
    return E.load_extrapolation()


@st.cache_data(show_spinner=False)
def _probe():
    return E.load_probe()


@st.cache_data(show_spinner=False)
def _fit(world_key, n, window, seed):
    return E.fit_sample(S.simulate(S.WORLDS[world_key], n, window, seed))


def _table(rows):
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in hzw_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📈 Hazard und Weibull: Form und Hochrechnen")
st.markdown(
    """
Ein Container steht im Hafen, bis er abgeholt wird. Die **Abholrate (Hazard)** sagt, wie wahrscheinlich die Abholung im nächsten Augenblick ist, wenn er schon so lange da ist.
Sie kann mit dem Alter **fallen, gleich bleiben oder steigen**. Die **Weibull-Verteilung** legt diesen Verlauf mit einer Zahl fest (der Form *k*) und erlaubt, über das
Beobachtungsfenster hinauszurechnen, wo der Kaplan–Meier-Schätzer endet. Das Hochrechnen ist aber nur verlässlich, wenn die Form stimmt; bei falscher Form kann ein einzelnes Quantil zufällig nahe liegen, ein anderes weit daneben, und die Daten allein sagen das bei starker Zensierung kaum.

**Annahme in allen vier Experimenten:** Wer am Ende des Fensters noch nicht abgeholt ist, ist *zensiert*, und die Zensierung gilt als **unabhängig vom Ereignis**
(hier folgt das aus der Konstruktion: das Fensterende hängt nur von der Ankunftszeit ab, nicht von der Verweilzeit).

Weiter unten: die Experimente **🔬 1 bis 4**, **🚧 Wo die Annahmen enden**, **⚠️ Annahmen, Grenzen und bekannte Probleme** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Hazard und Weibull", expanded=True):
    st.markdown(
        """
1. **Hazard:** *h(t)* ist die Rate, mit der Container im Alter *t* abgeholt werden, gegeben dass sie bis dahin noch stehen. Ob sie mit dem Alter fällt, gleich bleibt oder steigt, ist eine Eigenschaft der Verweilzeit.
2. **Weibull:** *S(t) = exp(−(t/λ)^k)*. Die Form *k* < 1 gibt eine fallende, *k* = 1 eine konstante (Exponentialverteilung, ohne Gedächtnis) und *k* > 1 eine steigende Abholrate; die Skala *λ* streckt die Zeitachse.
3. **Anpassen mit Zensierung:** Die Maximum-Likelihood-Schätzung bewertet jeden abgeholten Container mit seiner Dichte und jeden zensierten nur mit der Wahrscheinlichkeit, bis zu seiner Beobachtungszeit noch dagewesen zu sein.
4. **Hochrechnen:** Aus der angepassten Kurve folgt die Zeit, bis ein Anteil *q* der Container abgeholt ist, auch hinter dem Ende der Daten. Das ist eine Annahme der Form, kein Messwert.
5. **Warum das nur halb funktioniert:** Zwei verschiedene Formen können im beobachteten Bereich ähnlich aussehen (in der Hauptlage liegt die Weibull-Kurve im Mittel rund 0,06 von Kaplan–Meier entfernt, nach einer Prüfrechnung) und hinter dem Fenster auseinanderlaufen. Die Probe der Form (der Weibull-Plot) erkennt das nur, wenn viel beobachtet ist.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Daten für die Experimente 2 bis 4**")
    welt_key = st.selectbox(
        C.LABEL_WELT, list(S.WORLDS), format_func=lambda k: S.WORLDS[k].label, key="welt_select",
        help="Aus dieser Verteilung stammen die simulierten Verweilzeiten. Die Anpassung kennt sie nicht: sie rechnet immer mit Weibull bzw. Exponential.",
    )
    n = st.radio("Container n", C.N_CHOICES, horizontal=True, key="n_radio", help="Zahl der beobachteten Container. Die vorgerechneten Messungen der Experimente 3 und 4 gibt es für diese Werte.")
    fenster = st.radio(
        "Beobachtungsfenster", C.WINDOW_CHOICES, horizontal=True, format_func=lambda w: f"{w} Tage", key="fenster_radio",
        help="Container kommen gleichverteilt im Fenster an; wer am Fensterende noch steht, ist zensiert. Je kürzer das Fenster, desto mehr Zensierung.",
    )
    seed = st.number_input("Zufalls-Seed der Stichprobe", 0, C.SEED_MAX, key="seed_input", step=1, help="Gilt für die Einzelstichprobe der Experimente 2 und 4. Die Messung in Experiment 3 läuft über feste Seeds.")
    st.button("🎲 Neue Stichprobe ziehen", width="stretch", on_click=randomize_seed)

world = S.WORLDS[welt_key]
n, fenster, seed = int(n), int(fenster), int(seed)

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wie die Abholrate vom Alter abhängt")
st.caption("Regler für Form und Skala der Weibull-Verteilung. Die graue Linie ist zum Vergleich die Abholrate der in der Seitenleiste gewählten Welt.")
c1, c2 = st.columns(2)
form = c1.slider("Form k", C.FORM_MIN, C.FORM_MAX, key="form_slider", step=0.1, help="k < 1: Abholrate fällt mit dem Alter; k = 1: konstant; k > 1: steigt.")
skala = c2.slider("Skala λ (Tage)", C.SKALA_MIN, C.SKALA_MAX, key="skala_slider", step=0.5, help="Streckt die Zeitachse: bei gleichem k verdoppelt die doppelte Skala jede Zeit (Median, Mittel, Quantile).")
form, skala = float(form), float(skala)
t_h, h_w, s_w = E.hazard_curve(form, skala)
tt = t_h
h_truth = world.hazard(tt)
left, right = st.columns(2)
left.plotly_chart(build_hazard(t_h, h_w, tt, h_truth), width="stretch", key="hazard_chart")
right.plotly_chart(build_survival(t_h, s_w), width="stretch", key="survival_chart")
st.caption(f"Die Hazard-Achse endet bei {E.fmt(C.HAZARD_AXIS_CAP, 1)} je Tag; was darüber liegt, läuft aus dem Bild. Links die Abholrate, rechts der Anteil, der noch da ist.")
kind, text = E.verdict_shape(form, skala)
getattr(st, kind)(text)
if world.family == "lognormal":
    peak = E.lognormal_hazard_peak(world.median, world.shape)
    st.markdown(
        f"**Zum Vergleich die Wahrheit der Welt:** Ihre Abholrate steigt bis etwa Tag {E.fmt(peak, 1)} und fällt danach. Die Weibull-Abholrate *h(t) ∝ t^(k−1)* ist dagegen für jedes *k* monoton: "
        "keine Weibull-Kurve hat diese Gestalt, die Weibull-Form ist für diese Welt also nur eine Näherung."
    )
st.markdown(f"**Gedächtnis:** Wahrscheinlichkeit, noch {E.fmt(C.EXTRA_DAYS, 0)} Tage zu bleiben, wenn der Container schon *a* Tage da ist (mit den Reglern oben):")
stay = E.stay_table(form, skala)
_table([{"Alter a (Tage)": E.fmt(r["age"], 0), f"noch {E.fmt(C.EXTRA_DAYS, 0)} Tage": E.pct(100 * r["p"])} for r in stay])
if abs(form - 1.0) < 1e-9:
    st.caption("Bei k = 1 steht in jeder Zeile derselbe Wert: die Exponentialverteilung hat kein Gedächtnis.")

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Anpassen mit Zensierung")
view = _fit(welt_key, n, fenster, seed)
sample = S.simulate(world, n, fenster, seed)
st.caption(
    f"Eine Stichprobe aus der Welt „{world.label}“: {n} Container, Fenster {fenster} Tage, Zufalls-Seed {seed}. "
    "Die Kaplan–Meier-Kurve ist der Maßstab im beobachteten Bereich; die Anpassungen setzen sie hinter dem Ende der Daten fort."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Abgeholt (Ereignisse)", f"{sample.n_events} von {sample.n}")
m2.metric("Zensiert", E.pct(100 * sample.censored_share, 0))
if view.weibull["ok"]:
    m3.metric("Weibull: Form k̂", E.fmt(view.weibull["k"]), delta=E.shape_kind(view.weibull["k"]), delta_color="off")
    m4.metric("Weibull: Skala λ̂ (Tage)", E.fmt(view.weibull["lam"]))
else:
    m3.metric("Weibull: Form k̂", "kein Schätzer")
    m4.metric("Weibull: Skala λ̂ (Tage)", "kein Schätzer")
st.plotly_chart(build_fit(view, world), width="stretch", key="fit_chart")
kind, text = E.verdict_fit(view, fenster)
getattr(st, kind)(text)
st.markdown("**Quantile dieser einen Stichprobe** (Zeit in Tagen, bis der Anteil abgeholt ist):")
_table(E.quantile_rows(view, world))
st.caption("„nicht erreicht“: die Kaplan–Meier-Kurve fällt innerhalb der Daten nicht bis zu diesem Anteil, sie endet vorher. Eine einzelne Stichprobe ist keine Messung: Experiment 3 wiederholt das über feste Seeds.")

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Hochrechnen gegen die Wahrheit")
extra = _extrapolation()
cell = extra["cells"][E.cell_key(welt_key, n, fenster)]
st.caption(
    f"Vorgerechnete Messung ({cell['runs']} Läufe mit festen Seeds, `tools/measure_hochrechnen.py`) für die Welt, n und das Fenster der Seitenleiste. "
    "Der Zufalls-Seed der Seitenleiste wirkt hier nicht."
)
q = st.select_slider("Anteil abgeholt (Quantil)", C.QUANTILE_LEVELS, format_func=lambda v: f"{round(v * 100)} %", key="level_select",
                     help="Gesucht ist die Zeit, bis dieser Anteil der Container abgeholt ist. Je größer der Anteil, desto weiter liegt sie hinter dem Fenster.")
st.plotly_chart(build_extrapolation(cell, float(q)), width="stretch", key="extra_chart")
kind, text = E.verdict_extrapolation(cell, float(q), world.label)
getattr(st, kind)(text)
_table(E.extrapolation_rows(cell, float(q)))
st.caption(f"Mittel über die Läufe, in denen das Verfahren einen Schätzer liefert; Balken: ±1 Standardabweichung der Läufe. Die Wahrheit ist das Quantil der gewählten Welt. {cell['weibull_failed']} von {cell['runs']} Weibull-Anpassungen ohne Schätzer.")

st.markdown("---")

# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Die Probe der Form und ihre Grenze")
probe = _probe()
pcell = probe["cells"][E.probe_key(n, fenster)]
thr = pcell["threshold"]
st.markdown(
    "Der **Weibull-Plot** stellt log(−log Ŝ) über log t dar: Bei einer Weibull-Verteilung liegen die Punkte auf einer Geraden mit der Steigung *k*. Das Gütemaß ist das Bestimmtheitsmaß **R²** der Geraden. "
    f"Die **Schwelle** ist das 5-%-Quantil von R² bei echter Weibull-Wahrheit (Form {E.fmt(C.PROBE_REFERENCE_SHAPE, 1)}, Median {E.fmt(C.MEDIAN, 0)} Tage) im selben Design, simuliert mit {pcell['runs']} Läufen; "
    "liegt R² darunter, schlägt die Probe an. Links die Stichprobe der Seitenleiste, rechts zum Vergleich eine Stichprobe aus der anderen Familie (Zufalls-Seed + 1)."
)
other_key = "lognormal_09" if world.family == "weibull" else "weibull_13"
other = S.WORLDS[other_key]
res_a = E.probe_for_sample(sample, thr)
res_b = E.probe_for_sample(S.simulate(other, n, fenster, seed + 1), thr)
pa, pb = st.columns(2)
pa.plotly_chart(build_plot(res_a, f"{world.label}: R² = {E.fmt(res_a['r2'], 3)}"), width="stretch", key="plot_a")
pb.plotly_chart(build_plot(res_b, f"{other.label}: R² = {E.fmt(res_b['r2'], 3)}"), width="stretch", key="plot_b")
kind, text = E.verdict_probe(res_a, thr, pcell)
getattr(st, kind)(text)
st.markdown("**Trennschärfe aller Designs** (Anteil der Läufe mit R² unter der Schwelle; die Weibull-Spalten zeigen Fehlalarme):")
_table(E.probe_rows(probe, (n, fenster)))
st.caption(
    f"{probe['meta']['runs']} Läufe je Welt und Design (`tools/measure_probe.py`). Schwelle und Trennschärfe nutzen verschiedene Seeds. "
    "Die Aussage der Probe gilt nur dort, wo ihre Trennschärfe hoch ist: Schlägt sie bei starker Zensierung nicht an, sagt das über die Form wenig."
)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

lead_cell = extra["cells"][E.cell_key("lognormal_09", 300, 8)]
lead_q = str(C.DEFAULT_LEVEL)
lw, le, lk = lead_cell["weibull"][lead_q], lead_cell["exponential"][lead_q], lead_cell["km"][lead_q]
peak09 = E.lognormal_hazard_peak(S.WORLDS["lognormal_09"].median, S.WORLDS["lognormal_09"].shape)
p300_8 = probe["cells"][E.probe_key(300, 8)]["worlds"]["lognormal_09"]["detected_pct"]
p300_30 = probe["cells"][E.probe_key(300, 30)]["worlds"]["lognormal_09"]["detected_pct"]
p100_8 = probe["cells"][E.probe_key(100, 8)]["worlds"]["lognormal_09"]["detected_pct"]
thr_shapes = " / ".join(E.fmt(probe["cells"][E.probe_key(300, 8)]["threshold_by_shape"][k], 3) for k in ("0.7", "1.3", "2.5"))

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wo gemessen |
|---|---|---|
| **Zensierung unabhängig vom Ereignis** | Beide Schätzer, Kaplan–Meier und Weibull-ML, rechnen unter dieser Annahme. Fällt ein Container kurz vor der Abholung aus der Beobachtung, sind sie verzerrt, ohne dass die Daten es zeigen. Diese Demo verletzt die Annahme nirgends. | nicht gemessen |
| **Die Verweilzeit ist Weibull-verteilt** (monotone Abholrate) | Lognormale Wahrheit (Median 4 Tage, σ = 0,9; Abholrate steigt bis etwa Tag {E.fmt(peak09, 1)}, dann fällt sie): Beim Quantil 0,9 liegt die Weibull-Anpassung {E.signed_pct(lw['bias_pct'])} daneben, die Exponentialanpassung {E.signed_pct(le['bias_pct'])} (n = 300, Fenster 8 Tage, {lead_cell['runs']} Läufe). Die Richtung hängt an der Welt, es ist keine allgemeine Regel. | Experiment 3 |
| **Hochrechnen hinter das Fenster** | Das Quantil 0,9 liegt hinter dem Ende der Daten: Kaplan–Meier erreicht es in dieser Lage nur in {E.fmt(lk['defined_pct'], 0)} % der Läufe. Was dahinter steht, kommt aus der Form. | Experiment 3 |
| **Der Weibull-Plot erkennt eine falsche Form** | Nur bei wenig Zensierung und vielen Daten: bei Fenster 30 Tage und n = 300 wird die lognormale Wahrheit (σ = 0,9) in {E.fmt(p300_30, 1)} % der Läufe erkannt, bei Fenster 8 Tage und n = 300 in {E.fmt(p300_8, 1)} %, bei n = 100 in {E.fmt(p100_8, 1)} %. | Experiment 4 |
| **Schwelle der Probe** (Weibull-Wahrheit mit Form 1,3) | Eine andere Form der Weibull-Wahrheit verschiebt die Schwelle: bei n = 300 und Fenster 8 Tage liegt sie für die Form 0,7 / 1,3 / 2,5 bei {thr_shapes}. Die Trennschärfe ist nur mit der Schwelle für Form 1,3 gemessen. | Messdatei `data/messung_probe.json` |
| **Nur diese Welten** | Gemessen sind lognormale und Weibull-Wahrheiten mit Median 4 Tage. Badewanne (fallend, dann steigend), Mischverteilungen und andere Zensierarten sind nicht gemessen. | nicht gemessen |
"""
)

st.markdown("---")

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen**
- Die Verweilzeiten sind simuliert (Median 4 Tage in jeder Welt, kontinuierlich, ohne Rundung auf Tage); die Ankunft ist gleichverteilt im Fenster.
- Die Zensierung ist unabhängig vom Ereignis. Es gibt keine Merkmale der Container (kein Regressionsmodell).
- Alle Zufallszahlen der Stichproben kommen aus dem Generator SplitMix64 mit festem Seed: gleiche Einstellungen geben dieselben Zahlen (auf anderen Plattformen nicht geprüft); nur der Knopf „Neue Stichprobe ziehen“ wählt den Zufalls-Seed mit Pythons random.

**Grenzen**
- Nur Weibull und Exponential als Anpassungen; die Lognormal-Verteilung ist die Wahrheit der Messreihe, keine Modellwahl.
- R² des Weibull-Plots ist ein einfaches Maß, kein Anpassungstest mit eigenem Niveau. Die Schwelle ist ein simuliertes 5-%-Quantil bei Weibull-Wahrheit mit Form 1,3.
- Experimente 3 und 4 zeigen vorgerechnete Messungen für die gezeigten Zellen (Welt, n, Fenster); n und Fenster sind deshalb auf diese Werte begrenzt.
- Eine Quelle aus der Literatur ist nicht zitiert: die Formeln sind gegen unabhängige Rechnungen geprüft (Rastersuche, geschlossene Formeln, Bruchrechnung), nicht gegen ein Lehrbuch.

**Bekannte Probleme und Unsicherheiten**
- Die Anteile in Experiment 4 haben Zufallsfehler: mit weiteren Seed-Sätzen schwanken sie um einige Prozentpunkte (die Messdatei enthält die Gegenproben).
- Die App ist mit dem AppTest-Rauchtest und einem Start im Browser geprüft, nicht mit einem Mausklick-Durchlauf jeder Kombination.
        """
    )

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** *Gegeben:* Beobachtungszeiten $t_i$ und Ereignisanzeiger $\delta_i$ ($1$ = abgeholt, $0$ = zensiert). *Gesucht:* Form $k$ und Skala $\lambda$ der Weibull-Verteilung bzw. die Rate $\rho$ der Exponentialverteilung und daraus die Zeit $t_q$, bis der Anteil $q$ abgeholt ist.

| Symbol | Bedeutung |
|---|---|
| $t_i$, $\delta_i$ | Beobachtungszeit des Containers $i$ (Abholung oder Fensterende) und Anzeiger, ob er abgeholt wurde |
| $d = \sum_i \delta_i$ | Zahl der abgeholten Container (Ereignisse) |
| $S(t)$, $h(t)$ | Anteil, der nach $t$ Tagen noch da ist, und Abholrate im Alter $t$ |
| $k$, $\lambda$ | Form und Skala der Weibull-Verteilung |
| $\rho$ | Rate der Exponentialverteilung (je Tag); gleich $1/\lambda$ bei $k = 1$ |
| $q$, $t_q$ | Anteil abgeholt und die Zeit, zu der er erreicht ist |
| $\hat S$ | Kaplan–Meier-Schätzer |
| $R^2$ | Bestimmtheitsmaß der Geraden im Weibull-Plot |

**Abholrate und Überleben.**

$$h(t) = -\frac{d}{dt}\ln S(t)$$

**Weibull.**

$$S(t) = \exp\!\bigl(-(t/\lambda)^k\bigr), \qquad h(t) = \frac{k}{\lambda}\Bigl(\frac{t}{\lambda}\Bigr)^{k-1}$$

Für $k < 1$ fällt $h$, für $k = 1$ ist $h$ konstant, für $k > 1$ steigt $h$.

**Zensierte Likelihood.** Ein abgeholter Container geht mit seiner Dichte $h(t_i)\,S(t_i)$ ein, ein zensierter nur mit $S(t_i)$:

$$\ell(k,\lambda) = \sum_i \delta_i \ln h(t_i) - \sum_i \Bigl(\frac{t_i}{\lambda}\Bigr)^k = d\,(\ln k - k\ln\lambda) + (k-1)\sum_i \delta_i \ln t_i - \lambda^{-k}\sum_i t_i^k$$

Die Demo maximiert $\ell$ über $(\ln k, \ln\lambda)$ mit BFGS (analytischer Gradient, vier Startwerte, Konvergenzprüfung). Ohne mindestens zwei verschiedene Ereigniszeiten gibt es keinen Schätzer, die App meldet das.

**Exponential (k = 1).**

$$\hat\rho = \frac{d}{\sum_i t_i}$$

**Quantile.**

$$t_q = \lambda\bigl(-\ln(1-q)\bigr)^{1/k}, \qquad t_q = \frac{-\ln(1-q)}{\rho}$$

**Weibull-Plot.** Aus $S(t) = \exp(-(t/\lambda)^k)$ folgt

$$\ln\bigl(-\ln S(t)\bigr) = k\ln t - k\ln\lambda$$

also eine Gerade in $\ln t$ mit Steigung $k$. Die Probe passt eine Gerade $\hat y$ an die Punkte $(\ln t, \ln(-\ln\hat S(t)))$ an Ereigniszeiten mit $0 < \hat S < 1$ an (mindestens fünf Punkte):

$$R^2 = 1 - \frac{\operatorname{Var}(y - \hat y)}{\operatorname{Var}(y)}$$

**Wahrheit der Messreihe (lognormal, Median $m$, Streuung $\sigma$).**

$$S(t) = 1 - \Phi\!\Bigl(\frac{\ln t - \ln m}{\sigma}\Bigr)$$

Dabei ist $\Phi$ die Verteilungsfunktion der Standardnormalverteilung; die Abholrate $h = f/S$ steigt zuerst und fällt später.

**Handbeispiel.** Zeiten 2, 3+, 4, 5, 5+, 8 (+ = zensiert): $d = 4$, $\sum t_i = 27$, also $\hat\rho = 4/27 \approx 0{,}148$, mittlere Verweilzeit $1/\hat\rho = 6{,}75$ Tage, Median $\ln 2/\hat\rho \approx 4{,}68$ Tage.

**Literatur.** Keine Quelle zitiert (siehe „Annahmen, Grenzen und bekannte Probleme“).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Überlebenszeit: zensierte Zeiten lesen](https://sebastianhanisch.net/konzepte-ueberlebenszeit.html)."
)
