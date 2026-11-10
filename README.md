# Hazard und Weibull: Form und Hochrechnen – Streamlit-Demo

Dritte Demo der Linie **Überlebenszeit** (zensierte Zeiten bis zum Ereignis) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Sie setzt auf dem Kaplan–Meier-Schätzer von [kaplan-meier-demo](https://github.com/sebastian-hanisch/kaplan-meier-demo) auf: Kaplan–Meier endet am Beobachtungsfenster, und wer weiter hinausrechnen will,
braucht eine Form. Die Folgedemo [cox-regression-demo](https://github.com/sebastian-hanisch/cox-regression-demo) baut Merkmale ein, ohne eine Grundform festzulegen. Vehikel ist die Containerverweilzeit wie in
[verweilzeit-demo](https://github.com/sebastian-hanisch/verweilzeit-demo) (dort ohne Zensierung): Median 4 Tage, in der Hauptwelt lognormal mit σ = 0,9.

**Annahme in der ganzen Demo:** Die Zensierung ist unabhängig vom Ereignis (das Fensterende hängt nur von der Ankunftszeit ab). Sie wird hier nirgends geprüft oder verletzt.

## Warum dieses Problem

Die **Abholrate (Hazard)** eines Containers kann mit dem Alter fallen, gleich bleiben oder steigen. Die **Weibull-Verteilung** legt diesen Verlauf mit einer Zahl fest, der Form *k*, und erlaubt, über das Beobachtungsfenster hinauszurechnen,
wo der Kaplan–Meier-Schätzer endet. **Kernfrage:** Wie weit reicht dieses Hochrechnen? Es ist nur verlässlich, wenn die Form stimmt; bei falscher Form kann ein einzelnes Quantil zufällig nahe liegen, ein anderes weit daneben, und die Daten allein sagen das bei starker Zensierung kaum: In der Lage der Hauptmessung
(lognormale Wahrheit, 300 Container, Fenster 8 Tage, 56 % zensiert) liegt die Weibull-Anpassung beim Quantil 0,9 um 27 % unter der Wahrheit, die Exponentialanpassung um 16 % darüber, und der Weibull-Plot erkennt die falsche Form in 21,6 % der Läufe.

## Modell

- **Welten:** vier simulierte Verweilzeit-Verteilungen mit Median 4 Tage: lognormal mit σ = 0,9 (Hauptwelt, Quantil 0,9 gleich 12,68 Tage) und σ = 0,5, Weibull mit Form 1,3 und Form 0,7. Lognormal ist die Wahrheit der Messreihe, keine Modellwahl der App.
- **Beobachtung:** Ankunft gleichverteilt im Fenster *W* (8, 15 oder 30 Tage); Zeit = min(Verweilzeit, *W* − Ankunft), Ereignis = abgeholt. Alle Zufallszahlen der Stichproben kommen aus `SplitMix64` (feste Seeds, auf anderen Plattformen nicht geprüft), nicht aus numpy-Verteilungen; nur der Knopf „Neue Stichprobe ziehen“ wählt den Zufalls-Seed mit Pythons `random`.
- **Anpassungen:** Weibull *S(t) = exp(−(t/λ)^k)* per Maximum Likelihood mit Rechtszensierung (BFGS mit analytischem Gradienten, vier Startwerte, Konvergenzprüfung) und Exponential mit Rate *ρ = d / Σt*. Formeln: Expander `📐 Mathematische Formulierung` der App.
- **Probe der Form:** Weibull-Plot, log(−log Ŝ) gegen log *t* an den Kaplan–Meier-Punkten; Gütemaß R² der Ausgleichsgeraden; Schwelle = 5-%-Quantil von R² bei Weibull-Wahrheit mit Form 1,3 (gleiches Design).

## Methodik

Vier Experimente in der App. **(1) Gestalt:** Regler für Form und Skala, Abholrate und Überlebenskurve nebeneinander, dazu die Abholrate der gewählten Welt und die Wahrscheinlichkeit, noch 2 Tage zu bleiben, je Alter.
**(2) Anpassen mit Zensierung:** eine Stichprobe (Welt, *n*, Fenster, Zufalls-Seed der Seitenleiste): Kaplan–Meier mit Greenwood-Band, Weibull- und Exponentialkurve, Wahrheit, die Quantile dieser einen Stichprobe.
**(3) Hochrechnen:** vorgerechnete Messung über feste Seeds (200 Läufe je Zelle aus Welt, *n* und Fenster): Quantile aus Weibull-ML, Exponential und Kaplan–Meier gegen die Wahrheit.
**(4) Die Probe und ihre Grenze:** Weibull-Plot der Stichprobe neben einer Stichprobe der anderen Familie, daneben die gemessene Trennschärfe aller sechs Designs (1000 Läufe je Welt).
Die Messungen laufen mit `tools/measure_hochrechnen.py` und `tools/measure_probe.py` und stehen eingefroren in `data/`; die App liest nur diese Dateien. Jede Zahl der App-Texte, die ein Befund ist, wird aus ihnen gebildet.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: eigene Messung mit `SplitMix64`; „Zelle“ meint Welt | *n* | Fenster in `data/messung_hochrechnen.json` bzw. *n* | Fenster in `data/messung_probe.json`. Jede Zahl steht in `tests/test_claims.py` (die Tabelle am Ende dieses Abschnitts nennt den Test).

| Frage | Befund (Voraussetzung im selben Satz) |
|---|---|
| Hochrechnen bei falscher Form: lognormal, σ = 0,9, *n* = 300, Fenster 8 Tage (56 % zensiert), 200 Läufe, Quantil 0,9 (Wahrheit 12,68 Tage) | Weibull-ML 9,21 ± 0,06 (−27 %), Exponential 14,65 ± 0,09 (+16 %); Kaplan–Meier erreicht das Niveau nur in 5 % der Läufe (10 von 200, dort im Mittel 7,79). Beide Formen sind falsch, in entgegengesetzter Richtung. |
| Dieselbe Welt, Fenster 30 Tage (20 % zensiert) | Weibull-ML 11,93 (−6 %), Exponential 13,56 (+7 %); Kaplan–Meier ist in allen 200 Läufen definiert und liegt bei 12,70 (0 %). Mit weniger Zensierung sind beide Abweichungen kleiner (Quantil 0,9, auch bei *n* = 100). |
| Innerhalb des Fensters: Median (Quantil 0,5), Fenster 8 Tage, *n* = 300 | Kaplan–Meier 4,02 ± 0,03 (0 %), Weibull-ML 4,21 (+5 %), Exponential 4,41 (+10 %). |
| Die Weibull-Anpassung ist auch im beobachteten Bereich verzerrt: Median, Fenster 30 Tage, *n* = 300, 200 Läufe | Weibull-ML 4,53 (+13 %), Exponential 4,08 (+2 %), Kaplan–Meier 4,02 (+1 %); das ist dieselbe Abweichung von +0,5 % wie bei Fenster 8, nur anders gerundet. |
| Die Richtung hängt an der Welt: Quantil 0,9, Fenster 8 Tage, *n* = 300 | Exponential: lognormal σ = 0,9 +16 %, lognormal σ = 0,5 +96 %, Weibull Form 0,7 −43 %. Weibull-ML: −27 %, −11 % und +1 %. Das Vorzeichen der Exponential-Abweichung wechselt zwischen den Welten: eine feste Richtung „Form falsch, Quantil zu klein“ oder „zu groß“ gibt es in diesen vier Welten nicht. |
| Form richtig: Weibull-Wahrheit mit Form 1,3, Quantil 0,9 (Wahrheit 10,07), Fenster 8 Tage, *n* = 300 | Weibull-ML 10,09 (0 %), Exponential 13,64 (+35 %), Kaplan–Meier nur in 5 % der Läufe definiert. Der mittlere Schätzer k̂ liegt bei 1,31. |
| Weniger Daten: lognormal σ = 0,9, Fenster 8 Tage, *n* = 100, Quantil 0,9 | Weibull-ML 9,18 (−28 %), Exponential 14,50 (+14 %), Kaplan–Meier in 6,5 % der Läufe (13 von 200) definiert. |
| Trennschärfe des Weibull-Plots, lognormale Wahrheit σ = 0,9 (σ = 0,5), 1000 Läufe je Welt | Fenster 30 Tage, *n* = 300 (19 % zensiert): 89,5 % (95,4 %). Fenster 8 Tage, *n* = 300 (56 % zensiert): 21,6 % (33,2 %). Fenster 8 Tage, *n* = 100: 2,6 % (3,8 %). 80 % erreicht die Trennschärfe nur bei *n* = 300 (σ = 0,9: nur bei Fenster 30 Tage; σ = 0,5: bei Fenster 15 Tage rund 75 bis 81 % (Hauptlauf 80,9 %, mit den Gegenproben 75,2 bis 80,9 %), bei Fenster 30 Tage über 94 %); bei *n* = 100 bleibt sie in allen drei Fenstern unter 25 %. Alle sechs Designs: App, Experiment 4. |
| Gegenprobe der Trennschärfe mit drei weiteren Seed-Paaren (Schwelle und Test neu gezogen), σ = 0,9 | Fenster 30, *n* = 300: 87,4 bis 89,6 %. Fenster 8, *n* = 300: 16,0 bis 20,7 %. Fenster 8, *n* = 100: 2,3 bis 3,5 %. |
| Fehlalarme: der Weibull-Plot schlägt bei Weibull-Wahrheit an | Form 1,3 (die Form der Schwelle, andere Seeds als die Schwelle): 4,8 bis 6,0 % über die sechs Designs; Form 0,7: 5,7 bis 8,0 %. |
| Schwelle hängt an der Form der Weibull-Wahrheit: *n* = 300, Fenster 8 Tage | 5-%-Quantil von R² bei Form 0,7 / 1,3 / 2,5: 0,955 / 0,959 / 0,962. |
| Handbeispiel (Zeiten 2, 3+, 4, 5, 5+, 8) | ρ̂ = 4/27 ≈ 0,148; Mittel 1/ρ̂ = 6,75 Tage; Median ln 2/ρ̂ = 4,68 Tage (Bruchrechnung, `tests/test_core.py`, `tests/test_oracle_hazard_weibull.py`). |

**Zahl → Test:** Die Tabelle steht in `tests/test_claims.py`: `test_readme_zahlen_stehen_in_der_messung` liest `data/messung_hochrechnen.json` und `data/messung_probe.json` und prüft, dass jede genannte Zahl in dieser README steht;
`test_messung_hochrechnen_ist_reproduzierbar` und `test_messung_probe_ist_reproduzierbar` rechnen je eine Zelle neu (alle Läufe, gleiche Seeds) und vergleichen mit der Messdatei (Toleranz 1e-6 relativ);
die Ordnung der Aussagen (Vorzeichen, „in allen Läufen“, Spannen, „80 % nur bei *n* = 300“) prüfen `test_aussagen_der_readme_gelten_ueber_alle_zellen`, `test_trennschaerfe_nur_bei_n_300_und_wenig_zensierung` und `test_gegenprobe_spannen`.
Exakt nachgerechnet ist nur das Handbeispiel (Brüche); alle übrigen Zahlen sind Zufallsergebnisse und stehen im Test als Messdatei-Vergleich, nicht als Handwert.

## Ehrliche Grenzen

- **Nur zwei Anpassungsfamilien und zwei Welt-Familien.** Gemessen sind Weibull und Exponential als Anpassung, lognormale und Weibull-Verteilungen als Wahrheit (Median 4 Tage). Badewanne (fallend, dann steigend), Mischverteilungen und andere Zensierarten sind nicht gemessen.
- **Die Weibull-Probe ist ein einfaches Maß.** R² der Geraden ist kein Anpassungstest mit eigenem Niveau. Die Schwelle ist ein simuliertes 5-%-Quantil bei Weibull-Wahrheit mit Form 1,3; die Trennschärfe ist nur mit dieser Schwelle gemessen. Bei starker Zensierung und kleinem *n* sagt ein Nicht-Anschlagen über die Form wenig (21,6 % und 2,6 % in den beiden Designs mit Fenster 8 Tage).
- **Mittel über definierte Läufe.** Wo Kaplan–Meier das Niveau nur in einem Teil der Läufe erreicht (5 % im Hauptfall), ist sein Mittel eine Auslese: gerade die Läufe, in denen die Kurve weit genug fällt. Der Wert 7,79 ist kein Schätzer der Wahrheit, sondern zeigt, dass die Kurve am Fenster endet.
- **Abweichungen sind in den gezeigten Zellen gemessen, nicht allgemein.** Die Richtung der Verzerrung hängt an der Welt (siehe Befunde); „Form falsch, Quantil falsch“ gilt nur mit den gemessenen Richtungen.
- **Zensierung unabhängig vom Ereignis.** Fällt ein Container kurz vor der Abholung aus der Beobachtung, sind Kaplan–Meier und Weibull-ML verzerrt; das wird hier nicht gemessen.
- **Nur gegen unabhängige Rechnungen geprüft.** Die Anpassung ist gegen eine Rastersuche, die Standardformeln der Weibull-Likelihood und scipy-Verteilungen geprüft (Abschnitt Tests); ein Fremdpaket für die Überlebensanalyse stand nicht zur Verfügung, und eine Quelle aus der Literatur ist nicht zitiert, weil keine im Volltext gelesen wurde.
- **Mausklick nicht getestet.** Die App ist mit dem AppTest-Rauchtest und einem Start im Browser geprüft, nicht mit einem Mausklick-Durchlauf jeder Kombination.

## Befunde und Korrekturen gegenüber dem Plan

- **Trennschärfe des Weibull-Plots höher als im Plan.** Die Vormessung (200 Läufe, numpy-Zufall, `out_uz_v20_bis_v23.txt`) nannte für lognormal σ = 0,9 (σ = 0,5): Fenster 30 / *n* = 300 80,0 % (91,5 %); Fenster 8 / *n* = 300 7,5 % (16,5 %); Fenster 8 / *n* = 100 0,5 % (2,0 %); Fenster 30 / *n* = 100 17,0 % (23,5 %).
  Mit `SplitMix64` und 1000 Läufen ergeben sich 89,5 % (95,4 %), 21,6 % (33,2 %), 2,6 % (3,8 %) und 18,5 % (22,6 %). Es gelten die neuen Zahlen. Die Richtung bleibt: trennscharf nur bei wenig Zensierung und vielen Daten.
  Die Gegenproben mit drei weiteren Seed-Paaren schwanken im Design Fenster 8 / *n* = 300 zwischen 16,0 und 20,7 %; die 7,5 % der Vormessung liegen deutlich darunter. Die Ursache ist nicht untersucht; möglich ist, dass die Schwelle (ein 5-%-Quantil) aus 200 Läufen unsicher war.
- **Hochrechnen (V7) bestätigt, mit kleinen Abweichungen.** Plan: Weibull 9,20 ± 0,05 (−27 %), Exponential 14,68 ± 0,08 (+16 %), Kaplan–Meier in 4 % der Läufe definiert. Hier: 9,21 ± 0,06 (−27 %), 14,65 ± 0,09 (+16 %), 5 % (10 von 200). Zufallsbedingt.
- **Erweitert gegenüber dem Plan:** eine Weibull-Wahrheit als Kontrolle („Form richtig“), eine zweite lognormale Welt (σ = 0,5) und eine fallende Weibull-Welt (Form 0,7), drei Fenster (8, 15, 30 Tage) und zwei Stichprobengrößen (100, 300); dazu die Fehlalarm- und Schwellen-Messungen.
  Sie stützen die Aussage von Z3 und Z5 des Plans: die Richtung der Verzerrung hängt an der Welt, und die Probe ist nur in ihrer gemessenen Lage trennscharf.
- **Symbol:** Der Plan nennt die Exponential-Rate λ̂ (= 4/27). Hier heißt sie ρ, damit λ nur die Weibull-Skala bleibt.
- **Optimierer:** wie im Plan BFGS mit analytischem Gradienten und mehreren Startwerten statt Nelder-Mead; ohne mindestens zwei verschiedene Ereigniszeiten meldet die Anpassung `ok = False` statt einer stillen Zahl (Randfälle der Orakelprüfung des Plans).
- **n und Fenster in der Seitenleiste** sind auf die gemessenen Werte begrenzt (100 / 300 und 8 / 15 / 30 Tage); die App rechnet Experiment 3 und 4 nicht live.

## Tests

Zusammensetzung (Gesamtzahl: siehe `pytest --collect-only -q tests`, aktuell **120 Tests**):
`test_core.py` (Handbeispiel mit Brüchen, Gleichstände, Greenwood, Eingabeprüfung, `SplitMix64`-Referenzwert), `test_model.py` (Verteilungen an Handwerten, Weibull-Plot, Szenario, Zensierungsanteil gegen die Formel),
`test_oracle_hazard_weibull.py` (Weibull-ML gegen Rastersuche über (*k*, *λ*), gegen die Standardformeln ohne Zensierung und per finiter Differenz im Gradienten; Exponential gegen die geschlossene Formel),
`test_oracle_verteilungen.py` (Weibull, Lognormal, Normalquantil, Mittel gegen scipy, numerische Ableitung und Quadratur; scipy nur in `requirements-dev.txt`),
`test_oracle_probe.py` (Punkte des Weibull-Plots per Bruchrechnung, R² per scipy und Korrelation),
`test_eigenschaften_hazard_weibull.py` (Zeitskalierung, Reihenfolge, Verdoppeln, Gedächtnislosigkeit, mehr Daten, gleicher Seed),
`test_claims.py` (README-Zahlen gegen die Messdateien, Neuberechnung je einer Zelle), `test_verdicts.py` (Schwellen und Zahlen der Verdikte), `test_edge_cases.py` (kein Ereignis, ein Ereignis, alle Zeiten gleich, Zeiten ≤ 0, Reglergrenzen),
`test_zweige.py` (jede Variante weicht von der Nachbarvariante ab; jedes Preset zeigt die erwartete Zeile in Experiment 3 und 4), `test_presets.py`, `test_layout.py`, `test_app.py` (AppTest-Rauchtest, jede Welt, jedes Fenster, jede Auswahl), `test_footer.py`, `test_runtime_imports.py`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: vier Experimente, Grenzen, Formeln |
| `hzw_core.py` | Kern, Kopie aus `ueberlebenszeit-planung/kern/uz_core.py` (Stand 2026-11-10): `SplitMix64`, Eingabeprüfung, Kaplan–Meier, Weibull- und Exponential-Anpassung |
| `hzw_model.py` | Weibull, Exponential, Lognormal in geschlossener Form, Weibull-Plot und R² |
| `hzw_scenario.py` | Welten, Beobachtungsfenster, Stichprobe aus einem Zufalls-Seed |
| `hzw_evaluation.py` | Anpassung einer Stichprobe, Messungen (Hochrechnen, Probe), Verdikte, Tabellen |
| `hzw_constants.py` | Reglergrenzen, feste Annahmen, Seeds, Farben, Presets |
| `hzw_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `hzw_layout.py` | Fester Kopf (Stylesheet) |
| `hzw_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `data/messung_hochrechnen.json`, `data/messung_probe.json` | eingefrorene Messungen (feste Seeds) |
| `tools/measure_hochrechnen.py`, `tools/measure_probe.py` | erzeugen die Messdateien |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit-Bibliotheken (ungepinnt); dev: pytest und scipy für die Gegenproben |
| `tests/` | siehe Abschnitt Tests |

## Bewusst nicht umgesetzt

- **Merkmale und Regression** (Cox): das ist die Folgedemo; hier gibt es keine Merkmale der Container.
- **Badewanne, Erneuerung, Instandhaltung** und Mischverteilungen: nicht Teil dieser Demo und nicht gemessen.
- **Lognormal als Anpassung:** die Lognormal-Verteilung ist nur die Wahrheit der Messreihe, nicht Teil der Modellwahl in der App.
- **Anpassungstests, Konfidenzintervalle der Weibull-Parameter:** nicht gebaut; R² mit simulierter Schwelle ist die einzige Probe.
- **Live-Rechnung der Messungen** (Experiment 3 und 4): die Zellen sind vorgerechnet, damit die App ohne Wartezeit antwortet.
- **PDF-Export:** nicht gebaut.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die CI läuft mit Python 3.12; die in Streamlit Cloud eingestellte Version ist beim Deploy zu nennen.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Messungen neu erzeugen (Wurzelordner): `python tools/measure_hochrechnen.py` und `python tools/measure_probe.py` (feste Seeds, gleiche Zahlen).

Gebaut mit Streamlit, Plotly, pandas und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
