"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go

import hzw_constants as C
import hzw_core as K
import hzw_model as M


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _de(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


# ------------------------------------------------------------------------------------------------ Experiment 1


def build_hazard(t, h, truth_t=None, truth_h=None):
    """Abholrate über dem Alter; optional die Rate der lognormalen Wahrheit zum Vergleich (grau gestrichelt)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=h, mode="lines", line=dict(color=C.COLOR_WEIBULL, width=3), name="Weibull", hovertemplate="Alter %{x:.1f} Tage: %{y:.3f} je Tag<extra></extra>"))
    if truth_t is not None:
        fig.add_trace(go.Scatter(x=truth_t, y=truth_h, mode="lines", line=dict(color=C.COLOR_TRUTH, width=2, dash="dash"), name="Lognormal (Wahrheit der Messreihe)",
                                 hovertemplate="Alter %{x:.1f} Tage: %{y:.3f} je Tag<extra></extra>"))
    fig.update_xaxes(title_text="Alter des Containers (Tage)", range=[0, C.T_PLOT_MAX])
    top = float(np.max(h)) if truth_h is None else float(max(np.max(h), np.max(truth_h)))
    fig.update_yaxes(title_text="Abholrate h(t) je Tag", range=[0, min(1.1 * top, C.HAZARD_AXIS_CAP)])
    return _base(fig)


def build_survival(t, s):
    fig = go.Figure(go.Scatter(x=t, y=s, mode="lines", line=dict(color=C.COLOR_WEIBULL, width=3), name="Weibull", hovertemplate="Alter %{x:.1f} Tage: %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Alter des Containers (Tage)", range=[0, C.T_PLOT_MAX])
    fig.update_yaxes(title_text="Noch da: S(t)", range=[0, 1.02])
    return _base(fig)


# ------------------------------------------------------------------------------------------------ Experiment 2


def km_band(km: dict, level: float = 0.95):
    """Kaplan–Meier-Treppe mit punktweisem Greenwood-Band (log-log) an den Ereigniszeiten: (t, s, untere, obere); Grenzen nan, wo Ŝ ∈ {0, 1}."""
    lo, hi = [], []
    for x in km["t"]:
        _, a, b = K.greenwood_band(km, float(x), level)
        lo.append(a)
        hi.append(b)
    return km["t"], km["s"], np.array(lo), np.array(hi)


def build_fit(view, world, show_exponential: bool = True):
    """Kaplan–Meier mit Band, Weibull- und Exponentialkurve, Wahrheit; hinter dem Ende der Daten liegt der Bereich der Hochrechnung."""
    t_end = min(view.t_last, C.T_FIT_MAX)
    grid = np.linspace(0.0, C.T_FIT_MAX, 301)
    fig = go.Figure()
    fig.add_vrect(x0=t_end, x1=C.T_FIT_MAX, fillcolor="rgba(150,150,150,0.12)", line_width=0, layer="below")
    fig.add_annotation(x=(t_end + C.T_FIT_MAX) / 2, y=1.0, text="hinter den Daten: nur die Form rechnet", showarrow=False, yanchor="top", font=dict(size=11, color="#777777"))
    t, s, lo, hi = km_band(view.km)
    t0 = np.concatenate([[0.0], t])
    s0 = np.concatenate([[1.0], s])
    ok = ~np.isnan(lo)
    if ok.any():
        tt = t[ok]
        fig.add_trace(go.Scatter(x=np.concatenate([tt, tt[::-1]]), y=np.concatenate([hi[ok], lo[ok][::-1]]), fill="toself", fillcolor=C.COLOR_BAND, line=dict(width=0),
                                 hoverinfo="skip", name="95-%-Band (Greenwood)"))
    fig.add_trace(go.Scatter(x=np.append(t0, t_end), y=np.append(s0, s0[-1]), mode="lines", line=dict(color=C.COLOR_KM, width=2, shape="hv"), name="Kaplan–Meier",
                             hovertemplate="Tag %{x:.2f}: %{y:.3f}<extra></extra>"))
    if view.weibull["ok"]:
        fig.add_trace(go.Scatter(x=grid, y=M.weibull_survival(grid, view.weibull["k"], view.weibull["lam"]), mode="lines", line=dict(color=C.COLOR_WEIBULL, width=3),
                                 name=f"Weibull (k̂ = {_de(view.weibull['k'])})", hovertemplate="Tag %{x:.1f}: %{y:.3f}<extra></extra>"))
    if view.exponential["ok"] and show_exponential:
        fig.add_trace(go.Scatter(x=grid, y=np.exp(-view.exponential["rate"] * grid), mode="lines", line=dict(color=C.COLOR_EXP, width=2, dash="dash"), name="Exponential",
                                 hovertemplate="Tag %{x:.1f}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=grid[1:], y=world.survival(grid[1:]), mode="lines", line=dict(color=C.COLOR_TRUTH, width=2, dash="dot"), name="Wahrheit der Welt",
                             hovertemplate="Tag %{x:.1f}: %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Zeit seit Ankunft (Tage)", range=[0, C.T_FIT_MAX])
    fig.update_yaxes(title_text="Noch da: S(t)", range=[0, 1.02])
    return _base(fig, 420)


# ------------------------------------------------------------------------------------------------ Experiment 3


def build_extrapolation(cell: dict, q: float):
    """Mittel der Quantilschätzung je Verfahren mit ±1 Standardabweichung der Läufe; die Wahrheit als waagerechte Linie."""
    key = str(q)
    names = [("weibull", "Weibull-ML", C.COLOR_WEIBULL), ("exponential", "Exponential", C.COLOR_EXP), ("km", "Kaplan–Meier", C.COLOR_KM)]
    xs, ys, errs, cols = [], [], [], []
    for method, label, color in names:
        d = cell[method][key]
        suffix = f" (nur in {_de(d['defined_pct'], 0)} % der Läufe definiert)" if method == "km" and d["defined_pct"] < 100 else ""
        xs.append(label + suffix)
        ys.append(None if math.isnan(d["mean"]) else d["mean"])
        errs.append(0.0 if math.isnan(d["sd"]) else d["sd"])
        cols.append(color)
    fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=cols, error_y=dict(type="data", array=errs, visible=True), hovertemplate="%{x}: %{y:.2f} Tage<extra></extra>"))
    fig.add_hline(y=cell["truth"][key], line=dict(color=C.COLOR_TRUTH, width=2, dash="dot"), annotation_text=f"Wahrheit {_de(cell['truth'][key])}", annotation_position="top left")
    fig.update_yaxes(title_text=f"Zeit, bis {_de(q * 100, 0)} % abgeholt sind (Tage)", rangemode="tozero")
    return _base(fig, 380)


# ------------------------------------------------------------------------------------------------ Experiment 4

_Y_TICKS = (0.9, 0.5, 0.2, 0.05)
_X_TICKS = (0.5, 1, 2, 4, 8, 16)


def build_plot(result: dict, title: str):
    """Weibull-Plot: Punkte (ln t, ln(−ln Ŝ)) mit der Ausgleichsgeraden; Achsen mit Tagen bzw. Ŝ beschriftet."""
    fig = go.Figure()
    if result["ok"]:
        x, y = result["x"], result["y"]
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(color=C.COLOR_KM, size=5, opacity=0.7), name="Kaplan–Meier-Punkte",
                                 hovertemplate="Tag %{customdata[0]:.2f}, Ŝ = %{customdata[1]:.3f}<extra></extra>", customdata=np.c_[np.exp(x), np.exp(-np.exp(y))]))
        xl = np.array([x.min(), x.max()])
        fig.add_trace(go.Scatter(x=xl, y=result["intercept"] + result["slope"] * xl, mode="lines", line=dict(color=C.COLOR_WEIBULL, width=3), name="Ausgleichsgerade",
                                 hoverinfo="skip"))
    fig.update_xaxes(title_text="Zeit t (Tage, log-Achse)", range=[math.log(0.1), math.log(32)], tickvals=[math.log(v) for v in _X_TICKS], ticktext=[str(v).replace(".", ",") for v in _X_TICKS])
    fig.update_yaxes(title_text="Noch da Ŝ (doppelt-log-Achse)", range=[-5.0, 2.0], tickvals=[math.log(-math.log(v)) for v in _Y_TICKS], ticktext=[_de(v) for v in _Y_TICKS])
    fig.update_layout(title=dict(text=title, font=dict(size=13)))
    return _base(fig, 340)
