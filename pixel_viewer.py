"""
Pixel Waveform Viewer — Dash App
=================================
Run:
    pip install dash plotly numpy
    python pixel_viewer.py

Then open http://127.0.0.1:8050 in your browser.
"""

import os
import glob
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
PIXEL_PITCH = 3.72
N_COLS, N_ROWS = 256, 800
MAX_WAVEFORM_PTS = 2500   # downsample threshold
TICK_US = 0.1             # each sample = 0.1 microsecond

# Contrasting colour for hit marker lines (bright amber — visible against all
# palette waveform colours which are blues/greens/reds/purples)
HIT_LINE_COLOR = "rgba(255, 210, 50, 0.85)"

# ──────────────────────────────────────────────
# README panel — load Markdown
# ──────────────────────────────────────────────
README_PATH = os.path.join(os.path.dirname(__file__), "readme_panel.md")


def load_readme_md(path: str) -> dcc.Markdown:
    """Read a Markdown file and pass it directly to dcc.Markdown."""
    with open(path, encoding="utf-8") as fh:
        return dcc.Markdown(fh.read(), className="readme-body")

# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
def load_event(fname: str) -> dict:
    data = np.load(fname, allow_pickle=True)
    return {
        "name":           fname,
        "pixel_ids":      data["pixels"],
        "pixels_signals": data["pixels_signals"],
        "true_qs":        data["true_qs"],
        "qs":             data["qs"],
        "sig_sum":        np.cumsum(data["pixels_signals"], axis=1) * TICK_US,
        "adc_ticks_list": data["adc_ticks_list"],
        "integral_list":  data["integral_list"],
    }


def find_npz_files() -> list:
    """Return sorted list of .npz files in the working directory."""
    return sorted(glob.glob("*.npz"))


# ──────────────────────────────────────────────
# Downsample arrays with many points for improved responsiveness
# ──────────────────────────────────────────────
def ds_array(arr: np.ndarray):
    """Return (x_us, y) with x in µs, downsampled to MAX_WAVEFORM_PTS."""
    n = len(arr)
    step = max(1, n // MAX_WAVEFORM_PTS)
    y = arr[::step].tolist()
    x = [i * step * TICK_US for i in range(len(y))]
    return x, y


# ──────────────────────────────────────────────
# Pixel-map figure
# ──────────────────────────────────────────────
def build_pixel_map(events, lookups, all_pixel_ids, xs, ys) -> go.Figure:
    ref_lut = lookups[0]
    current_ref = np.array([
        np.max(np.abs(events[0]["pixels_signals"][ref_lut[int(pid)]]))
        if int(pid) in ref_lut else 0.0
        for pid in all_pixel_ids
    ])

    fig = go.Figure(go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(
            size=8,
            symbol="square",
            color=current_ref,
            colorscale="oranges",
            colorbar=dict(title="max |I|", thickness=14),
            line=dict(width=0),
        ),
        customdata=np.column_stack([all_pixel_ids, current_ref]),
        hovertemplate=(
            "Pixel %{customdata[0]:.0f}  "
            "z=%{x:.1f} mm  "
            "y=%{y:.1f} mm  "
            "max_I=%{customdata[1]:.1f}"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text="Pixel Map — click a pixel", font=dict(size=14)),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", family="'JetBrains Mono', monospace"),
        margin=dict(l=50, r=10, t=50, b=40),
        xaxis=dict(title="z (mm)", gridcolor="#21262d", zeroline=False),
        yaxis=dict(title="y (mm)", scaleanchor="x", scaleratio=1,
                   gridcolor="#21262d", zeroline=False),
        clickmode="event",
    )
    return fig


# ──────────────────────────────────────────────
# Waveform figures
# ──────────────────────────────────────────────
SUBPLOT_TITLES  = ("Pixel Signal (pixels_signals)", "True Charge (true_qs)", "Reco Charge (qs)")
SUBPLOT_YLABELS = ("Current (e-/&mu;s)", "True Q (e-)", "Reco Q (e-)")

PALETTE = [
    "#58a6ff", "#3fb950", "#f78166", "#d2a8ff",
    "#ffa657", "#79c0ff", "#56d364", "#ff7b72",
]


def _style_wave_fig(fig: go.Figure, title: str, n_events: int):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", family="'JetBrains Mono', monospace"),
        margin=dict(l=60, r=20, t=70, b=40),
        showlegend=n_events > 1,
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
    )
    axis_style = dict(gridcolor="#21262d", zeroline=False, linecolor="#30363d")
    for key in list(fig.layout):
        if key.startswith("xaxis") or key.startswith("yaxis"):
            fig.layout[key].update(**axis_style)
    for i, label in enumerate(SUBPLOT_YLABELS, start=1):
        ax = "yaxis" if i == 1 else f"yaxis{i}"
        fig.layout[ax].title = label
    fig.layout["xaxis3"].title = "Time (&mu;s)"


def empty_wave_fig(n_events: int = 1) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=SUBPLOT_TITLES, vertical_spacing=0.08)
    for row in range(1, 4):
        for _ in range(n_events):
            fig.add_trace(go.Scatter(x=[], y=[], mode="lines"), row=row, col=1)
    _style_wave_fig(fig, "Select a pixel", n_events)
    return fig


def build_wave_fig(pid: int, events, lookups) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=SUBPLOT_TITLES, vertical_spacing=0.08)

    keys_rows = [
        ("pixels_signals", 1, None,       None,     True),
        ("true_qs",        2, None,       None,     True),
        ("sig_sum",        2, "sig_sum",  dict(color="#42e0f5", dash="dash", width=1.2), False),
        ("qs",             3, None,       None,     True),
        ("sig_sum",        3, "sig_sum",  dict(color="#42e0f5", dash="dash", width=1.2), False),
    ]
    xref_names = ["x", "x2", "x3"]
    yref_names = ["y domain", "y2 domain", "y3 domain"]

    yref_annot = ["y", "y2", "y3"]
    shapes      = []
    annotations = []

    for ev_idx, (ev, lut) in enumerate(zip(events, lookups)):
        color = PALETTE[ev_idx % len(PALETTE)]
        name  = ev["name"]

        if int(pid) in lut:
            row_i = lut[int(pid)]

            raw_ticks    = ev["adc_ticks_list"][row_i]
            hit_times_us = np.asarray(raw_ticks).ravel()
            hit_times_us = hit_times_us[hit_times_us != 0]

            for key, row, label, line_override, show_integral in keys_rows:
                arr  = ev[key][row_i]
                x, y = ds_array(arr)

                line_style = dict(color=color, width=1.5)
                if line_override:
                    line_style.update(line_override)

                trace_name = label if label else name

                # Waveform traces
                fig.add_trace(
                    go.Scatter(
                        x=x, y=y,
                        mode="lines",
                        name=trace_name,
                        legendgroup=trace_name,
                        showlegend=(row == 1 or label is not None),
                        line=line_style,
                    ),
                    row=row, col=1,
                )

                if show_integral:
                    integral = float(np.trapezoid(ev[key][row_i], dx=TICK_US))
                    y_offset = 0.97 - ev_idx * 0.07
                    annotations.append(dict(
                        xref="paper",
                        yref=f"{yref_annot[row - 1]} domain",
                        x=0.15,
                        y=y_offset,
                        xanchor="right",
                        yanchor="top",
                        text=f"∫ = {integral:.3g}",
                        showarrow=False,
                        font=dict(size=12, color=color,
                                  family="'JetBrains Mono', monospace"),
                        bgcolor="rgba(13,17,23,0.7)",
                        bordercolor=color,
                        borderwidth=1,
                        borderpad=3,
                    ))

            # Vertical hit markers
            for t in hit_times_us:
                for xref, yref in zip(xref_names, yref_names):
                    shapes.append(dict(
                        type="line",
                        xref=xref, yref=yref,
                        x0=t, x1=t,
                        y0=0, y1=1,
                        line=dict(color=HIT_LINE_COLOR, width=1.3, dash="dot"),
                    ))
        else:
            for _, row in keys_rows:
                fig.add_trace(
                    go.Scatter(x=[], y=[], mode="lines", name=name,
                               legendgroup=name, showlegend=False,
                               line=dict(color=color)),
                    row=row, col=1,
                )

    _style_wave_fig(fig, f"Waveforms for Pixel {pid}", len(events))
    layout_updates = {}
    if shapes:
        layout_updates["shapes"] = shapes
    if annotations:
        layout_updates["annotations"] = annotations
    if layout_updates:
        fig.update_layout(**layout_updates)
    return fig

# ──────────────────────────────────────────────
# Dash App
# ──────────────────────────────────────────────
app = dash.Dash(__name__, title="Pixel Waveform Viewer")

_initial_files = find_npz_files()
_initial_value = _initial_files[0] if _initial_files else None

app.layout = html.Div(
    children=[
        # Header
        html.Div(
            className="header",
            children=[
                html.Div(children=[
                    html.H1("Pixel Waveform Viewer"),
                    html.Span(id="file-info", className="file-info"),
                ]),

                html.Div(
                    className="file-selector",
                    children=[
                        html.Label("File:"),
                        dcc.Dropdown(
                            id="file-dropdown",
                            className="file-dropdown",
                            options=[{"label": f, "value": f} for f in _initial_files],
                            value=_initial_value,
                            clearable=False,
                        ),
                        html.Button(
                            "⟳ Refresh",
                            id="refresh-btn",
                            n_clicks=0,
                            className="refresh-btn",
                        ),
                    ],
                ),
            ],
        ),

        # Two-panel plot layout
        html.Div(
            className="plot-panels",
            children=[
                html.Div(
                    className="plot-panel-left",
                    children=[
                        dcc.Graph(
                            id="pixel-map",
                            style={"height": "78vh"},
                            config={"scrollZoom": True, "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
                        ),
                    ],
                ),
                html.Div(
                    className="plot-panel-right",
                    children=[
                        dcc.Graph(
                            id="wave-plot",
                            style={"height": "78vh"},
                            config={"displayModeBar": True},
                        ),
                    ],
                ),
            ],
        ),

        # README / description panel
        html.Details(
            open=True,
            className="readme-details",
            children=[
                html.Summary("ℹ️  About this viewer"),
                load_readme_md(README_PATH),
            ],
        ),

        # Stores
        dcc.Store(id="selected-pixel", data=None),
        dcc.Store(id="loaded-file",    data=None),
    ],
)


# ──────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────

# 1. Refresh button repopulates the dropdown
@app.callback(
    Output("file-dropdown", "options"),
    Input("refresh-btn", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_file_list(_):
    files = find_npz_files()
    return [{"label": f, "value": f} for f in files]


# 2. File selection: rebuild pixel map, reset waveform, update info text
@app.callback(
    Output("pixel-map",      "figure"),
    Output("wave-plot",      "figure"),
    Output("file-info",      "children"),
    Output("loaded-file",    "data"),
    Output("selected-pixel", "data"),
    Input("file-dropdown",   "value"),
)
def load_file(fname):
    if not fname or not os.path.isfile(fname):
        blank = go.Figure()
        blank.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                            font=dict(color="#e6edf3"))
        return blank, empty_wave_fig(), "No file selected", None, None

    ev      = load_event(fname)
    events  = [ev]
    lookups = [{int(pid): i for i, pid in enumerate(ev["pixel_ids"])}]

    all_pids = np.unique(ev["pixel_ids"])
    xs = (all_pids % N_COLS) * PIXEL_PITCH
    ys = (all_pids // N_COLS) * PIXEL_PITCH

    pmap = build_pixel_map(events, lookups, all_pids, xs, ys)
    info = f"{fname}  ·  {len(all_pids)} pixels"

    return pmap, empty_wave_fig(len(events)), info, fname, None


# 3. Pixel click: store pixel id
@app.callback(
    Output("selected-pixel", "data", allow_duplicate=True),
    Input("pixel-map",  "clickData"),
    State("loaded-file", "data"),
    prevent_initial_call=True,
)
def store_click(click_data, fname):
    if click_data is None or fname is None:
        return dash.no_update
    pt = click_data["points"][0]
    if "customdata" in pt:
        return int(pt["customdata"])
    # fallback via pointIndex
    ev       = load_event(fname)
    all_pids = np.unique(ev["pixel_ids"])
    return int(all_pids[pt["pointIndex"]])


# 4. Selected pixel + current file: waveform
@app.callback(
    Output("wave-plot", "figure", allow_duplicate=True),
    Input("selected-pixel", "data"),
    State("loaded-file",    "data"),
    prevent_initial_call=True,
)
def update_waveform(pid, fname):
    if pid is None or fname is None:
        return empty_wave_fig()
    ev      = load_event(fname)
    events  = [ev]
    lookups = [{int(p): i for i, p in enumerate(ev["pixel_ids"])}]
    return build_wave_fig(pid, events, lookups)


# Run server
if __name__ == "__main__":
    app.run(debug=True, port=8050)
