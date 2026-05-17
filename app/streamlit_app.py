"""
Li-Ion Battery Reliability & Degradation Analytics Dashboard
============================================================
Modern dark theme · sidebar navigation · Plotly visuals
"""

from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Battery Reliability Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "battery_features.parquet"
XGB_MODEL_PATH = PROJECT_ROOT / "artifacts" / "xgboost_rul_model.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "artifacts" / "feature_columns.json"
WEIBULL_SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "weibull_summary.json"


# =====================================================
# Theme Palette (Vercel / Linear inspired)
# =====================================================

COLORS = {
    "bg":           "#0A0A0A",   # near black
    "bg_card":      "#111111",   # card background
    "bg_elevated":  "#161616",   # hover/elevated surfaces
    "border":       "#1F1F1F",   # subtle borders
    "border_hover": "#2A2A2A",
    "text":         "#EDEDED",   # primary text
    "text_dim":     "#A0A0A0",   # secondary text
    "text_muted":   "#666666",   # tertiary / labels
    "accent":       "#3B82F6",   # primary blue
    "accent_2":     "#8B5CF6",   # purple
    "green":        "#10B981",   # healthy
    "amber":        "#F59E0B",   # warning
    "red":          "#EF4444",   # critical / EoL
    "gradient_1":   "linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%)",
    "gradient_2":   "linear-gradient(135deg, #10B981 0%, #3B82F6 100%)",
}

CELL_COLORS = ["#3B82F6", "#8B5CF6", "#EC4899", "#F59E0B", "#10B981",
               "#06B6D4", "#EF4444", "#14B8A6", "#A855F7", "#F97316"]


# =====================================================
# Global CSS — Dark Theme
# =====================================================

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* App background */
    .stApp {{
        background: {COLORS['bg']};
        color: {COLORS['text']};
        font-family: 'Inter', -apple-system, sans-serif;
    }}

    /* Remove top padding, hide Streamlit chrome */
    .block-container {{
        padding: 2rem 3rem 3rem 3rem !important;
        max-width: 100%;
    }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    [data-testid="stToolbar"] {{ display: none; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: #050505 !important;
        border-right: 1px solid {COLORS['border']};
    }}
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1.5rem;
    }}
    [data-testid="stSidebar"] .stRadio > label {{
        color: {COLORS['text_dim']};
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    [data-testid="stSidebar"] .stRadio > div {{
        gap: 4px !important;
    }}
    [data-testid="stSidebar"] .stRadio > div > label {{
        background: transparent;
        padding: 10px 14px !important;
        border-radius: 8px;
        border: 1px solid transparent;
        transition: all 0.15s;
        cursor: pointer;
        color: {COLORS['text_dim']} !important;
    }}
    [data-testid="stSidebar"] .stRadio > div > label:hover {{
        background: {COLORS['bg_card']};
        color: {COLORS['text']} !important;
    }}
    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
    [data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {{
        background: {COLORS['bg_elevated']};
        color: {COLORS['text']} !important;
        border-color: {COLORS['border_hover']};
    }}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {{
        font-size: 13.5px;
        font-weight: 500;
    }}
    [data-testid="stSidebar"] hr {{
        margin: 1rem 0 !important;
        border-color: {COLORS['border']} !important;
    }}

    /* Headings */
    h1, h2, h3, h4 {{
        color: {COLORS['text']} !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }}
    .stMarkdown p {{ color: {COLORS['text_dim']}; }}

    /* Hero / Page title */
    .page-title {{
        font-size: 28px;
        font-weight: 700;
        color: {COLORS['text']};
        margin: 0 0 4px 0;
        letter-spacing: -0.025em;
    }}
    .page-subtitle {{
        font-size: 14px;
        color: {COLORS['text_dim']};
        margin: 0 0 24px 0;
        font-weight: 400;
    }}

    /* Section header */
    .section-title {{
        font-size: 13px;
        font-weight: 600;
        color: {COLORS['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 24px 0 12px 0;
    }}

    /* KPI Cards */
    .kpi {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 18px 20px;
        transition: all 0.2s;
        height: 100%;
    }}
    .kpi:hover {{
        border-color: {COLORS['border_hover']};
        background: {COLORS['bg_elevated']};
    }}
    .kpi-label {{
        font-size: 11px;
        font-weight: 600;
        color: {COLORS['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }}
    .kpi-value {{
        font-size: 28px;
        font-weight: 700;
        color: {COLORS['text']};
        line-height: 1;
        letter-spacing: -0.02em;
        font-family: 'Inter', sans-serif;
    }}
    .kpi-unit {{
        font-size: 14px;
        color: {COLORS['text_muted']};
        font-weight: 500;
        margin-left: 4px;
    }}
    .kpi-delta {{
        font-size: 12px;
        margin-top: 6px;
        font-weight: 500;
    }}
    .kpi-delta.up   {{ color: {COLORS['green']}; }}
    .kpi-delta.down {{ color: {COLORS['red']}; }}
    .kpi-delta.flat {{ color: {COLORS['text_muted']}; }}

    /* Chart container */
    .chart-card {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }}
    .chart-title {{
        font-size: 14px;
        font-weight: 600;
        color: {COLORS['text']};
        margin-bottom: 4px;
    }}
    .chart-subtitle {{
        font-size: 12px;
        color: {COLORS['text_muted']};
        margin-bottom: 16px;
    }}

    /* Info / callout */
    .callout {{
        background: linear-gradient(135deg, rgba(59,130,246,0.08) 0%, rgba(139,92,246,0.05) 100%);
        border: 1px solid rgba(59,130,246,0.2);
        border-left: 3px solid {COLORS['accent']};
        border-radius: 10px;
        padding: 14px 18px;
        margin: 16px 0;
        font-size: 13.5px;
        color: {COLORS['text']};
        line-height: 1.6;
    }}
    .callout b {{ color: {COLORS['text']}; }}
    .callout-success {{
        border-left-color: {COLORS['green']};
        background: linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(16,185,129,0.03) 100%);
    }}
    .callout-warning {{
        border-left-color: {COLORS['amber']};
        background: linear-gradient(135deg, rgba(245,158,11,0.08) 0%, rgba(245,158,11,0.03) 100%);
    }}

    /* Status pills */
    .pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .pill-green {{ background: rgba(16,185,129,0.12); color: {COLORS['green']}; }}
    .pill-amber {{ background: rgba(245,158,11,0.12); color: {COLORS['amber']}; }}
    .pill-red   {{ background: rgba(239,68,68,0.12); color: {COLORS['red']}; }}
    .pill-blue  {{ background: rgba(59,130,246,0.12); color: {COLORS['accent']}; }}
    .pill .dot {{
        width: 6px; height: 6px; border-radius: 50%;
        background: currentColor;
        box-shadow: 0 0 8px currentColor;
    }}

    /* Streamlit widgets */
    .stSelectbox > div > div {{
        background: {COLORS['bg_card']} !important;
        border: 1px solid {COLORS['border']} !important;
        color: {COLORS['text']} !important;
        border-radius: 8px !important;
    }}
    .stSelectbox label, .stSlider label {{
        color: {COLORS['text_dim']} !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    .stSlider > div > div > div {{
        background: {COLORS['border']} !important;
    }}
    .stSlider [data-baseweb="slider"] [role="slider"] {{
        background: {COLORS['accent']} !important;
        border: 2px solid {COLORS['text']} !important;
    }}

    /* DataFrames */
    .stDataFrame {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        overflow: hidden;
    }}

    /* Sidebar brand */
    .brand {{
        display: flex; align-items: center; gap: 10px;
        padding: 8px 14px 20px 14px;
        border-bottom: 1px solid {COLORS['border']};
        margin-bottom: 18px;
    }}
    .brand-logo {{
        width: 32px; height: 32px;
        border-radius: 8px;
        background: {COLORS['gradient_1']};
        display: flex; align-items: center; justify-content: center;
        font-size: 16px; font-weight: 800; color: white;
    }}
    .brand-name {{
        font-size: 14px; font-weight: 700; color: {COLORS['text']};
        line-height: 1.2;
    }}
    .brand-sub {{
        font-size: 11px; color: {COLORS['text_muted']};
    }}

    /* Sidebar context block */
    .ctx-block {{
        padding: 12px 14px;
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        margin: 12px 0;
    }}
    .ctx-row {{
        display: flex; justify-content: space-between; align-items: center;
        font-size: 12px; padding: 3px 0;
    }}
    .ctx-row .lbl {{ color: {COLORS['text_muted']}; }}
    .ctx-row .val {{ color: {COLORS['text']}; font-weight: 600; }}

    /* Section divider */
    .divider {{
        height: 1px;
        background: {COLORS['border']};
        margin: 28px 0 24px 0;
    }}
</style>
""", unsafe_allow_html=True)


# =====================================================
# Plotly Default Layout
# =====================================================

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS['text_dim'], size=12),
    xaxis=dict(
        gridcolor=COLORS['border'],
        zerolinecolor=COLORS['border'],
        linecolor=COLORS['border'],
        tickfont=dict(color=COLORS['text_muted'], size=11),
        title_font=dict(color=COLORS['text_dim'], size=12)
    ),
    yaxis=dict(
        gridcolor=COLORS['border'],
        zerolinecolor=COLORS['border'],
        linecolor=COLORS['border'],
        tickfont=dict(color=COLORS['text_muted'], size=11),
        title_font=dict(color=COLORS['text_dim'], size=12)
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=COLORS['border'],
        font=dict(color=COLORS['text_dim'], size=11)
    ),
    margin=dict(l=10, r=10, t=10, b=10),
    hoverlabel=dict(
        bgcolor=COLORS['bg_elevated'],
        bordercolor=COLORS['border_hover'],
        font=dict(family="Inter, sans-serif", color=COLORS['text'])
    )
)


def style_fig(fig, height=380, title=None):
    """Apply consistent dark styling to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    if title is None:
        fig.update_layout(title=None)
    return fig


# =====================================================
# Data Loaders (cached)
# =====================================================

@st.cache_data
def load_features():
    return pd.read_parquet(FEATURE_PATH)


@st.cache_resource
def load_xgb_model():
    return joblib.load(XGB_MODEL_PATH)


@st.cache_data
def load_feature_columns():
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["feature_columns"] if isinstance(data, dict) else data


@st.cache_data
def load_weibull_summary():
    with open(WEIBULL_SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================
# Helpers
# =====================================================

def kpi(label, value, unit="", delta=None, delta_direction="flat"):
    """Render a single KPI card."""
    unit_html = f'<span class="kpi-unit">{unit}</span>' if unit else ""
    delta_html = (
        f'<div class="kpi-delta {delta_direction}">{delta}</div>'
        if delta else ""
    )
    st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}{unit_html}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


def health_pill(retention):
    """Return a coloured health pill for a given capacity retention."""
    if retention >= 0.85:
        return '<span class="pill pill-green"><span class="dot"></span>Healthy</span>'
    if retention >= 0.75:
        return '<span class="pill pill-amber"><span class="dot"></span>Aging</span>'
    if retention >= 0.70:
        return '<span class="pill pill-amber"><span class="dot"></span>Near EoL</span>'
    return '<span class="pill pill-red"><span class="dot"></span>End of Life</span>'


def predict_rul(model, feature_columns, row_df):
    X = row_df[feature_columns].copy()
    X = X.fillna(X.median(numeric_only=True))
    pred = model.predict(X)[0]
    return max(0, float(pred))


def make_weibull_curves(beta, eta, max_cycle=300):
    cycles = np.arange(1, max_cycle + 1)
    survival = np.exp(-((cycles / eta) ** beta))
    hazard = (beta / eta) * ((cycles / eta) ** (beta - 1))
    return pd.DataFrame({
        "cycle": cycles,
        "survival_probability": survival,
        "hazard_rate": hazard
    })


def clean_cell_name(cell_id):
    return cell_id.split("_")[-1] if "_" in cell_id else cell_id


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# =====================================================
# Load Everything
# =====================================================

features = load_features()
xgb_model = load_xgb_model()
feature_columns = load_feature_columns()
weibull_summary = load_weibull_summary()


# =====================================================
# Sidebar — Brand + Navigation + Filters
# =====================================================

with st.sidebar:
    st.markdown("""
        <div class="brand">
            <div class="brand-logo">⚡</div>
            <div>
                <div class="brand-name">Battery Analytics</div>
                <div class="brand-sub">Reliability · RUL · ML</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio(
    "Navigation",
    [
        "Fleet Overview",
        "Cell Explorer",
        "RUL Prediction",
        "Reliability Analysis",
        "Model Explainability",
        "Model Comparison"
    ],
    label_visibility="visible",
    key="nav"
)

    st.markdown("---")

    cell_ids = sorted(features["cell_id"].unique())
    selected_cell = st.selectbox(
        "Battery Cell",
        cell_ids,
        format_func=clean_cell_name
    )

    selected_cell_df = features[features["cell_id"] == selected_cell].copy()
    cycle_min = int(selected_cell_df["discharge_cycle"].min())
    cycle_max = int(selected_cell_df["discharge_cycle"].max())

    selected_cycle = st.slider(
        "Discharge Cycle",
        min_value=cycle_min,
        max_value=cycle_max,
        value=cycle_max
    )

    selected_cycle_df = selected_cell_df[
        selected_cell_df["discharge_cycle"] == selected_cycle
    ].copy()
    if selected_cycle_df.empty:
        selected_cycle_df = selected_cell_df.tail(1)

    # Dataset context block
    latest_retention = float(selected_cell_df["capacity_retention"].iloc[-1])
    st.markdown(f"""
        <div class="ctx-block">
            <div class="ctx-row"><span class="lbl">Active cell</span>
                <span class="val">{clean_cell_name(selected_cell)}</span></div>
            <div class="ctx-row"><span class="lbl">Health</span>
                <span class="val">{health_pill(latest_retention)}</span></div>
            <div class="ctx-row"><span class="lbl">Cycles</span>
                <span class="val">{cycle_max}</span></div>
        </div>
        <div class="ctx-block">
            <div class="ctx-row"><span class="lbl">Total cells</span>
                <span class="val">{features['cell_id'].nunique()}</span></div>
            <div class="ctx-row"><span class="lbl">Total cycles</span>
                <span class="val">{len(features):,}</span></div>
        </div>
    """, unsafe_allow_html=True)


# =====================================================
# PAGE: Fleet Overview
# =====================================================

if page == "Fleet Overview":
    st.markdown('<div class="page-title">Fleet Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Cross-cell degradation patterns and health distribution</div>',
                unsafe_allow_html=True)

    # KPI row
    total_cells = features["cell_id"].nunique()
    total_cycles = len(features)
    avg_retention = features["capacity_retention"].mean()
    failed_cells = (features.groupby("cell_id")["capacity_retention"].min() < 0.70).sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Cells Monitored", total_cells)
    with c2: kpi("Discharge Cycles", f"{total_cycles:,}")
    with c3: kpi("Avg Retention", f"{avg_retention:.1%}",
                 delta="across fleet", delta_direction="flat")
    with c4: kpi("Reached EoL", f"{failed_cells}",
                 delta=f"of {total_cells} cells",
                 delta_direction="down" if failed_cells > 0 else "flat")

    # Capacity fade chart
    section("Capacity Fade — Full Fleet")
    with st.container():
        fig = px.line(
            features, x="discharge_cycle", y="capacity_retention",
            color="cell_id", color_discrete_sequence=CELL_COLORS,
            hover_data={"capacity_ah": ":.3f", "rul": True, "cell_id": False}
        )
        fig.add_hline(y=0.70, line_dash="dash", line_color=COLORS['red'],
                      line_width=1.5, opacity=0.7,
                      annotation_text="EoL · 70%",
                      annotation_position="bottom right",
                      annotation_font_color=COLORS['red'],
                      annotation_font_size=11)
        fig.update_traces(line=dict(width=2), hovertemplate=None)
        fig.update_layout(
            xaxis_title="Discharge Cycle",
            yaxis_title="Capacity Retention",
            legend_title=None,
            showlegend=False
        )
        style_fig(fig, height=440)
        st.plotly_chart(fig, use_container_width=True)

    # Two side-by-side distributions
    section("Distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(features, x="rul", nbins=40,
                           color_discrete_sequence=[COLORS['accent']])
        fig.update_traces(marker_line_width=0, opacity=0.85)
        fig.update_layout(
            xaxis_title="Remaining Useful Life (cycles)",
            yaxis_title="Count",
            bargap=0.05
        )
        style_fig(fig, height=320)
        st.markdown('<div class="chart-title">RUL Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Cycles remaining across all observations</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(features, x="capacity_ah", nbins=40,
                           color_discrete_sequence=[COLORS['accent_2']])
        fig.update_traces(marker_line_width=0, opacity=0.85)
        fig.update_layout(
            xaxis_title="Discharge Capacity (Ah)",
            yaxis_title="Count",
            bargap=0.05
        )
        style_fig(fig, height=320)
        st.markdown('<div class="chart-title">Capacity Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Distribution of discharge capacity values</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

    # Per-cell summary table
    section("Per-Cell Summary")
    summary = features.groupby("cell_id").agg(
        cycles=("discharge_cycle", "max"),
        first_capacity=("capacity_ah", "first"),
        last_capacity=("capacity_ah", "last"),
        last_retention=("capacity_retention", "last"),
        min_retention=("capacity_retention", "min"),
    ).reset_index()
    summary["cell"] = summary["cell_id"].apply(clean_cell_name)
    summary["status"] = summary["min_retention"].apply(
        lambda r: "End of Life" if r < 0.70 else
                  ("Near EoL" if r < 0.75 else
                  ("Aging" if r < 0.85 else "Healthy"))
    )
    display_summary = summary[["cell", "cycles", "first_capacity",
                                "last_capacity", "last_retention", "status"]].copy()
    display_summary.columns = ["Cell", "Cycles", "Initial Cap (Ah)",
                                "Final Cap (Ah)", "Retention", "Status"]
    display_summary["Initial Cap (Ah)"] = display_summary["Initial Cap (Ah)"].round(3)
    display_summary["Final Cap (Ah)"]   = display_summary["Final Cap (Ah)"].round(3)
    display_summary["Retention"]        = (display_summary["Retention"] * 100).round(1).astype(str) + "%"
    st.dataframe(display_summary, use_container_width=True, hide_index=True)


# =====================================================
# PAGE: Cell Explorer
# =====================================================

elif page == "Cell Explorer":
    st.markdown(f'<div class="page-title">Cell Explorer · {clean_cell_name(selected_cell)}</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Per-cycle degradation signals for the selected cell</div>',
                unsafe_allow_html=True)

    latest = selected_cell_df.tail(1).iloc[0]
    first  = selected_cell_df.iloc[0]
    cap_drop = first["capacity_ah"] - latest["capacity_ah"]

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Latest Cycle", int(latest["discharge_cycle"]))
    with c2: kpi("Capacity", f"{latest['capacity_ah']:.3f}", unit="Ah",
                 delta=f"-{cap_drop:.3f} Ah from start", delta_direction="down")
    with c3: kpi("Retention", f"{latest['capacity_retention']*100:.1f}", unit="%")
    with c4: kpi("Actual RUL", int(latest["rul"]), unit="cycles")

    # Capacity fade with selected cycle marker
    section("Capacity Retention")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=selected_cell_df["discharge_cycle"],
        y=selected_cell_df["capacity_retention"],
        mode="lines", line=dict(color=COLORS['accent'], width=2.5),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
        name="Retention",
        hovertemplate="Cycle %{x}<br>Retention %{y:.1%}<extra></extra>"
    ))
    fig.add_hline(y=0.70, line_dash="dash", line_color=COLORS['red'],
                  line_width=1, opacity=0.6,
                  annotation_text="EoL · 70%",
                  annotation_position="bottom right",
                  annotation_font_color=COLORS['red'])
    fig.add_vline(x=selected_cycle, line_dash="dot",
                  line_color=COLORS['amber'], line_width=2,
                  annotation_text=f"Cycle {selected_cycle}",
                  annotation_position="top",
                  annotation_font_color=COLORS['amber'])
    fig.update_layout(xaxis_title="Discharge Cycle",
                      yaxis_title="Capacity Retention",
                      yaxis_tickformat=".0%")
    style_fig(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)

    # 2x2 grid of secondary signals
    section("Degradation Signals")
    col1, col2 = st.columns(2)

    def small_line(df, y_col, title, subtitle, color, y_label):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["discharge_cycle"], y=df[y_col],
            mode="lines", line=dict(color=color, width=2),
            hovertemplate=f"Cycle %{{x}}<br>{y_label} %{{y:.4f}}<extra></extra>"
        ))
        fig.update_layout(xaxis_title="Cycle", yaxis_title=y_label)
        style_fig(fig, height=270)
        st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chart-subtitle">{subtitle}</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

    with col1:
        small_line(selected_cell_df, "voltage_mean",
                   "Mean Discharge Voltage", "Average voltage during discharge",
                   COLORS['accent_2'], "Voltage (V)")
    with col2:
        small_line(selected_cell_df, "time_to_3v",
                   "Time to 3.0 V Cutoff", "How long the cell sustains voltage",
                   COLORS['green'], "Seconds")

    col3, col4 = st.columns(2)
    with col3:
        small_line(selected_cell_df, "temp_max",
                   "Peak Temperature", "Maximum cell temperature per cycle",
                   COLORS['amber'], "Temperature (°C)")
    with col4:
        # Combined impedance chart
        fig = go.Figure()
        if "Re" in selected_cell_df.columns:
            fig.add_trace(go.Scatter(
                x=selected_cell_df["discharge_cycle"],
                y=selected_cell_df["Re"],
                mode="lines", name="Re (electrolyte)",
                line=dict(color=COLORS['accent'], width=2)
            ))
        if "Rct" in selected_cell_df.columns:
            fig.add_trace(go.Scatter(
                x=selected_cell_df["discharge_cycle"],
                y=selected_cell_df["Rct"],
                mode="lines", name="Rct (charge transfer)",
                line=dict(color="#EC4899", width=2)
            ))
        fig.update_layout(xaxis_title="Cycle", yaxis_title="Resistance (Ω)",
                          legend=dict(orientation="h", yanchor="bottom",
                                      y=1.02, xanchor="right", x=1))
        style_fig(fig, height=270)
        st.markdown('<div class="chart-title">Impedance Growth</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Internal resistance — leading degradation indicator</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)


# =====================================================
# PAGE: RUL Prediction
# =====================================================

elif page == "RUL Prediction":
    st.markdown('<div class="page-title">Remaining Useful Life</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">XGBoost prediction for {clean_cell_name(selected_cell)} · Cycle {selected_cycle}</div>',
                unsafe_allow_html=True)

    pred_rul = predict_rul(xgb_model, feature_columns, selected_cycle_df)
    actual_rul = float(selected_cycle_df["rul"].iloc[0])
    error = pred_rul - actual_rul
    current_retention = float(selected_cycle_df["capacity_retention"].iloc[0])
    current_capacity = float(selected_cycle_df["capacity_ah"].iloc[0])

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Selected Cycle", selected_cycle)
    with c2: kpi("Predicted RUL", f"{pred_rul:.0f}", unit="cycles",
                 delta=f"Δ {error:+.0f} vs actual",
                 delta_direction="up" if abs(error) < 10 else ("down" if abs(error) > 30 else "flat"))
    with c3: kpi("Actual RUL", f"{actual_rul:.0f}", unit="cycles")
    with c4: kpi("Retention", f"{current_retention*100:.1f}", unit="%")

    st.markdown(f"""
        <div class="callout">
            At cycle <b>{selected_cycle}</b>, the model predicts <b>{pred_rul:.0f}</b> remaining cycles
            for cell <b>{clean_cell_name(selected_cell)}</b>. Current capacity is <b>{current_capacity:.3f} Ah</b>
            ({current_retention:.1%} of original). Actual recorded RUL is <b>{actual_rul:.0f}</b> cycles.
        </div>
    """, unsafe_allow_html=True)

    # Actual vs predicted across full life
    section("Predicted vs Actual RUL Trajectory")

    pred_df = selected_cell_df.copy()
    X_pred = pred_df[feature_columns].copy()
    X_pred = X_pred.fillna(X_pred.median(numeric_only=True))
    pred_df["predicted_rul"] = xgb_model.predict(X_pred).clip(min=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pred_df["discharge_cycle"], y=pred_df["rul"],
        mode="lines", name="Actual",
        line=dict(color=COLORS['text_dim'], width=2, dash="dash"),
        hovertemplate="Cycle %{x}<br>Actual RUL %{y:.0f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=pred_df["discharge_cycle"], y=pred_df["predicted_rul"],
        mode="lines", name="Predicted",
        line=dict(color=COLORS['accent'], width=2.5),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.06)",
        hovertemplate="Cycle %{x}<br>Predicted RUL %{y:.0f}<extra></extra>"
    ))
    fig.add_vline(x=selected_cycle, line_dash="dot",
                  line_color=COLORS['amber'], line_width=2,
                  annotation_text=f"Cycle {selected_cycle}",
                  annotation_position="top",
                  annotation_font_color=COLORS['amber'])
    fig.update_layout(
        xaxis_title="Discharge Cycle",
        yaxis_title="Remaining Useful Life",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    style_fig(fig, height=420)
    st.plotly_chart(fig, use_container_width=True)

    # Feature values at selected cycle
    section("Feature Snapshot · Selected Cycle")
    snapshot = selected_cycle_df[feature_columns + ["rul"]].T.reset_index()
    snapshot.columns = ["Feature", "Value"]
    snapshot["Value"] = snapshot["Value"].apply(
        lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
    )
    st.dataframe(snapshot, use_container_width=True, hide_index=True, height=400)


# =====================================================
# PAGE: Reliability Analysis
# =====================================================

elif page == "Reliability Analysis":
    st.markdown('<div class="page-title">Weibull Reliability Analysis</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Fleet-level failure distribution and survival curves</div>',
                unsafe_allow_html=True)

    beta = weibull_summary["beta"]
    eta = weibull_summary["eta"]
    b10 = weibull_summary["b10_life"]
    b50 = weibull_summary["b50_life"]

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("β · Shape", f"{beta:.2f}",
                 delta="Wear-out" if beta > 1 else ("Random" if abs(beta-1) < 0.1 else "Infant"),
                 delta_direction="flat")
    with c2: kpi("η · Scale", f"{eta:.0f}", unit="cycles")
    with c3: kpi("B10 Life", f"{b10:.0f}", unit="cycles",
                 delta="10% of fleet fails by")
    with c4: kpi("B50 Life", f"{b50:.0f}", unit="cycles",
                 delta="median life")

    behavior = weibull_summary.get("failure_behavior", "Wear-out")
    st.markdown(f"""
        <div class="callout">
            Estimated failure behavior: <b>{behavior}</b>.
            With β = {beta:.2f} the failure rate {'increases' if beta > 1 else 'is constant'} with cycles —
            characteristic of {'a wear-out mechanism' if beta > 1 else 'random failures'}.
            Characteristic life η = {eta:.0f} cycles (the cycle at which ~63% of cells have failed).
        </div>
    """, unsafe_allow_html=True)

    max_cycle = int(features["discharge_cycle"].max())
    curve_df = make_weibull_curves(beta, eta, max_cycle=max_cycle)

    # Survival + hazard side by side
    section("Survival & Hazard")
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve_df["cycle"], y=curve_df["survival_probability"],
            mode="lines", line=dict(color=COLORS['accent'], width=2.5),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.1)",
            hovertemplate="Cycle %{x}<br>Survival %{y:.1%}<extra></extra>"
        ))
        fig.add_vline(x=b10, line_dash="dot", line_color=COLORS['amber'],
                      annotation_text=f"B10 · {b10:.0f}",
                      annotation_font_color=COLORS['amber'])
        fig.add_vline(x=b50, line_dash="dot", line_color=COLORS['accent_2'],
                      annotation_text=f"B50 · {b50:.0f}",
                      annotation_font_color=COLORS['accent_2'])
        fig.update_layout(xaxis_title="Cycle",
                          yaxis_title="Survival Probability",
                          yaxis_tickformat=".0%")
        style_fig(fig, height=360)
        st.markdown('<div class="chart-title">Survival Function S(t)</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Probability a cell survives past cycle N</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve_df["cycle"], y=curve_df["hazard_rate"],
            mode="lines", line=dict(color=COLORS['red'], width=2.5),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.1)",
            hovertemplate="Cycle %{x}<br>Hazard %{y:.4f}<extra></extra>"
        ))
        fig.update_layout(xaxis_title="Cycle",
                          yaxis_title="Instantaneous Failure Rate")
        style_fig(fig, height=360)
        st.markdown('<div class="chart-title">Hazard Function h(t)</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Risk of failure at each cycle, given survival</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

    # Dataset details
    section("Reliability Dataset")
    c5, c6, c7 = st.columns(3)
    with c5: kpi("Cells in Study", weibull_summary["num_cells"])
    with c6: kpi("Failed Cells", weibull_summary["num_failed_cells"],
                 delta="reached 70% retention", delta_direction="down")
    with c7: kpi("Censored Cells", weibull_summary["num_censored_cells"],
                 delta="still alive at end of test", delta_direction="flat")


# =====================================================
# PAGE: Model Explainability
# =====================================================

elif page == "Model Explainability":
    st.markdown('<div class="page-title">Model Explainability</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">XGBoost feature importance and model insights</div>',
                unsafe_allow_html=True)

    importance = pd.Series(
        xgb_model.feature_importances_, index=feature_columns
    ).sort_values(ascending=True)

    n_features = len(importance)
    top_feature = importance.idxmax()
    top_importance = importance.max()

    c1, c2, c3 = st.columns(3)
    with c1: kpi("Total Features", n_features)
    with c2: kpi("Top Driver", top_feature)
    with c3: kpi("Top Importance", f"{top_importance:.3f}",
                 delta="of cumulative gain", delta_direction="flat")

    st.markdown(f"""
        <div class="callout">
            The RUL model is driven primarily by <b>{top_feature}</b> and other degradation-related signals.
            Capacity retention, voltage behavior during discharge, and impedance growth dominate
            the prediction — consistent with established lithium-ion ageing mechanisms.
        </div>
    """, unsafe_allow_html=True)

    section("Global Feature Importance")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importance.values, y=importance.index,
        orientation="h",
        marker=dict(
            color=importance.values,
            colorscale=[[0, "#1F2937"], [0.5, COLORS['accent']], [1, COLORS['accent_2']]],
            line=dict(width=0)
        ),
        hovertemplate="<b>%{y}</b><br>Importance %{x:.4f}<extra></extra>"
    ))
    fig.update_layout(
        xaxis_title="Importance (gain)",
        yaxis_title=None,
        showlegend=False
    )
    style_fig(fig, height=max(420, n_features * 24))
    st.plotly_chart(fig, use_container_width=True)

    section("Ranked Features")
    table_df = importance.reset_index()
    table_df.columns = ["Feature", "Importance"]
    table_df = table_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    table_df["Rank"] = table_df.index + 1
    table_df["Importance"] = table_df["Importance"].round(4)
    table_df = table_df[["Rank", "Feature", "Importance"]]
    st.dataframe(table_df, use_container_width=True, hide_index=True)


# =====================================================
# PAGE: Model Comparison
# =====================================================

elif page == "Model Comparison":
    st.markdown(
        '<div class="page-title">Model Comparison</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-subtitle">Baseline forecasting models compared against the full multivariate XGBoost model</div>',
        unsafe_allow_html=True
    )

    comparison_df = pd.DataFrame({
        "Model": [
            "ARIMA Baseline",
            "XGBoost Capacity-Only",
            "XGBoost Full Features"
        ],
        "RMSE": [
            267.57,
            22.11,
            13.52
        ],
        "MAE": [
            244.20,
            15.82,
            10.01
        ],
        "R²": [
            -26.20,
            0.5144,
            0.8247
        ]
    })

    best_model = comparison_df.loc[comparison_df["RMSE"].idxmin(), "Model"]
    best_rmse = comparison_df["RMSE"].min()
    best_mae = comparison_df.loc[comparison_df["RMSE"].idxmin(), "MAE"]

    c1, c2, c3 = st.columns(3)

    with c1:
        kpi("Best Model", best_model)

    with c2:
        kpi("Best RMSE", f"{best_rmse:.2f}", unit="cycles")

    with c3:
        kpi("Best MAE", f"{best_mae:.2f}", unit="cycles")

    st.markdown("""
        <div class="callout callout-success">
            The full-feature XGBoost model achieved the best forecasting performance.
            This shows that voltage, impedance, thermal, and discharge-duration features
            add significant predictive value beyond capacity-only degradation signals.
        </div>
    """, unsafe_allow_html=True)

    section("Performance Summary")

    display_df = comparison_df.copy()
    display_df["RMSE"] = display_df["RMSE"].round(2)
    display_df["MAE"] = display_df["MAE"].round(2)
    display_df["R²"] = display_df["R²"].round(4)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    section("RMSE Comparison")

    fig_rmse = go.Figure()

    fig_rmse.add_trace(go.Bar(
        x=comparison_df["Model"],
        y=comparison_df["RMSE"],
        marker=dict(
            color=[
                COLORS["red"],
                COLORS["amber"],
                COLORS["green"]
            ],
            line=dict(width=0)
        ),
        hovertemplate="<b>%{x}</b><br>RMSE: %{y:.2f} cycles<extra></extra>"
    ))

    fig_rmse.update_layout(
        xaxis_title=None,
        yaxis_title="RMSE, cycles",
        showlegend=False
    )

    style_fig(fig_rmse, height=380)
    st.plotly_chart(fig_rmse, use_container_width=True)

    section("MAE Comparison")

    fig_mae = go.Figure()

    fig_mae.add_trace(go.Bar(
        x=comparison_df["Model"],
        y=comparison_df["MAE"],
        marker=dict(
            color=[
                COLORS["red"],
                COLORS["amber"],
                COLORS["green"]
            ],
            line=dict(width=0)
        ),
        hovertemplate="<b>%{x}</b><br>MAE: %{y:.2f} cycles<extra></extra>"
    ))

    fig_mae.update_layout(
        xaxis_title=None,
        yaxis_title="MAE, cycles",
        showlegend=False
    )

    style_fig(fig_mae, height=380)
    st.plotly_chart(fig_mae, use_container_width=True)

    section("Modeling Insight")

    st.markdown("""
        <div class="callout">
            ARIMA performed poorly because it only models a univariate capacity-retention
            trajectory and struggles with noisy, heterogeneous, and censored battery lifecycles.
            The capacity-only XGBoost baseline improved performance by learning nonlinear
            degradation patterns. The full-feature XGBoost model performed best because it
            combines capacity, voltage, impedance, temperature, and discharge-duration signals.
        </div>
    """, unsafe_allow_html=True)

    csv = comparison_df.to_csv(index=False)

    st.download_button(
        label="Download Model Comparison CSV",
        data=csv,
        file_name="model_comparison.csv",
        mime="text/csv"
    )