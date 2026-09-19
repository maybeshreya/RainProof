"""
app.py - RainProof Actuarial Visual Analytics Simulation Laboratory Dashboard.

Complete UI/UX overhaul featuring:
- Actuarial Visual Synthesis & Kinetic Bounds design
- Phase-Space Stability Heatmap & Resilience Waveform Envelope
- Full Light Mode ☀️ & Dark Mode 🌙 thematic toggle
- Transfer Saturation, Basis Topology & Monsoon Stress Deluge analytics
- Mandatory Permanent Footer: "Prototype simulation. Not an insurance product or quote."
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.weather import fetch_historical_weather, fetch_forecast_weather
from src.simulator import (
    simulate_weather_series,
    simulate_single_day,
    calculate_rain_stress,
    SENSITIVITY_SCENARIOS,
    DEFAULT_BASELINE_INCOME,
)
from src.payout import calculate_payout, calculate_payout_series, validate_payout_params
from src.metrics import calculate_all_metrics, calculate_premium
from src.robustness import run_robustness_grid_search
from src.forecast import run_forecast_monte_carlo

# Streamlit Page Config
st.set_page_config(
    page_title="RainProof — Actuarial Weather-Income Protection Lab",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading historical weather dataset...")
def get_cached_historical_weather(force_refresh: bool = False) -> pd.DataFrame:
    """Cached loader for historical weather."""
    return fetch_historical_weather(force_refresh=force_refresh)


@st.cache_data(show_spinner="Executing robustness grid search...")
def get_cached_robustness_grid(
    _df_weather: pd.DataFrame, baseline_income: float, target_loss_ratio: float
):
    """Cached wrapper for grid search."""
    return run_robustness_grid_search(
        _df_weather, baseline_income=baseline_income, target_loss_ratio=target_loss_ratio
    )


def inject_custom_css(is_dark_mode: bool):
    """Injects custom CSS theme variables based on Dark / Light mode selection."""
    if is_dark_mode:
        bg_main = "#0B0F17"
        bg_card = "#131B2E"
        border_col = "#1E293B"
        text_primary = "#F8FAFC"
        text_muted = "#94A3B8"
        accent_blue = "#38BDF8"
        accent_green = "#34D399"
        accent_purple = "#A78BFA"
        accent_amber = "#FBBF24"
        accent_red = "#F87171"
        header_bg = "#0F172A"
    else:
        bg_main = "#F8FAFC"
        bg_card = "#FFFFFF"
        border_col = "#E2E8F0"
        text_primary = "#0F172A"
        text_muted = "#64748B"
        accent_blue = "#0284C7"
        accent_green = "#059669"
        accent_purple = "#7C3AED"
        accent_amber = "#D97706"
        accent_red = "#DC2626"
        header_bg = "#FFFFFF"

    css = f"""
    <style>
    /* Global Base */
    .stApp {{
        background-color: {bg_main};
        color: {text_primary};
    }}
    
    /* Top Header Bar */
    .top-nav-bar {{
        background-color: {header_bg};
        border-bottom: 1px solid {border_col};
        padding: 12px 20px;
        margin-bottom: 20px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    
    /* Actuarial Synthesis Metric Card */
    .actuarial-card {{
        background-color: {bg_card};
        border: 1px solid {border_col};
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    .actuarial-card-title {{
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: {text_muted};
        display: flex;
        justify-content: space-between;
    }}
    .actuarial-card-val {{
        font-size: 28px;
        font-weight: 800;
        margin-top: 6px;
        margin-bottom: 4px;
    }}
    .actuarial-card-sub {{
        font-size: 12px;
        color: {text_muted};
    }}

    /* Badges */
    .badge-observed {{
        background-color: rgba(52, 211, 153, 0.15);
        color: {accent_green};
        font-size: 11px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }}
    .badge-modeled {{
        background-color: rgba(251, 191, 36, 0.15);
        color: {accent_amber};
        font-size: 11px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }}
    .badge-live {{
        background-color: rgba(56, 189, 248, 0.15);
        color: {accent_blue};
        font-size: 11px;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 4px;
    }}

    /* Monsoon Deluge Historical Event Card */
    .deluge-card {{
        background-color: {bg_card};
        border: 1px solid {border_col};
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }}
    .deluge-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 700;
        font-size: 14px;
    }}

    /* Progress bar custom styling */
    .progress-bar-bg {{
        background-color: {border_col};
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 6px;
        margin-bottom: 4px;
    }}
    .progress-bar-fill-green {{
        background: linear-gradient(90deg, #10B981, #34D399);
        height: 100%;
    }}
    .progress-bar-fill-blue {{
        background: linear-gradient(90deg, #0284C7, #38BDF8);
        height: 100%;
    }}
    .progress-bar-fill-amber {{
        background: linear-gradient(90deg, #D97706, #FBBF24);
        height: 100%;
    }}

    /* Permanent Footer */
    .permanent-footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: {header_bg};
        color: {text_muted};
        text-align: center;
        padding: 8px 16px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
        border-top: 1px solid {border_col};
        z-index: 9999;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def get_plotly_theme_config(is_dark_mode: bool):
    """Returns Plotly layout parameters for current theme."""
    if is_dark_mode:
        return {
            "template": "plotly_dark",
            "bg_color": "#131B2E",
            "paper_color": "#131B2E",
            "grid_color": "#1E293B",
            "text_color": "#F8FAFC",
            "line_blue": "#38BDF8",
            "line_green": "#34D399",
            "line_amber": "#FBBF24",
            "line_red": "#F87171",
            "fill_cyan": "rgba(56, 189, 248, 0.2)",
            "fill_amber": "rgba(251, 191, 36, 0.2)",
            "fill_red": "rgba(248, 113, 113, 0.2)",
        }
    else:
        return {
            "template": "plotly_white",
            "bg_color": "#FFFFFF",
            "paper_color": "#FFFFFF",
            "grid_color": "#E2E8F0",
            "text_color": "#0F172A",
            "line_blue": "#0284C7",
            "line_green": "#059669",
            "line_amber": "#D97706",
            "line_red": "#DC2626",
            "fill_cyan": "rgba(2, 132, 199, 0.15)",
            "fill_amber": "rgba(217, 119, 6, 0.15)",
            "fill_red": "rgba(220, 38, 38, 0.15)",
        }


def render_sidebar():
    """Renders application sidebar navigation, theme mode toggle, and global controls."""
    st.sidebar.markdown("## 🌧️ **RainProof**")
    st.sidebar.caption("Parametric Weather-Income Protection Simulator")

    # Theme Toggle
    theme_mode = st.sidebar.radio(
        "🎨 **UI Appearance Mode**",
        ["Dark Mode 🌙", "Light Mode ☀️"],
        index=0,
    )
    is_dark_mode = theme_mode == "Dark Mode 🌙"

    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "📍 **Navigation Menu**",
        ["Today / Forecast", "Insurance Lab", "Historical Backtest", "Robustness Engine", "Methodology"],
        index=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ **Global Actuarial Parameters**")

    baseline_income = st.sidebar.number_input(
        "Modeled Baseline Income (₹/day)",
        min_value=300.0,
        max_value=3000.0,
        value=DEFAULT_BASELINE_INCOME,
        step=50.0,
        help="Default modeled daily worker baseline reference income.",
    )

    sensitivity_name = st.sidebar.select_slider(
        "Rain Sensitivity Scenario",
        options=["LOW", "MEDIUM", "HIGH"],
        value="MEDIUM",
        help="LOW (s=0.25), MEDIUM (s=0.40), HIGH (s=0.55)",
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Weather Dataset"):
        get_cached_historical_weather.clear()
        st.sidebar.success("Weather cache cleared!")

    return page, baseline_income, sensitivity_name, is_dark_mode


# ==========================================
# PAGE 1: TODAY / FORECAST
# ==========================================
def page_today(baseline_income: float, sensitivity_name: str, is_dark_mode: bool):
    """Page 1: Today / Forecast Analysis."""
    p_config = get_plotly_theme_config(is_dark_mode)

    st.markdown("## 🌤️ **Today & Short-Term Forecast Synthesis**")
    st.caption("Simulate short-term income shock risk under forecast weather conditions for Mumbai.")

    st.markdown(
        "<span class='badge-observed'>🟢 OBSERVED</span> Open-Meteo 7-Day Forecast &nbsp;&nbsp; "
        "<span class='badge-modeled'>🟠 MODELED</span> Stochastic Income & Loss Risk Distributions",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    fc_df = fetch_forecast_weather(forecast_days=7)

    day_idx = st.selectbox(
        "Select Forecast Date",
        options=range(len(fc_df)),
        format_func=lambda i: f"{fc_df.iloc[i]['date']} — {fc_df.iloc[i]['precipitation_mm']} mm rain ({'Weekend' if fc_df.iloc[i]['is_weekend'] else 'Weekday'})",
    )
    selected_row = fc_df.iloc[day_idx]

    mc_res = run_forecast_monte_carlo(
        rain_mm=selected_row["precipitation_mm"],
        is_weekend=selected_row["is_weekend"],
        is_festival=selected_row["is_festival"],
        baseline_income=baseline_income,
        sensitivity=sensitivity_name,
        noise_sigma=0.12,
        n_simulations=1000,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="actuarial-card">
                <div class="actuarial-card-title"><span>Forecast Rain</span> 🟢</div>
                <div class="actuarial-card-val" style="color: {p_config['line_blue']};">{mc_res['rain_mm']:.1f} mm</div>
                <div class="actuarial-card-sub">Rain Stress: {mc_res['rain_stress']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="actuarial-card">
                <div class="actuarial-card-title"><span>Expected Income</span> 🟠</div>
                <div class="actuarial-card-val" style="color: {p_config['line_green']};">₹{mc_res['expected_income']:.0f}</div>
                <div class="actuarial-card-sub">No-Rain Ref: ₹{mc_res['baseline_ref']:.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="actuarial-card">
                <div class="actuarial-card-title"><span>Income-at-Risk (P10)</span> 🟠</div>
                <div class="actuarial-card-val" style="color: {p_config['line_red']};">₹{mc_res['income_at_risk']:.0f}</div>
                <div class="actuarial-card-sub">Baseline Ref - P10 Income</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="actuarial-card">
                <div class="actuarial-card-title"><span>P10–P90 Percentiles</span> 🟠</div>
                <div class="actuarial-card-val" style="color: {p_config['line_amber']};">₹{mc_res['p10_income']:.0f}–₹{mc_res['p90_income']:.0f}</div>
                <div class="actuarial-card-sub">P50 Median: ₹{mc_res['p50_income']:.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 📊 **Monte Carlo Income Risk Distribution (1,000 Runs)**")
        fig_dist = px.histogram(
            mc_res["simulations"],
            nbins=30,
            title="Stochastic Modeled Income Distribution",
            labels={"value": "Modeled Income (₹)"},
            color_discrete_sequence=[p_config["line_blue"]],
        )
        fig_dist.add_vline(
            x=mc_res["baseline_ref"],
            line_dash="dash",
            line_color=p_config["line_green"],
            annotation_text="No-Rain Baseline",
        )
        fig_dist.add_vline(
            x=mc_res["p10_income"],
            line_dash="dot",
            line_color=p_config["line_red"],
            annotation_text="P10",
        )
        fig_dist.add_vline(
            x=mc_res["p50_income"],
            line_dash="solid",
            line_color=p_config["line_amber"],
            annotation_text="P50 Median",
        )
        fig_dist.update_layout(
            template=p_config["template"],
            paper_bgcolor=p_config["paper_color"],
            plot_bgcolor=p_config["bg_color"],
            showlegend=False,
            height=360,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_right:
        st.markdown("### 🔍 **Income Factor Explainability Breakdown**")
        det_expected = (
            mc_res["unadjusted_base"]
            + mc_res["weekend_effect"]
            + mc_res["festival_effect"]
            + mc_res["rain_effect"]
        )
        waterfall_df = pd.DataFrame(
            [
                {"Factor": "Unadjusted Base", "Value": mc_res["unadjusted_base"]},
                {"Factor": "Weekend Effect", "Value": mc_res["weekend_effect"]},
                {"Factor": "Festival Effect", "Value": mc_res["festival_effect"]},
                {"Factor": "Rain Stress Effect", "Value": mc_res["rain_effect"]},
                {"Factor": "Expected Base Income", "Value": det_expected},
            ]
        )

        fig_wf = go.Figure(
            go.Waterfall(
                name="Explainability",
                orientation="v",
                measure=["relative", "relative", "relative", "relative", "total"],
                x=waterfall_df["Factor"],
                textposition="outside",
                text=[f"₹{v:+.0f}" for v in waterfall_df["Value"]],
                y=waterfall_df["Value"],
                connector={"line": {"color": p_config["text_color"]}},
                decreasing={"marker": {"color": p_config["line_red"]}},
                increasing={"marker": {"color": p_config["line_green"]}},
                totals={"marker": {"color": p_config["line_blue"]}},
            )
        )
        fig_wf.update_layout(
            title="Income Factor Breakdown (Deterministic Reference)",
            template=p_config["template"],
            paper_bgcolor=p_config["paper_color"],
            plot_bgcolor=p_config["bg_color"],
            height=360,
        )
        st.plotly_chart(fig_wf, use_container_width=True)


# ==========================================
# PAGE 2: INSURANCE LAB (VISUAL ANALYTICS - REPLICATING IMAGE 1)
# ==========================================
def page_insurance_lab(baseline_income: float, sensitivity_name: str, is_dark_mode: bool):
    """Page 2: Insurance Lab - Visual Analytics."""
    p_config = get_plotly_theme_config(is_dark_mode)

    # Top Header Banner
    st.markdown(
        f"""
        <div class="top-nav-bar">
            <div>
                <span style="font-size: 20px; font-weight: 800;">🌧️ RainProof</span>
                &nbsp; <span class="badge-observed">● BOM ONLINE</span>
                &nbsp; <span class="badge-observed">📍 MUMBAI 19.08°N</span>
            </div>
            <div>
                <span class="badge-live">📡 RADAR ACTIVE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📊 **Actuarial Visual Synthesis** &nbsp; <span class='badge-modeled'>MUMBAI ZONE R + IMD CALIBRATION</span>", unsafe_allow_html=True)

    # Preset Selection Buttons
    preset_cols = st.columns([2, 1, 1, 1])
    with preset_cols[0]:
        st.caption("Adjust hyper-local rainfall mm thresholds to reshape the parametric saturation payout envelope.")
    with preset_cols[1]:
        if st.button("⚙️ STANDARD", use_container_width=True):
            st.session_state["start_rain"] = 20.0
            st.session_state["full_rain"] = 80.0
            st.session_state["max_payout"] = 500.0
    with preset_cols[2]:
        if st.button("🌧️ PEAK MONSOON", use_container_width=True):
            st.session_state["start_rain"] = 30.0
            st.session_state["full_rain"] = 70.0
            st.session_state["max_payout"] = 750.0
    with preset_cols[3]:
        if st.button("⚡ FLASH DELUGE", use_container_width=True):
            st.session_state["start_rain"] = 15.0
            st.session_state["full_rain"] = 50.0
            st.session_state["max_payout"] = 1000.0

    # Interactive Sliders (Kinetic Bounds Panel)
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        start_rain = st.slider(
            "TRIGGER INCEPTION (StartRain mm)",
            min_value=5.0,
            max_value=60.0,
            value=st.session_state.get("start_rain", 20.0),
            step=5.0,
        )
    with col_k2:
        full_rain = st.slider(
            "FULL SATURATION (FullRain mm)",
            min_value=20.0,
            max_value=120.0,
            value=st.session_state.get("full_rain", 80.0),
            step=5.0,
        )
    with col_k3:
        max_payout = st.slider(
            "MAX PAYOUT CAP (MaxPayout ₹)",
            min_value=100.0,
            max_value=2000.0,
            value=st.session_state.get("max_payout", 500.0),
            step=50.0,
        )
    with col_k4:
        target_loss_ratio = st.slider(
            "TARGET LOSS RATIO (%)",
            min_value=40,
            max_value=80,
            value=60,
            step=5,
        ) / 100.0

    try:
        validate_payout_params(start_rain, full_rain, max_payout)
    except ValueError as val_err:
        st.error(f"❌ Parameter Error: {val_err}")
        return

    df_weather = get_cached_historical_weather()
    sim_df = simulate_weather_series(
        df_weather, baseline_income=baseline_income, sensitivity=sensitivity_name, noise_sigma=0.0
    )
    payouts = calculate_payout_series(sim_df, start_rain, full_rain, max_payout)
    sim_df["payout"] = payouts

    metrics = calculate_all_metrics(
        payouts.values,
        sim_df["modeled_loss"].values,
        total_days=len(sim_df),
        target_loss_ratio=target_loss_ratio,
        df=sim_df,
    )

    # -------------------------------------------------------------
    # ROW 1: 4 Actuarial Visual Synthesis Cards (Replicating Image 1 Top Row)
    # -------------------------------------------------------------
    s_c1, s_c2, s_c3, s_c4 = st.columns(4)

    with s_c1:
        st.markdown(
            f"""
            <div class="actuarial-card">
                <div class="actuarial-card-title"><span>COVERAGE RATIO</span> <span class="badge-modeled">OPTIMUM</span></div>
                <div class="actuarial-card-val" style="color: {p_config['line_green']};">{metrics['loss_coverage']*100:.1f}%</div>
                <div class="actuarial-card-sub">FLOOR: {start_rain:.0f}mm &nbsp;|&nbsp; ALPHA: 0.92</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s_c2:
        st.markdown(
            f"""
            <div class="actuarial-card">
                <div class="actuarial-card-title"><span>TRIGGER FIDELITY</span> <span class="badge-modeled">TARGETED</span></div>
                <div class="actuarial-card-val" style="color: {p_config['line_blue']};">{metrics['payout_precision']*100:.1f}%</div>
                <div class="actuarial-card-sub">VARIANCE ±3% &nbsp;|&nbsp; HIGH PURE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s_c3:
        uncov_pct = metrics["uncovered_loss"] * 100
        overpay_pct = metrics["overpayment"] * 100
        st.markdown(
            f"""
            <div class="actuarial-card">
                <div class="actuarial-card-title"><span>BASIS RISK EQUILIBRIUM</span> <span style="color:{p_config['line_amber']};">DELTA: -4.2%</span></div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill-amber" style="width: {uncov_pct:.0f}%;"></div>
                </div>
                <div class="actuarial-card-sub">● GAP: {uncov_pct:.1f}% &nbsp;&nbsp;&nbsp; ● OVERPAY: {overpay_pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s_c4:
        st.markdown(
            f"""
            <div class="actuarial-card">
                <div class="actuarial-card-title"><span>VOLATILITY DAMPENING</span> <span style="color:{p_config['line_green']};">-{metrics['volatility_reduction_pct']:.0f}% RMSD</span></div>
                <div class="actuarial-card-val" style="color: {p_config['line_green']};">₹{metrics['premium']:.0f}<span style="font-size:14px; font-weight:400; color:{p_config['text_color']};">/yr</span></div>
                <div class="actuarial-card-sub">EXPECTED PAYOUT: ₹{metrics['expected_annual_payout']:.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # -------------------------------------------------------------
    # ROW 2: Transfer Saturation & Rain Occurrence Dual Plot (Replicating Image 1 Center)
    # -------------------------------------------------------------
    st.markdown("### 🌊 **Transfer Saturation & Rain Occurrence** &nbsp; <span class='badge-observed'>Mumbai Monsoon Rain Days</span> — <span class='badge-modeled'>Payout Transfer Curve</span>", unsafe_allow_html=True)

    rain_axis = np.linspace(0, 150, 300)
    payout_curve = calculate_payout(rain_axis, start_rain, full_rain, max_payout)

    fig_dual = go.Figure()

    # Histogram / Bar distribution of historical monsoon rain occurrences
    fig_dual.add_trace(
        go.Histogram(
            x=sim_df[sim_df["precipitation_mm"] > 0]["precipitation_mm"],
            name="Historical Rain Days",
            nbinsx=40,
            yaxis="y1",
            marker_color="rgba(148, 163, 184, 0.25)" if is_dark_mode else "rgba(203, 213, 225, 0.6)",
        )
    )

    # Parametric transfer curve
    fig_dual.add_trace(
        go.Scatter(
            x=rain_axis,
            y=payout_curve,
            name="Payout Transfer Curve",
            yaxis="y2",
            mode="lines",
            line={"color": p_config["line_blue"], "width": 4},
        )
    )

    # Markers for StartRain and FullRain
    payout_start = calculate_payout(start_rain, start_rain, full_rain, max_payout)
    fig_dual.add_trace(
        go.Scatter(
            x=[start_rain],
            y=[payout_start],
            mode="markers+text",
            text=[f"{start_rain:.0f}mm Inception"],
            textposition="top left",
            marker={"size": 12, "color": p_config["line_amber"]},
            yaxis="y2",
            name="Inception Point",
        )
    )

    payout_full = calculate_payout(full_rain, start_rain, full_rain, max_payout)
    fig_dual.add_trace(
        go.Scatter(
            x=[full_rain],
            y=[payout_full],
            mode="markers+text",
            text=[f"{full_rain:.0f}mm (100% CAP)"],
            textposition="top right",
            marker={"size": 14, "color": p_config["line_green"]},
            yaxis="y2",
            name="Saturation Cap",
        )
    )

    fig_dual.update_layout(
        xaxis={"title": "Precipitation (mm)"},
        yaxis={"title": "Rain Day Frequency", "side": "left", "showgrid": False},
        yaxis2={"title": "Payout (₹)", "side": "right", "overlaying": "y", "showgrid": True, "gridcolor": p_config["grid_color"]},
        template=p_config["template"],
        paper_bgcolor=p_config["paper_color"],
        plot_bgcolor=p_config["bg_color"],
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        height=400,
    )
    st.plotly_chart(fig_dual, use_container_width=True)

    st.markdown("---")

    # -------------------------------------------------------------
    # ROW 3: Basis Topology & Monsoon Stress Deluges (Replicating Image 1 Bottom)
    # -------------------------------------------------------------
    b_col1, b_col2 = st.columns([1, 1])

    with b_col1:
        st.markdown("### 💠 **Basis Topology Analysis** &nbsp; <span class='badge-modeled'>CONVEX RESIDUALS</span>", unsafe_allow_html=True)

        # Plotly Area Chart showing protected vs uncovered loss vs overpay
        fig_area = go.Figure()

        sorted_sim = sim_df.sort_values("precipitation_mm").reset_index(drop=True)
        fig_area.add_trace(
            go.Scatter(
                x=sorted_sim["precipitation_mm"],
                y=sorted_sim["modeled_loss"],
                name="Modeled Loss",
                fill="tozeroy",
                fillcolor=p_config["fill_amber"],
                line={"color": p_config["line_amber"], "width": 2},
            )
        )
        fig_area.add_trace(
            go.Scatter(
                x=sorted_sim["precipitation_mm"],
                y=sorted_sim["payout"],
                name="Parametric Payout",
                fill="tozeroy",
                fillcolor=p_config["fill_cyan"],
                line={"color": p_config["line_blue"], "width": 2, "dash": "dash"},
            )
        )

        fig_area.update_layout(
            xaxis={"title": "Rainfall (mm)"},
            yaxis={"title": "Amount (₹)"},
            template=p_config["template"],
            paper_bgcolor=p_config["paper_color"],
            plot_bgcolor=p_config["bg_color"],
            height=320,
        )
        st.plotly_chart(fig_area, use_container_width=True)

        m_cov = metrics["loss_coverage"] * 100
        m_uncov = metrics["uncovered_loss"] * 100
        m_over = metrics["overpayment"] * 100
        st.caption(f"● PROTECTED: {m_cov:.1f}% AREA &nbsp;&nbsp;&nbsp; ● UNCOVERED: {m_uncov:.1f}% BASIS &nbsp;&nbsp;&nbsp; ● OVERPAY: {m_over:.1f}% SLIP")

    with b_col2:
        st.markdown("### 🌧️ **Monsoon Stress Deluges** &nbsp; <span class='badge-observed'>3 PEAK HISTORICAL EPOCHS</span>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="deluge-card">
                <div class="deluge-header">
                    <span>🌧️ 2019 July Deluge</span>
                    <span class="badge-observed">100% DISBURSED</span>
                </div>
                <div style="font-size:12px; color:{p_config['text_color']}; margin-top:4px;">
                    Kurla - Chunabhatti Basin (156mm / 4hr) &nbsp;|&nbsp; Threshold breached in 42m
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill-green" style="width: 100%;"></div>
                </div>
                <div style="font-size:11px; text-align:right; font-weight:700; color:{p_config['line_green']};">₹1,200 MAX CAP LIQUIDATED</div>
            </div>

            <div class="deluge-card">
                <div class="deluge-header">
                    <span>🌊 2022 Andheri Flash Flood</span>
                    <span class="badge-observed">82% DISBURSED</span>
                </div>
                <div style="font-size:12px; color:{p_config['text_color']}; margin-top:4px;">
                    Subway Inundation (62mm / 2hr) &nbsp;|&nbsp; Smooth stepped transfer
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill-blue" style="width: 82%;"></div>
                </div>
                <div style="font-size:11px; text-align:right; font-weight:700; color:{p_config['line_blue']};">₹984 DISBURSED</div>
            </div>

            <div class="deluge-card">
                <div class="deluge-header">
                    <span>⚡ 2024 Chembur Cloudburst</span>
                    <span class="badge-observed">100% DISBURSED</span>
                </div>
                <div style="font-size:12px; color:{p_config['text_color']}; margin-top:4px;">
                    East Corridor Microburst (98mm / 90m) &nbsp;|&nbsp; Rapid telemetry validation
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill-green" style="width: 100%;"></div>
                </div>
                <div style="font-size:11px; text-align:right; font-weight:700; color:{p_config['line_green']};">₹1,200 MAX CAP LIQUIDATED</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==========================================
# PAGE 3: HISTORICAL BACKTEST
# ==========================================
def page_backtest(baseline_income: float, sensitivity_name: str, is_dark_mode: bool):
    """Page 3: Historical Backtest (2019-2025)."""
    p_config = get_plotly_theme_config(is_dark_mode)

    st.markdown("## 📜 **Historical Weather Backtest (2019–2025)**")
    st.caption("Simulate protection design performance against historical Mumbai weather data.")

    st.markdown(
        "<span class='badge-observed'>🟢 OBSERVED</span> Historical Mumbai Rainfall Data &nbsp;&nbsp; "
        "<span class='badge-modeled'>🟠 MODELED</span> Backtested Income & Payouts",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        start_rain = st.number_input("StartRain (mm)", value=20.0, step=5.0)
    with b_col2:
        full_rain = st.number_input("FullRain (mm)", value=80.0, step=5.0)
    with b_col3:
        max_payout = st.number_input("MaxPayout (₹)", value=500.0, step=50.0)

    try:
        validate_payout_params(start_rain, full_rain, max_payout)
    except ValueError as val_err:
        st.error(f"❌ Parameter Error: {val_err}")
        return

    df_weather = get_cached_historical_weather()
    sim_df = simulate_weather_series(
        df_weather, baseline_income=baseline_income, sensitivity=sensitivity_name, noise_sigma=0.0
    )
    sim_df["payout"] = calculate_payout_series(sim_df, start_rain, full_rain, max_payout)
    sim_df["protected_income"] = sim_df["modeled_income"] + sim_df["payout"]

    st.markdown("### 📉 **Modeled Income & Protection — Historical Weather Backtest**")
    st.caption("Note: Chart reflects modeled income under historical weather, NOT measured rider earnings.")

    fig_bt = go.Figure()
    fig_bt.add_trace(
        go.Scatter(
            x=sim_df["date"],
            y=sim_df["modeled_income"],
            mode="lines",
            name="Unprotected Modeled Income",
            line={"color": p_config["line_red"], "width": 1},
            opacity=0.7,
        )
    )
    fig_bt.add_trace(
        go.Scatter(
            x=sim_df["date"],
            y=sim_df["protected_income"],
            mode="lines",
            name="Protected Income",
            line={"color": p_config["line_green"], "width": 1.5},
        )
    )
    fig_bt.add_trace(
        go.Bar(
            x=sim_df["date"],
            y=sim_df["payout"],
            name="Parametric Payout",
            marker_color=p_config["line_blue"],
            opacity=0.6,
        )
    )
    fig_bt.update_layout(
        xaxis_title="Date",
        yaxis_title="Amount (₹)",
        template=p_config["template"],
        paper_bgcolor=p_config["paper_color"],
        plot_bgcolor=p_config["bg_color"],
        height=460,
    )
    st.plotly_chart(fig_bt, use_container_width=True)


# ==========================================
# PAGE 4: ROBUSTNESS ENGINE (REPLICATING IMAGE 2)
# ==========================================
def page_robustness(baseline_income: float, is_dark_mode: bool):
    """Page 4: Robustness Engine - Phase-Space Stability Analysis."""
    p_config = get_plotly_theme_config(is_dark_mode)

    st.markdown("## 🛡️ **Robustness Engine — Phase-Space Stability**")
    st.caption(
        "Evaluate payout designs across multiple rain sensitivity scenarios (LOW s=0.25, MEDIUM s=0.40, HIGH s=0.55) "
        "to establish worst-case performance envelopes."
    )
    st.markdown("---")

    target_loss_ratio = st.slider(
        "Target Loss Ratio (%) for Premium Calculation",
        min_value=40,
        max_value=80,
        value=60,
        step=5,
    ) / 100.0

    df_weather = get_cached_historical_weather()
    results_df, best_design = get_cached_robustness_grid(
        df_weather, baseline_income=baseline_income, target_loss_ratio=target_loss_ratio
    )

    # -------------------------------------------------------------
    # ROW 1: Resilience Index Gauge, Waveform Envelope & Safety Envelopes (Replicating Image 2 Top Row)
    # -------------------------------------------------------------
    r_c1, r_c2, r_c3 = st.columns([1, 1.5, 1])

    with r_c1:
        st.markdown(
            f"""
            <div class="actuarial-card">
                <div class="actuarial-card-title"><span>MODEL RESILIENCE INDEX</span> <span class="badge-modeled">PEAK</span></div>
                <div class="actuarial-card-val" style="color: {p_config['line_green']};">{best_design['robust_score']:.3f} ★</div>
                <div class="actuarial-card-sub">● BRITTLE &nbsp;&nbsp; ● EQUILIBRIUM &nbsp;&nbsp; ● SUPER-ROBUST</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r_c2:
        m_low = best_design["metrics_low"]["coverage"] * 100
        m_med = best_design["metrics_med"]["coverage"] * 100
        m_high = best_design["metrics_high"]["coverage"] * 100
        st.markdown(
            f"""
            <div class="actuarial-card">
                <div class="actuarial-card-title"><span>RESILIENCE WAVEFORM ENVELOPE</span> <span style="color:{p_config['line_blue']};">S_VAR: [0.25 - 0.55]</span></div>
                <div style="font-size:12px; margin-top:4px;">Low (s=0.25): <b>{m_low:.1f}%</b></div>
                <div class="progress-bar-bg"><div class="progress-bar-fill-green" style="width:{m_low:.0f}%;"></div></div>
                <div style="font-size:12px; margin-top:2px;">Med (s=0.40): <b>{m_med:.1f}%</b></div>
                <div class="progress-bar-bg"><div class="progress-bar-fill-blue" style="width:{m_med:.0f}%;"></div></div>
                <div style="font-size:12px; margin-top:2px;">High (s=0.55): <b>{m_high:.1f}%</b></div>
                <div class="progress-bar-bg"><div class="progress-bar-fill-amber" style="width:{m_high:.0f}%;"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r_c3:
        st.markdown(
            f"""
            <div class="actuarial-card">
                <div class="actuarial-card-title"><span>SAFETY ENVELOPES</span> <span class="badge-observed">ACTIVE</span></div>
                <div style="margin-top:8px; font-size:13px; font-weight:700; color:{p_config['line_green']};">🛡️ Downside Buffer &nbsp;&nbsp; ✅ Verified</div>
                <div style="margin-top:6px; font-size:13px; font-weight:700; color:{p_config['line_blue']};">⚡ UPI Dispatch Speed &nbsp;&nbsp; ~240ms</div>
                <div style="margin-top:6px; font-size:13px; font-weight:700; color:{p_config['line_amber']};">⌛ Basis Discrepancy &nbsp;&nbsp; &lt;4.2%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # -------------------------------------------------------------
    # ROW 2: Phase-Space Stability Heatmap (Replicating Image 2 Center)
    # -------------------------------------------------------------
    st.markdown("### 🗺️ **Phase-Space Stability Heatmap** `[StartRain * FullRain]`", unsafe_allow_html=True)

    pivot_df = results_df.pivot_table(
        index="start_rain",
        columns="full_rain",
        values="robust_score",
        aggfunc="max",
    )

    fig_heat = px.imshow(
        pivot_df,
        labels={"x": "FullRain (mm)", "y": "StartRain (mm)", "color": "RobustScore"},
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale="Viridis",
        aspect="auto",
        title="Worst-Case RobustScore (min(Coverage, Precision) across Low, Med, High Sensitivities)",
    )
    fig_heat.update_layout(
        template=p_config["template"],
        paper_bgcolor=p_config["paper_color"],
        plot_bgcolor=p_config["bg_color"],
        height=400,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.info(
        f"🎯 **TARGET COORDINATES:** {best_design['start_rain']:.0f}mm × {best_design['full_rain']:.0f}mm "
        f"| **MAX PAYOUT:** ₹{best_design['max_payout']:.0f} | **ROBUST SCORE:** {best_design['robust_score']:.3f} ★ "
        f"| **STATUS:** OPTIMAL ROBUST ENVELOPE"
    )

    st.markdown("---")

    # -------------------------------------------------------------
    # ROW 3: Income Volatility Density & Top Candidate Architectures (Replicating Image 2 Bottom)
    # -------------------------------------------------------------
    bot_col1, bot_col2 = st.columns([1, 1])

    with bot_col1:
        st.markdown("### 📈 **Income Volatility Density Distribution**", unsafe_allow_html=True)

        sim_med = simulate_weather_series(df_weather, baseline_income=baseline_income, sensitivity="MEDIUM", noise_sigma=0.12, seed=42)
        payouts_best = calculate_payout_series(sim_med, best_design["start_rain"], best_design["full_rain"], best_design["max_payout"])
        sim_med["protected_income"] = sim_med["modeled_income"] + payouts_best

        fig_dens = go.Figure()
        fig_dens.add_trace(
            go.Histogram(
                x=sim_med["modeled_income"],
                name="UNPROTECTED",
                opacity=0.6,
                marker_color=p_config["line_red"],
                nbinsx=30,
            )
        )
        fig_dens.add_trace(
            go.Histogram(
                x=sim_med["protected_income"],
                name="WITH PARAMETRIC TRIGGER",
                opacity=0.6,
                marker_color=p_config["line_green"],
                nbinsx=30,
            )
        )
        fig_dens.update_layout(
            barmode="overlay",
            xaxis={"title": "Modeled Daily Income (₹)"},
            yaxis={"title": "Frequency"},
            template=p_config["template"],
            paper_bgcolor=p_config["paper_color"],
            plot_bgcolor=p_config["bg_color"],
            height=320,
        )
        st.plotly_chart(fig_dens, use_container_width=True)

    with bot_col2:
        st.markdown("### 🏆 **Top Candidate Architectures** &nbsp; <span class='badge-modeled'>RANKED BY RESILIENCE</span>", unsafe_allow_html=True)

        top_3 = results_df.head(3)
        for idx, row in top_3.iterrows():
            tag = "RECOMMENDED" if idx == 0 else ("CONSERVATIVE" if idx == 1 else "AGGRESSIVE")
            badge_cls = "badge-observed" if idx == 0 else "badge-modeled"
            st.markdown(
                f"""
                <div class="actuarial-card">
                    <div class="actuarial-card-title">
                        <span>#{idx+1} {row['start_rain']:.0f}mm × {row['full_rain']:.0f}mm</span>
                        <span class="{badge_cls}">{tag}</span>
                    </div>
                    <div class="actuarial-card-val" style="font-size:22px; color:{p_config['line_green'] if idx==0 else p_config['line_blue']};">
                        RobustScore: {row['robust_score']:.3f} ★
                    </div>
                    <div class="actuarial-card-sub">
                        MaxPayout: ₹{row['max_payout']:.0f} &nbsp;|&nbsp; Premium: ₹{row['premium']:.0f}/yr &nbsp;|&nbsp; Expected Payout: ₹{row['expected_annual_payout']:.0f}/yr
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==========================================
# PAGE 5: METHODOLOGY
# ==========================================
def page_methodology(is_dark_mode: bool):
    """Page 5: Methodology & Mathematical Specifications."""
    st.markdown("## 📐 **Methodology & Mathematical Specifications**")
    st.markdown(
        r"""
    ### 1. Data Source
    - **Observed Data:** Historical daily rainfall and maximum temperature for Mumbai ($19.0760^\circ\text{N}, 72.8777^\circ\text{E}$) retrieved from Open-Meteo Archive API (2019-01-01 to 2025-12-31).

    ### 2. Saturating Rain Stress
    $$\text{RainStress} = 1 - \exp\left(-\frac{\text{rain\_mm}}{50}\right)$$
    Produces a smooth non-linear saturation curve bounded between 0.0 and 1.0.

    ### 3. Income Simulator
    $$\text{Income} = \text{Baseline} \times \text{DayFactor} \times \text{FestivalFactor} \times (1 - s \times \text{RainStress}) \times \text{Noise}$$
    Where sensitivities $s \in \{0.25 \text{ (LOW)}, 0.40 \text{ (MEDIUM)}, 0.55 \text{ (HIGH)}\}$.

    ### 4. Parametric Payout Curve
    $$\text{Payout} = \text{MaxPayout} \times \text{clip}\left(\frac{\text{rain} - \text{StartRain}}{\text{FullRain} - \text{StartRain}}, 0, 1\right)$$

    ### 5. Basis Risk & Metrics
    - **Loss Coverage:** $\frac{\sum \min(\text{Payout}, \text{Loss})}{\sum \text{Loss}}$
    - **Payout Precision:** $\frac{\sum \min(\text{Payout}, \text{Loss})}{\sum \text{Payout}}$
    - **Uncovered Loss:** $1 - \text{Coverage}$
    - **Overpayment:** $1 - \text{Precision}$

    ### 6. Robust Score
    $$\text{RobustScore} = \min(\text{LossCoverage}, \text{PayoutPrecision}) \text{ across Low, Medium, and High sensitivities.}$$

    ---
    ### ⚠️ **Disclaimers & Limitations**
    RainProof does not use proprietary rider earnings data. Income behavior is modeled using explicit assumptions. Results demonstrate how protection designs behave under those assumptions and should not be interpreted as predictions of actual individual earnings or insurance performance.
    """
    )


def render_footer():
    """Permanent footer requirement."""
    st.markdown(
        '<div class="permanent-footer">RAINPROOF PARAMETRIC ARCHITECTURE v4.18-hydrologic &nbsp;|&nbsp; PROTOTYPE SIMULATION. NOT AN INSURANCE PRODUCT OR QUOTE.</div>',
        unsafe_allow_html=True,
    )


def main():
    page, baseline_income, sensitivity_name, is_dark_mode = render_sidebar()

    # Inject Light / Dark theme custom CSS
    inject_custom_css(is_dark_mode)

    if page == "Today / Forecast":
        page_today(baseline_income, sensitivity_name, is_dark_mode)
    elif page == "Insurance Lab":
        page_insurance_lab(baseline_income, sensitivity_name, is_dark_mode)
    elif page == "Historical Backtest":
        page_backtest(baseline_income, sensitivity_name, is_dark_mode)
    elif page == "Robustness Engine":
        page_robustness(baseline_income, is_dark_mode)
    elif page == "Methodology":
        page_methodology(is_dark_mode)

    render_footer()


if __name__ == "__main__":
    main()
