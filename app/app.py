"""
VayuDrishti: Air Quality Forecasting and Automated Alert System
Streamlit Interactive Web Application
Author: Khushi Yadav (@khushiyadav35)
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st

from config.settings import MONITORED_CITIES, ALERT_THRESHOLDS, MODELS_DIR
from config.naqi_breakpoints import NAQI_CATEGORIES, CPCB_BREAKPOINTS
from src.data.ingestion import DataIngestionPipeline
from src.preprocessing.cpcb_calculator import CPCBCalculator
from src.preprocessing.feature_engineering import TimeSeriesFeatureEngineer
from src.models.lstm_forecaster import LSTMTrainer
from src.models.prophet_forecaster import AdditiveProphetForecaster
from src.models.evaluator import ForecastingEvaluator
from src.alerts.alert_manager import AlertManager
from src.alerts.telegram_bot import TelegramAlertService
from app.components.charts import create_forecast_chart, create_pollutant_breakdown_chart

# Page Configuration
st.set_page_config(
    page_title="VayuDrishti | Air Quality Forecasting & Alert System",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 20px;
    }
    .aqi-card {
        padding: 24px;
        border-radius: 12px;
        color: #FFFFFF;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 16px;
    }
    .metric-container {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/fluency/96/air-quality.png", width=64)
st.sidebar.title("🎛️ Control Center")
st.sidebar.caption("Air Quality Forecasting System (CPCB / OpenAQ)")

selected_city = st.sidebar.selectbox(
    "📍 Select Focus Megacity",
    options=list(MONITORED_CITIES.keys()),
    index=0,
    help="Delhi and Lucknow face seasonal winter inversions and hazardous particulate spikes.",
)

city_meta = MONITORED_CITIES[selected_city]

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Forecasting Engine")
selected_model = st.sidebar.radio(
    "Select Forecasting Architecture",
    options=["PyTorch Deep LSTM", "Prophet / Additive Model"],
    index=0,
    help="PyTorch LSTM performs multi-step sequence modeling; Prophet captures Fourier diurnal/seasonal harmonics.",
)

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (Hours)",
    min_value=24,
    max_value=72,
    value=24,
    step=24,
    help="Project atmospheric air quality index 24h, 48h, or 72h into the future.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Alert Configuration")
custom_threshold = st.sidebar.slider(
    "Alert Trigger Threshold (NAQI)",
    min_value=100,
    max_value=450,
    value=201,
    step=25,
    help="Threshold for triggering Telegram Bot and Email notifications (201 = Poor).",
)

st.sidebar.markdown(
    """
    <div style="font-size: 0.82rem; color: #64748B; margin-top: 20px;">
    <b>Project:</b> B.Tech Capstone Engineering Project<br>
    <b>Author:</b> Khushi Yadav (<a href="https://github.com/khushiyadav35" target="_blank">@khushiyadav35</a>)<br>
    <b>Standard:</b> Indian CPCB NAQI (2014)
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=600)
def get_city_data(city: str) -> pd.DataFrame:
    """Fetches atmospheric observations and computes CPCB NAQI."""
    pipeline = DataIngestionPipeline(city)
    df = pipeline.fetch_live_hourly(past_days=7, forecast_days=3)
    df_naqi = CPCBCalculator.compute_dataframe_naqi(df)
    return df_naqi


# Load Data
with st.spinner(f"Retrieving atmospheric observations for {selected_city}..."):
    data_df = get_city_data(selected_city)

latest_row = data_df.iloc[-1]
current_aqi = int(latest_row["cpcb_aqi"])
category_meta = CPCBCalculator.get_category_metadata(current_aqi)

# Header Section
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown('<div class="main-title">🍃 VayuDrishti: Air Quality Forecasting & Alert System</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-title">Continuous CPCB / OpenAQ Monitoring, 24h–72h Recurrent Forecasting, and Multi-Channel Alert Engine for <b>{selected_city}</b> ({city_meta["state"]})</div>',
        unsafe_allow_html=True,
    )
with col_head2:
    st.markdown(
        f"""
        <div style="text-align: right; margin-top: 10px;">
            <span style="background-color: #E2E8F0; padding: 6px 12px; border-radius: 6px; font-size: 13px; font-weight: 600;">
                🕒 {latest_row['timestamp'].strftime('%d %b %Y, %I:%M %p')}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Tab Navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Real-Time NAQI Monitor",
    "📈 24h–72h Forecasting Studio",
    "🚨 Automated Alert Dispatcher",
    "🗺️ Geospatial & Station Grid",
    "📖 CPCB Scientific Framework",
])


# ==========================================
# TAB 1: REAL-TIME NAQI MONITOR
# ==========================================
with tab1:
    col_card, col_advisory = st.columns([1, 2])

    with col_card:
        st.markdown(
            f"""
            <div class="aqi-card" style="background-color: {category_meta['color']};">
                <div style="font-size: 14px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; opacity: 0.9;">
                    National Air Quality Index (NAQI)
                </div>
                <div style="font-size: 64px; font-weight: 800; line-height: 1.1; margin: 8px 0;">
                    {current_aqi}
                </div>
                <div style="font-size: 20px; font-weight: 700;">
                    {category_meta['emoji']} {category_meta['category'].upper()}
                </div>
                <div style="font-size: 13px; margin-top: 8px; opacity: 0.9;">
                    Prominent Hazard: <b>{latest_row['prominent_pollutant']}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_advisory:
        st.info(f"**🩺 Health Advisory ({category_meta['category']}):**\n{category_meta['health_impact']}")
        st.warning(f"**🛡️ Recommended Protective Action:**\n{category_meta['advisory']}")

    st.markdown("### 🔬 Atmospheric Pollutant Breakdown (CPCB Sub-Indices)")
    pcol1, pcol2, pcol3, pcol4, pcol5, pcol6 = st.columns(6)

    sub_dict = {}
    with pcol1:
        pm25_val = latest_row.get("pm25", 0)
        pm25_sub = CPCBCalculator.calculate_sub_index("pm25", pm25_val)
        sub_dict["pm25"] = pm25_sub
        st.metric("PM2.5 (Fine)", f"{pm25_val:.1f} µg/m³", delta=f"Sub-index: {pm25_sub}")
    with pcol2:
        pm10_val = latest_row.get("pm10", 0)
        pm10_sub = CPCBCalculator.calculate_sub_index("pm10", pm10_val)
        sub_dict["pm10"] = pm10_sub
        st.metric("PM10 (Coarse)", f"{pm10_val:.1f} µg/m³", delta=f"Sub-index: {pm10_sub}")
    with pcol3:
        no2_val = latest_row.get("no2", 0)
        no2_sub = CPCBCalculator.calculate_sub_index("no2", no2_val)
        sub_dict["no2"] = no2_sub
        st.metric("NO₂ (Nitrogen)", f"{no2_val:.1f} µg/m³", delta=f"Sub-index: {no2_sub}")
    with pcol4:
        so2_val = latest_row.get("so2", 0)
        so2_sub = CPCBCalculator.calculate_sub_index("so2", so2_val)
        sub_dict["so2"] = so2_sub
        st.metric("SO₂ (Sulphur)", f"{so2_val:.1f} µg/m³", delta=f"Sub-index: {so2_sub}")
    with pcol5:
        co_val = latest_row.get("co", 0)
        co_sub = CPCBCalculator.calculate_sub_index("co", co_val)
        sub_dict["co"] = co_sub
        st.metric("CO (Carbon)", f"{co_val:.2f} mg/m³", delta=f"Sub-index: {co_sub}")
    with pcol6:
        o3_val = latest_row.get("o3", 0)
        o3_sub = CPCBCalculator.calculate_sub_index("o3", o3_val)
        sub_dict["o3"] = o3_sub
        st.metric("O₃ (Ozone)", f"{o3_val:.1f} µg/m³", delta=f"Sub-index: {o3_sub}")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### 📊 Relative Pollutant Contribution")
        st.altair_chart(create_pollutant_breakdown_chart(sub_dict), use_container_width=True)

    with c2:
        st.markdown("#### 🌤️ Companion Meteorology")
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric("Ambient Temperature", f"{latest_row.get('temperature', 25.0):.1f} °C")
        with mcol2:
            st.metric("Relative Humidity", f"{latest_row.get('relative_humidity', 60.0):.1f} %")
        with mcol3:
            wind = latest_row.get('wind_speed', 2.0)
            st.metric("Wind Speed", f"{wind:.2f} m/s", delta="Stagnant" if wind < 1.5 else "Dispersing")

        st.caption(
            "Note: Boundary layer thermal inversions occur in North India when wind speeds drop below 1.5 m/s, trapping particulates."
        )


# ==========================================
# TAB 2: FORECASTING STUDIO
# ==========================================
with tab2:
    st.markdown(f"### 📈 {forecast_horizon}-Hour Forecast Outlook ({selected_model})")

    # Generate forecast using selected model
    last_time = pd.Timestamp(latest_row["timestamp"])

    if "LSTM" in selected_model:
        lstm_path = MODELS_DIR / f"lstm_{selected_city.lower()}_24h.pt"
        fe_path = MODELS_DIR / f"feature_engineer_{selected_city.lower()}.pkl"

        if lstm_path.exists() and fe_path.exists():
            fe = TimeSeriesFeatureEngineer.load(fe_path)
            lstm_trainer = LSTMTrainer.load_checkpoint(lstm_path)

            # Engineer features for available data
            feat_df = fe.engineer_features(data_df)
            X_all, _ = fe.get_feature_matrix(feat_df)
            X_scaled, _ = fe.transform(X_all)

            # Take last lookback window (72h)
            recent_seq = X_scaled[-72:].reshape(1, 72, -1)
            y_hat, y_lower, y_upper = lstm_trainer.predict_with_intervals(recent_seq, fe)

            # Truncate or expand to horizon
            y_hat = y_hat[0][:forecast_horizon]
            y_lower = y_lower[0][:forecast_horizon]
            y_upper = y_upper[0][:forecast_horizon]

            fc_times = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=forecast_horizon, freq="h")
            fc_df = pd.DataFrame({
                "timestamp": fc_times,
                "cpcb_aqi": np.round(y_hat, 1),
                "lower": np.round(y_lower, 1),
                "upper": np.round(y_upper, 1),
            })
        else:
            st.warning("LSTM checkpoint not found. Using Prophet fallback.")
            selected_model = "Prophet / Additive Model"

    if "Prophet" in selected_model or "LSTM" not in selected_model:
        prophet_path = MODELS_DIR / f"prophet_{selected_city.lower()}.pkl"
        if prophet_path.exists():
            prophet = AdditiveProphetForecaster.load(prophet_path)
            fc_df, p_hat, p_lower, p_upper = prophet.predict_future(
                last_timestamp=last_time,
                forecast_hours=forecast_horizon,
                future_weather=data_df[["temperature", "relative_humidity", "wind_speed"]].tail(forecast_horizon),
            )
            fc_df["cpcb_aqi"] = np.round(p_hat, 1)
            fc_df["lower"] = np.round(p_lower, 1)
            fc_df["upper"] = np.round(p_upper, 1)

    # Render Interactive Chart
    chart = create_forecast_chart(data_df, fc_df, selected_city)
    st.altair_chart(chart, use_container_width=True)

    # Peak Prediction Banner
    peak_val = fc_df["cpcb_aqi"].max()
    peak_idx = fc_df["cpcb_aqi"].idxmax()
    peak_row = fc_df.loc[peak_idx]

    peak_cat = CPCBCalculator.get_category_metadata(peak_val)

    b1, b2, b3 = st.columns(3)
    with b1:
        st.metric("Projected Peak NAQI", f"{peak_val:.0f}", delta=f"{peak_cat['category']}")
    with b2:
        st.metric("Peak Anticipated At", peak_row["timestamp"].strftime("%I:%M %p (%a)"))
    with b3:
        st.metric("Impending Hazard Tier", f"{peak_cat['emoji']} {peak_cat['category']}")

    # Benchmark Performance Table
    st.markdown("#### 🔬 Model Cross-Horizon Benchmark Evaluation (Test Split)")
    meta_path = MODELS_DIR / "benchmark_metrics.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            benchmarks = json.load(f)
        city_b = benchmarks.get(selected_city, {})
        rows = []
        for m_name, horz in city_b.items():
            for h_name, vals in horz.items():
                rows.append({
                    "Model Architecture": m_name,
                    "Evaluation Horizon": h_name,
                    "MAE (AQI Units)": vals["mae"],
                    "RMSE (AQI Units)": vals["rmse"],
                    "MAPE (%)": f"{vals['mape']}%",
                    "R² Coefficient": vals["r2"],
                    "Spike Recall (>200)": f"{vals['spike_recall']}%",
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ==========================================
# TAB 3: AUTOMATED ALERT DISPATCHER
# ==========================================
with tab3:
    st.markdown("### 🚨 Multi-Channel Alert Engine (Telegram Bot & SMTP Email)")
    st.caption("Automatic triggering on threshold breach or predictive 24h inversion spike with debounce cooldown.")

    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown("#### ⚙️ Test Dispatch Console")
        target_email = st.text_input("Test Recipient Email", value="student.capstone@university.ac.in")
        send_tg = st.checkbox("Dispatch via Telegram Bot", value=True)
        send_em = st.checkbox("Dispatch via SMTP Email", value=True)
        force_dispatch = st.checkbox("Bypass Cooldown (Force Dispatch)", value=True)

        if st.button("🚀 Trigger Instant Alert Dispatch", type="primary"):
            alert_mgr = AlertManager()
            payload = {
                "city": selected_city,
                "current_aqi": current_aqi,
                "category": category_meta["category"],
                "color": category_meta["color"],
                "emoji": category_meta["emoji"],
                "prominent_pollutant": latest_row["prominent_pollutant"],
                "forecast_peak_aqi": int(peak_val),
                "peak_time": peak_row["timestamp"].strftime("%d %b %I:%M %p"),
                "advisory": category_meta["advisory"],
                "sub_indices": {k: int(v) for k, v in sub_dict.items() if v is not None},
            }

            with st.spinner("Dispatching alerts across configured channels..."):
                res = alert_mgr.dispatch_alert(
                    alert_payload=payload,
                    send_telegram=send_tg,
                    send_email=send_em,
                    recipient_emails=target_email,
                    force=force_dispatch,
                    threshold=custom_threshold,
                )

            if res["triggered"]:
                st.success(f"✅ Alert Successfully Dispatched!\nReason: {res['reason']}")
            else:
                st.warning(f"⚠️ Alert Suppressed:\n{res['reason']}")

    with c_right:
        st.markdown("#### 📱 Formatted Telegram Alert Preview")
        tg_preview = TelegramAlertService().format_alert_message({
            "city": selected_city,
            "current_aqi": current_aqi,
            "category": category_meta["category"],
            "emoji": category_meta["emoji"],
            "prominent_pollutant": latest_row["prominent_pollutant"],
            "forecast_peak_aqi": int(peak_val),
            "peak_time": peak_row["timestamp"].strftime("%d %b %I:%M %p"),
            "advisory": category_meta["advisory"],
            "sub_indices": {k: int(v) for k, v in sub_dict.items() if v is not None},
        })
        st.markdown(
            f"""
            <div style="background: #1E293B; color: #F1F5F9; padding: 16px; border-radius: 8px; font-family: monospace; font-size: 13px; white-space: pre-wrap;">
{tg_preview}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 📋 Alert Dispatch Audit Log")
    history_file = BASE_DIR / "data" / "processed" / "alert_history.json"
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as f:
            hist_records = json.load(f)
        if hist_records:
            hist_table = [
                {
                    "Timestamp": r.get("timestamp")[:19].replace("T", " "),
                    "City": r.get("city"),
                    "NAQI": r.get("current_aqi"),
                    "Category": r.get("category"),
                    "Peak Forecast": r.get("forecast_peak_aqi"),
                    "Trigger Reason": r.get("reason"),
                }
                for r in reversed(hist_records)
            ]
            st.dataframe(pd.DataFrame(hist_table), use_container_width=True, hide_index=True)


# ==========================================
# TAB 4: GEOSPATIAL & STATION GRID
# ==========================================
with tab4:
    st.markdown(f"### 🗺️ Continuous Ambient Monitoring Stations in {selected_city}")
    stations = city_meta.get("stations", [])

    st_df = pd.DataFrame(stations)
    # Estimate station variations based on hotspot type
    st_df["Current AQI"] = [
        int(current_aqi * (1.18 if "Industrial" in s["type"] else (1.05 if "Transport" in s["type"] else 0.92)))
        for s in stations
    ]
    st_df["Status"] = [CPCBCalculator.get_category_metadata(a)["category"] for a in st_df["Current AQI"]]

    scol1, scol2 = st.columns([1, 1])
    with scol1:
        st.dataframe(
            st_df[["id", "name", "type", "Current AQI", "Status"]],
            use_container_width=True,
            hide_index=True,
        )
    with scol2:
        map_df = pd.DataFrame({
            "lat": [s["lat"] for s in stations],
            "lon": [s["lon"] for s in stations],
        })
        st.map(map_df, zoom=10)


# ==========================================
# TAB 5: CPCB SCIENTIFIC FRAMEWORK
# ==========================================
with tab5:
    st.markdown("### 📖 Central Pollution Control Board (CPCB) Mathematical Standard")
    st.latex(r"I_p = I_{lo} + \left( \frac{I_{hi} - I_{lo}}{B_{hi} - B_{lo}} \right) \cdot (C_p - B_{lo})")
    st.latex(r"AQI = \max(I_1, I_2, \dots, I_n) \quad \text{where } n \ge 3 \text{ and includes } PM_{2.5} \text{ or } PM_{10}")

    st.markdown("#### 🏷️ Official NAQI Breakpoint Thresholds")
    b_rows = []
    for cat in NAQI_CATEGORIES:
        b_rows.append({
            "Category": f"{cat['emoji']} {cat['category']}",
            "Index Range": f"{cat['aqi_min']} – {cat['aqi_max']}",
            "Color Code": cat["hex_badge"],
            "Health Impact Summary": cat["health_impact"],
            "Precautionary Directive": cat["advisory"],
        })
    st.dataframe(pd.DataFrame(b_rows), use_container_width=True, hide_index=True)

    st.markdown("#### 🚨 Delhi-NCR GRAP (Graded Response Action Plan) Protocol")
    st.markdown(
        """
        - **Stage I (Poor, AQI 201–300):** Dust suppression on construction sites, anti-smog guns, ban on open burning.
        - **Stage II (Very Poor, AQI 301–400):** Mechanized sweeping, diesel generator bans, increase parking fees to discourage vehicular traffic.
        - **Stage III (Severe, AQI 401–450):** Ban on non-essential construction, restriction on BS-III petrol & BS-IV diesel cars.
        - **Stage IV (Severe+, AQI > 450):** Stop entry of non-essential trucks into Delhi, closure of schools, 50% work-from-home mandate.
        """
    )
