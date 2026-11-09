"""
Altair & Matplotlib Visualizations for Air Quality Forecasting
Author: Khushi Yadav (@khushiyadav35)
"""

import altair as alt
import numpy as np
import pandas as pd


def create_forecast_chart(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    city_name: str,
    y_col: str = "cpcb_aqi",
) -> alt.Chart:
    """
    Creates an interactive forecast chart with historical trend,
    forecast projection line, 95% confidence band, and CPCB threshold reference lines.
    """
    # Prepare history slice (last 48h)
    hist = history_df.tail(48)[["timestamp", y_col]].copy()
    hist["type"] = "Historical Observation"
    hist["lower"] = hist[y_col]
    hist["upper"] = hist[y_col]

    # Forecast slice
    fc = forecast_df.copy()
    fc["type"] = "Model Forecast"

    combined = pd.concat([hist, fc], ignore_index=True)

    # 1. Uncertainty Ribbon for Forecast
    band = (
        alt.Chart(fc)
        .mark_area(opacity=0.25, color="#2563EB")
        .encode(
            x=alt.X("timestamp:T", title="Timeline (Hourly)"),
            y=alt.Y("lower:Q", title="National Air Quality Index (NAQI)"),
            y2="upper:Q",
        )
    )

    # 2. Main Trajectory Line
    lines = (
        alt.Chart(combined)
        .mark_line(strokeWidth=3)
        .encode(
            x="timestamp:T",
            y=alt.Y(f"{y_col}:Q"),
            color=alt.Color(
                "type:N",
                scale=alt.Scale(
                    domain=["Historical Observation", "Model Forecast"],
                    range=["#0F172A", "#2563EB"],
                ),
                legend=alt.Legend(title="Series"),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Timestamp"),
                alt.Tooltip(f"{y_col}:Q", title="AQI", format=".1f"),
                alt.Tooltip("type:N", title="Mode"),
            ],
        )
    )

    # 3. Threshold Lines
    threshold_data = pd.DataFrame([
        {"aqi": 200, "label": "Poor Threshold (200)"},
        {"aqi": 300, "label": "Very Poor Threshold (300)"},
        {"aqi": 400, "label": "Severe Threshold (400)"},
    ])
    rule = (
        alt.Chart(threshold_data)
        .mark_rule(strokeDash=[4, 4], opacity=0.6, color="#EF4444")
        .encode(y="aqi:Q")
    )

    chart = (
        (band + lines + rule)
        .properties(
            title=f"Air Quality Trajectory & 24h-72h Outlook - {city_name}",
            width="container",
            height=380,
        )
        .interactive()
    )

    return chart


def create_pollutant_breakdown_chart(sub_indices: dict) -> alt.Chart:
    """Creates a horizontal bar chart displaying individual CPCB pollutant sub-indices."""
    data = pd.DataFrame([
        {"Pollutant": k.upper(), "SubIndex": v} for k, v in sub_indices.items()
    ]).sort_values("SubIndex", ascending=False)

    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadius=4)
        .encode(
            x=alt.X("SubIndex:Q", title="CPCB Sub-Index Value"),
            y=alt.Y("Pollutant:N", sort="-x", title="Atmospheric Pollutant"),
            color=alt.Color(
                "SubIndex:Q",
                scale=alt.Scale(
                    domain=[50, 100, 200, 300, 400],
                    range=["#10B981", "#84CC16", "#EAB308", "#F97316", "#EF4444"],
                ),
                legend=None,
            ),
            tooltip=["Pollutant:N", "SubIndex:Q"],
        )
        .properties(height=220, width="container")
    )
    return chart
