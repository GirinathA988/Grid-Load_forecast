import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Grid Load Forecast", layout="wide", page_icon="⚡")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
PREDICTIONS_PATH = "data/predictions/predictions.csv"

@st.cache_data(ttl=60)
def load_data():
    if not Path(PREDICTIONS_PATH).exists():
        return None
    df = pd.read_csv(PREDICTIONS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["error"] = df["actual_load"] - df["predicted_load"]
    df["abs_pct_error"] = (df["error"].abs() / df["actual_load"]) * 100
    return df

df = load_data()

st.title("⚡ Grid Load Forecast")
st.caption("Daily peak electricity demand — actual vs. model prediction")

if df is None:
    st.warning("No forecast data available yet. Please check back once the model has run.")
    st.stop()

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
col_a, col_b = st.columns([1, 2])
with col_a:
    states = sorted(df["state"].unique())
    selected_state = st.selectbox("State", states, index=0)
with col_b:
    date_min, date_max = df["timestamp"].min(), df["timestamp"].max()
    date_range = st.slider(
        "Date range",
        min_value=date_min.to_pydatetime(),
        max_value=date_max.to_pydatetime(),
        value=(date_min.to_pydatetime(), date_max.to_pydatetime()),
        format="DD MMM YYYY",
    )

filtered = df[df["state"] == selected_state]
filtered = filtered[
    (filtered["timestamp"] >= date_range[0]) & (filtered["timestamp"] <= date_range[1])
]

if filtered.empty:
    st.info("No data in the selected range.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
rmse = (filtered["error"] ** 2).mean() ** 0.5
mape = filtered["abs_pct_error"].mean()
latest = filtered.iloc[-1]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Latest Actual Load", f"{latest['actual_load']:,.0f} MW")
k2.metric("Latest Predicted Load", f"{latest['predicted_load']:,.0f} MW",
           delta=f"{latest['predicted_load'] - latest['actual_load']:+.0f} MW")
k3.metric("Forecast Accuracy (MAPE)", f"{mape:.2f}%")
k4.metric("RMSE", f"{rmse:,.0f} MW")

# ---------------------------------------------------------------------------
# Main chart
# ---------------------------------------------------------------------------
st.subheader(f"Load Trend — {selected_state}")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=filtered["timestamp"], y=filtered["actual_load"],
    mode="lines", name="Actual", line=dict(color="#2563eb", width=2.5)
))
fig.add_trace(go.Scatter(
    x=filtered["timestamp"], y=filtered["predicted_load"],
    mode="lines", name="Predicted", line=dict(color="#f97316", width=2, dash="dash")
))
fig.update_layout(
    xaxis_title="Date", yaxis_title="Peak Demand (MW)",
    hovermode="x unified", height=460,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=30),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Stress indicator
# ---------------------------------------------------------------------------
threshold = filtered["actual_load"].quantile(0.9)
high_days = filtered[filtered["predicted_load"] > threshold]
if not high_days.empty:
    st.subheader("⚠️ High-Demand Days (top 10th percentile forecast)")
    st.dataframe(
        high_days[["timestamp", "predicted_load"]]
        .rename(columns={"timestamp": "Date", "predicted_load": "Predicted Load (MW)"})
        .round(0),
        use_container_width=True, hide_index=True,
    )

with st.expander("View full data"):
    st.dataframe(
        filtered[["timestamp", "state", "actual_load", "predicted_load"]]
        .rename(columns={"timestamp": "Date", "state": "State",
                          "actual_load": "Actual (MW)", "predicted_load": "Predicted (MW)"})
        .round(1),
        use_container_width=True, hide_index=True,
    )

st.caption(f"Data last refreshed from pipeline output · {len(df)} total records across {df['state'].nunique()} state(s)")
