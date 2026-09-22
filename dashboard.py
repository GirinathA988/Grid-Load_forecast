import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Grid Load Forecast", layout="wide")

st.title("⚡ Grid Load Forecast Dashboard")
st.caption("HDFS → Hive → PySpark → Spark MLlib → Streamlit")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/predictions/predictions.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["error"] = df["actual_load"] - df["predicted_load"]
    df["abs_pct_error"] = (df["error"].abs() / df["actual_load"]) * 100
    return df

df = load_data()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
states = sorted(df["state"].unique())
selected_state = st.sidebar.selectbox("State", states, index=0)

date_min, date_max = df["timestamp"].min(), df["timestamp"].max()
date_range = st.sidebar.date_input(
    "Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max
)

filtered = df[df["state"] == selected_state]
if len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered = filtered[(filtered["timestamp"] >= start) & (filtered["timestamp"] <= end)]

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
rmse = (filtered["error"] ** 2).mean() ** 0.5
mape = filtered["abs_pct_error"].mean()

col1.metric("Rows", len(filtered))
col2.metric("RMSE (MW)", f"{rmse:,.1f}")
col3.metric("MAPE", f"{mape:.2f}%")
col4.metric("Avg Actual Load (MW)", f"{filtered['actual_load'].mean():,.0f}")

# ---------------------------------------------------------------------------
# Main chart: actual vs predicted
# ---------------------------------------------------------------------------
st.subheader(f"Actual vs Predicted Load — {selected_state}")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=filtered["timestamp"], y=filtered["actual_load"],
    mode="lines", name="Actual", line=dict(color="#2563eb", width=2)
))
fig.add_trace(go.Scatter(
    x=filtered["timestamp"], y=filtered["predicted_load"],
    mode="lines", name="Predicted", line=dict(color="#f97316", width=2, dash="dash")
))
fig.update_layout(
    xaxis_title="Date", yaxis_title="Peak Demand (MW)",
    hovermode="x unified", height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Error chart
# ---------------------------------------------------------------------------
st.subheader("Prediction Error Over Time")
fig_err = go.Figure()
fig_err.add_trace(go.Bar(
    x=filtered["timestamp"], y=filtered["error"],
    marker_color=["#dc2626" if e < 0 else "#16a34a" for e in filtered["error"]],
    name="Error (Actual − Predicted)",
))
fig_err.update_layout(
    xaxis_title="Date", yaxis_title="Error (MW)", height=300,
)
st.plotly_chart(fig_err, use_container_width=True)

# ---------------------------------------------------------------------------
# Data table
# ---------------------------------------------------------------------------
with st.expander("View raw data"):
    st.dataframe(
        filtered[["timestamp", "state", "actual_load", "predicted_load", "error", "abs_pct_error"]]
        .rename(columns={"abs_pct_error": "abs % error"})
        .round(2),
        use_container_width=True,
    )