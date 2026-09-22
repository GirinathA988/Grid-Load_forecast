import streamlit as st
import pandas as pd
import subprocess
import time
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Pipeline Operator Console", layout="wide", page_icon="🛠️")

PREDICTIONS_PATH = Path("data/predictions/predictions.csv")
DATA_DIR = Path("data")
SCRIPT_PATH = "/scripts/train_model.py"

st.title("🛠️ Grid Forecast — Operator Console")
st.caption("Pipeline health, data management, and model retraining")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_cmd(cmd, timeout=60):
    """Run a shell command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, (result.stdout + result.stderr)
    except subprocess.TimeoutExpired:
        return False, "Command timed out."
    except Exception as e:
        return False, str(e)

def check_container(name):
    ok, out = run_cmd(f'docker inspect -f "{{{{.State.Status}}}}" {name}')
    if ok and "running" in out.lower():
        return "🟢 Running"
    elif ok:
        return f"🟡 {out.strip()}"
    return "🔴 Not found / stopped"

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_health, tab_view, tab_data, tab_train, tab_logs = st.tabs(
    ["Cluster Health", "View Data", "Data Management", "Model Training", "Logs"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Cluster health
# ---------------------------------------------------------------------------
with tab_health:
    st.subheader("Container Status")
    if st.button("🔄 Refresh status"):
        st.rerun()

    containers = ["namenode", "datanode", "hive-metastore-postgresql",
                  "hive-metastore", "hive-server", "spark-master", "spark-worker"]

    cols = st.columns(4)
    for i, c in enumerate(containers):
        with cols[i % 4]:
            st.metric(c, check_container(c))

    st.divider()
    st.subheader("Quick Links")
    l1, l2 = st.columns(2)
    l1.link_button("HDFS NameNode UI", "http://localhost:9870", use_container_width=True)
    l2.link_button("Spark Master UI", "http://localhost:8080", use_container_width=True)

    st.divider()
    st.subheader("Container Resource Usage")
    if st.button("Fetch docker stats (one-time snapshot)"):
        ok, out = run_cmd(
            'docker stats --no-stream --format "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"',
            timeout=15,
        )
        if ok and out.strip():
            rows = [line.split("\t") for line in out.strip().split("\n") if line.strip()]
            stats_df = pd.DataFrame(rows, columns=["Container", "CPU %", "Memory"])
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        else:
            st.error("Could not fetch stats. Is Docker running?")

# ---------------------------------------------------------------------------
# TAB 2 — View data (raw dataset + predictions, filterable)
# ---------------------------------------------------------------------------
with tab_view:
    st.subheader("Predictions Output")
    if PREDICTIONS_PATH.exists():
        preds_df = pd.read_csv(PREDICTIONS_PATH)
        preds_df["timestamp"] = pd.to_datetime(preds_df["timestamp"])

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            state_filter = st.multiselect(
                "Filter by state",
                sorted(preds_df["state"].unique()),
                default=None,
                placeholder="All states",
            )
        with c2:
            date_from = st.date_input("From", value=preds_df["timestamp"].min().date())
        with c3:
            date_to = st.date_input("To", value=preds_df["timestamp"].max().date())

        view = preds_df.copy()
        if state_filter:
            view = view[view["state"].isin(state_filter)]
        view = view[
            (view["timestamp"].dt.date >= date_from) & (view["timestamp"].dt.date <= date_to)
        ]

        st.caption(f"{len(view):,} of {len(preds_df):,} rows shown")
        st.dataframe(view, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download filtered rows as CSV",
            view.to_csv(index=False).encode("utf-8"),
            file_name="predictions_filtered.csv",
            mime="text/csv",
        )
    else:
        st.warning("No predictions file found yet. Run training from the Model Training tab.")

    st.divider()
    st.subheader("Raw Dataset")
    raw_files = list(DATA_DIR.glob("*.csv"))
    if raw_files:
        selected_raw = st.selectbox("Select raw file", [f.name for f in raw_files])
        raw_path = DATA_DIR / selected_raw

        # Only read a preview for large files to keep the UI responsive
        preview_rows = st.slider("Rows to preview", 10, 1000, 100, step=10)
        raw_df = pd.read_csv(raw_path, nrows=preview_rows)
        st.caption(f"Showing first {len(raw_df):,} rows of `{selected_raw}`")
        st.dataframe(raw_df, use_container_width=True, hide_index=True)
    else:
        st.info("No raw CSV files found in `data/`.")

# ---------------------------------------------------------------------------
# TAB 3 — Data management
# ---------------------------------------------------------------------------
with tab_data:
    st.subheader("Dataset Status")

    raw_csvs = list(DATA_DIR.glob("*.csv"))
    if raw_csvs:
        for f in raw_csvs:
            size_mb = f.stat().st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(f.stat().st_mtime).strftime("%d %b %Y, %H:%M")
            st.write(f"📄 **{f.name}** — {size_mb:.2f} MB — last modified {modified}")
    else:
        st.warning("No raw CSV found in `data/`.")

    st.divider()
    st.subheader("Upload New Dataset")
    uploaded = st.file_uploader("Upload a new grid demand CSV", type=["csv"])
    if uploaded is not None:
        dest = DATA_DIR / uploaded.name
        with open(dest, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved to `{dest}`. Load it into HDFS from the command line, then re-run training:")
        st.code(
            f'docker exec -it namenode bash -c '
            f'"hdfs dfs -put -f /data/{uploaded.name} /data/grid/raw/"',
            language="powershell",
        )

    st.divider()
    st.subheader("HDFS Contents")
    if st.button("List HDFS files"):
        ok, out = run_cmd('docker exec namenode hdfs dfs -ls /data/grid/raw', timeout=20)
        st.code(out if out.strip() else "No output — check the cluster is running.", language="text")

# ---------------------------------------------------------------------------
# TAB 4 — Model training
# ---------------------------------------------------------------------------
with tab_train:
    st.subheader("Retrain Model")
    st.write(
        "Runs the PySpark training job against the current Hive table and "
        "overwrites `data/predictions/predictions.csv`."
    )

    confirm = st.checkbox("I understand this will overwrite the current predictions file.")
    if st.button("▶️ Run training job", disabled=not confirm, type="primary"):
        with st.status("Running Spark job...", expanded=True) as status:
            st.write("Submitting to Spark cluster — this can take 1–3 minutes...")
            cmd = (
                'docker exec -u root -e HOME=/tmp spark-master '
                'spark-submit --master spark://spark-master:7077 '
                '--conf spark.jars.ivy=/tmp/.ivy2 ' + SCRIPT_PATH
            )
            start = time.time()
            ok, out = run_cmd(cmd, timeout=600)
            elapsed = time.time() - start

            st.session_state["last_train_log"] = out
            st.session_state["last_train_time"] = datetime.now().strftime("%d %b %Y, %H:%M:%S")

            if ok:
                status.update(label=f"Training completed in {elapsed:.0f}s", state="complete")
                st.success("Model retrained. Predictions file updated.")
            else:
                status.update(label="Training failed", state="error")
                st.error("Training job failed — see the Logs tab for details.")

    if PREDICTIONS_PATH.exists():
        mtime = datetime.fromtimestamp(PREDICTIONS_PATH.stat().st_mtime)
        st.caption(f"Predictions last generated: {mtime.strftime('%d %b %Y, %H:%M:%S')}")

# ---------------------------------------------------------------------------
# TAB 5 — Logs
# ---------------------------------------------------------------------------
with tab_logs:
    st.subheader("Last Training Run Output")
    if "last_train_log" in st.session_state:
        st.caption(f"From run at {st.session_state.get('last_train_time', 'unknown time')}")
        st.text_area("Spark job output", st.session_state["last_train_log"], height=400)
    else:
        st.info("No training run yet this session. Run one from the Model Training tab.")

    st.divider()
    st.subheader("Container Logs")
    log_container = st.selectbox(
        "Select container",
        ["namenode", "datanode", "hive-server", "hive-metastore", "spark-master", "spark-worker"],
    )
    if st.button("Fetch last 50 log lines"):
        ok, out = run_cmd(f"docker logs --tail 50 {log_container}", timeout=15)
        st.code(out if out.strip() else "No logs returned.", language="text")
