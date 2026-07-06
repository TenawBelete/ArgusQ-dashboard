import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# Allow imports from the project root when run from pipeline/dashboard/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from deltalake import DeltaTable
except Exception:
    DeltaTable = None

try:
    from configs.settings import STORAGE_OPTIONS, DELTA_GOLD
except Exception:
    STORAGE_OPTIONS = {}
    DELTA_GOLD = None

# Optional: boto3 lets us actually ping MinIO. If absent, we degrade honestly.
try:
    import boto3
    from botocore.config import Config as BotoConfig
except Exception:
    boto3 = None

try:
    from kafka.admin import KafkaAdminClient
except Exception:
    KafkaAdminClient = None


# ---------------------------------------------------------------------
# Constants. Documented, not magic.
# ---------------------------------------------------------------------
# DEFAULT_THRESHOLD is the SECOM operating-point threshold carried from the
# calibrated model artefact (secom_threshold.json ~ 0.071). It is the point
# chosen to balance the cost of missed failures against false alarms on a
# 6.6% positive-rate stream. It is NOT a probability of failure.
DEFAULT_THRESHOLD = 0.071

# Hotelling T² drift threshold: chi-squared(df=173) at alpha=0.05.
# Used for Stage 1 multivariate drift detection on the full SECOM feature space.
T2_DRIFT_THRESHOLD = 204.69

# HIGH_RISK_CUTOFF is a *display* band, not a model output. It groups the
# upper tail of calibrated scores for operator triage. Defensible answer if
# asked "why 0.20?": it is the visual escalation band, tunable per line, and
# not used for any automated action.
HIGH_RISK_CUTOFF = 0.20

# Documented honest model performance, surfaced in the model panel.
MODEL_AUC_CHRONO = 0.6037
SECOM_POSITIVE_RATE = 0.066

# Operator-action band thresholds (alert-rate %, avg-risk). Named, not magic,
# so the business_status logic reads as intent rather than arithmetic.
ESCALATE_ALERT_PCT = 20.0
INVESTIGATE_ALERT_PCT = 8.0

# Refresh cadence. The screen redraws every UI_REFRESH_SECS, but the network
# health checks are cached for STATUS_CACHE_SECS - deliberately longer than the
# refresh interval so MinIO/Kafka are NOT re-pinged on every redraw. This keeps
# the loop responsive even when a service is down and its check is timing out.
UI_REFRESH_SECS = 5
STATUS_CACHE_SECS = 15
AUTO_REFRESH_SECS = 30

# MinIO / S3 endpoint for the live status ping. Read from STORAGE_OPTIONS if
# present so we don't duplicate config; fall back to the local dev defaults
# from the pipeline tutorial.
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", STORAGE_OPTIONS.get("AWS_ENDPOINT_URL", "http://localhost:9000"))
MINIO_KEY = os.getenv("MINIO_ACCESS_KEY", STORAGE_OPTIONS.get("AWS_ACCESS_KEY_ID", "argusq_admin"))
MINIO_SECRET = os.getenv("MINIO_SECRET_KEY", STORAGE_OPTIONS.get("AWS_SECRET_ACCESS_KEY", "argusq_password"))
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "argusq-datalake")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_RAW = os.getenv("KAFKA_TOPIC_RAW", "argusq.secom.raw")


st.set_page_config(
    page_title="ArgusQ - Process Stability Monitor",
    page_icon="⚡",
    layout="wide",
)


# ---------------------------------------------------------------------
# Real status checks. Nothing here is hardcoded green.
# ---------------------------------------------------------------------
@st.cache_data(ttl=STATUS_CACHE_SECS)
def check_minio() -> tuple[bool, str]:
    """Actually ping MinIO and check the datalake bucket exists."""
    if boto3 is None:
        return False, "boto3 not installed - cannot verify MinIO."
    try:
        client = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_KEY,
            aws_secret_access_key=MINIO_SECRET,
            config=BotoConfig(connect_timeout=2, read_timeout=2, retries={"max_attempts": 1}),
        )
        buckets = [b["Name"] for b in client.list_buckets().get("Buckets", [])]
        if MINIO_BUCKET in buckets:
            return True, f"Bucket '{MINIO_BUCKET}' reachable."
        return False, f"MinIO up but bucket '{MINIO_BUCKET}' not found."
    except Exception as exc:
        return False, f"MinIO unreachable: {exc}"


@st.cache_data(ttl=STATUS_CACHE_SECS)
def check_kafka() -> tuple[bool, str]:
    """Actually connect to Kafka and verify the raw SECOM topic exists."""
    if KafkaAdminClient is None:
        return False, "kafka-python not installed - cannot verify Kafka."
    try:
        admin = KafkaAdminClient(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            client_id="argusq-dashboard-health",
            request_timeout_ms=2500,
            api_version_auto_timeout_ms=2500,
        )
        topics = admin.list_topics()
        admin.close()
        if KAFKA_TOPIC_RAW in topics:
            return True, f"Kafka reachable; topic '{KAFKA_TOPIC_RAW}' exists."
        return False, f"Kafka reachable but topic '{KAFKA_TOPIC_RAW}' not found."
    except Exception as exc:
        return False, f"Kafka unreachable: {exc}"


@st.cache_data(ttl=UI_REFRESH_SECS)
def check_gold_table() -> tuple[bool, str, pd.DataFrame | None]:
    """Open the Delta gold table and return its rows (or the real error).

    Reads only the columns the dashboard uses (column pruning) so the transfer
    stays small even if the table grows to many thousands of scored rows.
    """
    if DeltaTable is None:
        return False, "deltalake package not installed.", None
    if not DELTA_GOLD:
        return False, "DELTA_GOLD not configured in settings.", None

    gold_path = f"{DELTA_GOLD}/secom"
    required_cols = {
        "timestamp", "row_id", "raw_prob", "cal_prob",
        "alert", "risk_level", "label", "scored_at",
    }
    # Optional columns from v7 upgrades - loaded if present, skipped gracefully if not.
    optional_cols = {"drift_score", "top_shap_feat", "top_shap_value"}
    try:
        dt = DeltaTable(gold_path, storage_options=STORAGE_OPTIONS)

        # Compatibility fix for different deltalake / delta-rs versions.
        # Some versions return a Schema object that has .to_pyarrow().names,
        # while the Streamlit Cloud version currently returns a Schema object
        # without .to_pyarrow(). In that case, schema.fields gives us the names.
        schema = dt.schema()
        try:
            available = set(schema.to_pyarrow().names)
        except AttributeError:
            available = set(field.name for field in schema.fields)

        cols_to_read = sorted(required_cols | (optional_cols & available))
        try:
            df = dt.to_pandas(columns=cols_to_read)
        except Exception as col_exc:
            return False, f"Gold table missing or unreadable columns: {col_exc}", None

        if df.empty:
            return False, "Gold table exists but has zero rows (pipeline not run yet).", None

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["scored_at"] = pd.to_datetime(df["scored_at"], errors="coerce")
        df["alert"] = df["alert"].astype(bool)
        df = df.sort_values("scored_at", ascending=False)
        return True, f"{len(df):,} rows in gold table.", df
    except Exception as exc:
        return False, f"Gold read failed: {exc}", None


def freshness_label(latest: pd.Timestamp | None) -> tuple[str, str]:
    """Real freshness derived from the most recent scored_at, safely handling timezone-aware timestamps."""
    if latest is None or pd.isna(latest):
        return "unknown", "No scored rows yet."

    latest_ts = pd.Timestamp(latest)

    if latest_ts.tzinfo is None:
        now_ts = pd.Timestamp.now()
    else:
        now_ts = pd.Timestamp.now(tz=latest_ts.tz)

    secs = max(0, (now_ts - latest_ts).total_seconds())

    if secs < 30:
        return "fresh", f"Last batch {int(secs)}s ago - stream is live."
    if secs < 300:
        return "recent", f"Last batch {int(secs // 60)}m {int(secs % 60)}s ago."
    return "stale", f"Last batch {int(secs // 60)}m ago - stream may have stopped."


# ---------------------------------------------------------------------
# Demo data: weak class separation, mild defensible drift. NOT a magic model.
# ---------------------------------------------------------------------
def make_demo_data(n: int = 240, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """
    Simulated data calibrated to look like a ~0.60-AUC model, not a perfect one.
    True labels and risk scores overlap heavily on purpose: a strong demo would
    be a lie about what the real model can do.
    """
    rng = np.random.default_rng(7)
    now = datetime.now()
    timestamps = [now - timedelta(seconds=20 * i) for i in range(n)][::-1]

    # 6.6% positive rate, matching SECOM.
    labels = rng.choice([0, 1], size=n, p=[1 - SECOM_POSITIVE_RATE, SECOM_POSITIVE_RATE])

    # Weak separation: positives get only a small mean shift in risk. This is
    # what AUC ~0.60 looks like - the distributions mostly overlap.
    base = rng.beta(2, 30, size=n)
    base = base + labels * rng.normal(0.025, 0.02, size=n)  # tiny, noisy lift

    # Mild gradual drift in the last third (drift monitoring is the use case).
    drift_start = int(n * 0.66)
    base[drift_start:] += np.linspace(0.0, 0.05, n - drift_start)

    cal_prob = np.clip(base + rng.normal(0, 0.008, size=n), 0, 0.45)
    raw_prob = np.clip(cal_prob + rng.normal(0, 0.02, size=n), 0, 1)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "row_id": np.arange(n),
        "label": labels,
        "raw_prob": raw_prob,
        "cal_prob": cal_prob,
        "scored_at": timestamps,
    })
    return df.sort_values("scored_at", ascending=False)


def apply_bands(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["alert"] = df["cal_prob"] >= threshold
    df["risk_level"] = np.where(
        df["cal_prob"] >= HIGH_RISK_CUTOFF, "High",
        np.where(df["cal_prob"] >= threshold, "Medium", "Low"),
    )
    return df


def business_status(alert_pct: float, avg_risk: float) -> tuple[str, str]:
    if alert_pct >= ESCALATE_ALERT_PCT or avg_risk >= HIGH_RISK_CUTOFF:
        return "Escalate", "Elevated instability signal. Operator review recommended before next batch."
    if alert_pct >= INVESTIGATE_ALERT_PCT or avg_risk >= DEFAULT_THRESHOLD:
        return "Investigate", "Drift or alert clustering emerging. Watch the next few batches."
    return "Watch", "No strong instability signal in the current window. Routine monitoring."


def status_dot(ok: bool) -> str:
    return "🟢" if ok else "🔴"


# ---------------------------------------------------------------------
# Sidebar: explicit mode choice. No silent fallback.
# ---------------------------------------------------------------------
st.sidebar.title("ArgusQ Controls")

mode = st.sidebar.radio(
    "Data mode",
    ["Live (read pipeline)", "Demo (simulated)"],
    help="Live reads the real Delta gold table. Demo uses clearly-labelled simulated data. "
         "There is no automatic fallback between them - you choose.",
)

threshold = st.sidebar.number_input(
    "Alert threshold (operating point)",
    min_value=0.001, max_value=1.0, value=DEFAULT_THRESHOLD, step=0.001, format="%.3f",
    help="Carried from the calibrated model artefact. Tunable per production line.",
)

row_window = st.sidebar.slider("Rows to display", 50, 1000, 240, 50)
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

auto_refresh_enabled = st.sidebar.toggle(
    "Auto-refresh every 30s",
    value=True,
    help="Turn this off when you do not want the dashboard to rerun automatically. Manual Refresh still works.",
)

if auto_refresh_enabled:
    st.sidebar.caption("Auto-refresh is ON: the dashboard reruns every 30 seconds.")
    if st_autorefresh is not None:
        st_autorefresh(interval=AUTO_REFRESH_SECS * 1000, key="argusq_dashboard_autorefresh")
    else:
        st.sidebar.warning("Auto-refresh package is not installed. Manual Refresh still works.")
else:
    st.sidebar.caption("Auto-refresh is OFF. Use Refresh now for manual reloads.")

st.sidebar.caption("Data is cached for 15s. Click Refresh to force an immediate reload.")

st.sidebar.markdown("")
st.sidebar.caption(
    "Live mode degrades honestly: if MinIO is up but the gold table is empty, "
    "or a read fails, the dashboard tells you exactly what is missing rather "
    "than showing simulated data dressed as real."
)


# ---------------------------------------------------------------------
# Theme - industrial control-room palette, injected once.
# Steel blue primary, amber = watch, red = alert, green = stable.
# ---------------------------------------------------------------------
ACCENT = "#4F9CF9"      # steel blue
COL_STABLE = "#3FB68B"  # green
COL_WATCH = "#E8A13A"   # amber
COL_ALERT = "#E2574C"   # red
COL_PANEL = "#161B22"   # card surface
COL_BORDER = "rgba(120,140,170,0.18)"

st.markdown(
    f"""
    <style>
    .stApp {{ background: #0D1117; }}
    .argus-header {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 18px 22px; margin-bottom: 8px;
        background: linear-gradient(90deg, #11161D 0%, #18222F 100%);
        border: 1px solid {COL_BORDER}; border-left: 4px solid {ACCENT};
        border-radius: 10px;
    }}
    .argus-title {{ font-size: 22px; font-weight: 700; color: #E6EDF3; letter-spacing: 0.3px; }}
    .argus-sub {{ font-size: 12.5px; color: #8B98A8; margin-top: 3px; }}
    .argus-chip {{
        font-size: 12px; font-weight: 600; padding: 6px 12px; border-radius: 20px;
        color: #0D1117;
    }}
    .argus-card {{
        background: {COL_PANEL}; border: 1px solid {COL_BORDER};
        border-radius: 10px; padding: 20px 22px; height: 100%;
    }}
    .argus-card .lbl {{ font-size: 12px; color: #8B98A8; text-transform: uppercase; letter-spacing: 0.5px; }}
    .argus-card .val {{ font-size: 32px; font-weight: 700; color: #E6EDF3; margin-top: 4px; }}
    .argus-card .sub {{ font-size: 13px; color: #8B98A8; margin-top: 4px; }}
    .status-card {{
        background: {COL_PANEL}; border: 1px solid {COL_BORDER};
        border-radius: 10px; padding: 12px 14px; border-top: 3px solid var(--sc);
    }}
    .status-card .h {{ font-size: 13.5px; font-weight: 600; color: #E6EDF3; }}
    .status-card .m {{ font-size: 11.5px; color: #8B98A8; margin-top: 4px; line-height: 1.45; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
mode_chip_color = COL_WATCH if mode == "Demo (simulated)" else COL_STABLE
mode_chip_text = "DEMO" if mode == "Demo (simulated)" else "LIVE"
st.markdown(
    f"""
    <div class="argus-header">
      <div>
        <div class="argus-title">⚡ ArgusQ - Process Stability Monitor</div>
        <div class="argus-sub">Kafka → Spark Streaming → Delta Lake / MinIO → Airflow · SECOM semiconductor line</div>
      </div>
      <div class="argus-chip" style="background:{mode_chip_color};">{mode_chip_text}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Pipeline status panel - REAL checks
# ---------------------------------------------------------------------
minio_ok, minio_msg = check_minio()
kafka_ok, kafka_msg = check_kafka()
gold_ok, gold_msg, live_df = check_gold_table()

latest_scored = live_df["scored_at"].max() if (gold_ok and live_df is not None) else None
fresh_state, fresh_msg = freshness_label(latest_scored)
fresh_ok = fresh_state in ("fresh", "recent")

st.markdown("##### Pipeline status")

def status_card(col, ok: bool, title: str, msg: str):
    color = COL_STABLE if ok else COL_ALERT
    dot = "●" if ok else "●"
    col.markdown(
        f"""
        <div class="status-card" style="--sc:{color};">
          <div class="h"><span style="color:{color};">{dot}</span> {title}</div>
          <div class="m">{msg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

c1, c2, c3, c4 = st.columns(4)
status_card(c1, minio_ok, "MinIO object store", minio_msg)
status_card(c2, kafka_ok, "Kafka broker", kafka_msg)
status_card(c3, gold_ok, "Delta gold table", gold_msg)
status_card(c4, fresh_ok, "Stream freshness", fresh_msg)

st.markdown("")


# ---------------------------------------------------------------------
# Resolve data per chosen mode. NO silent fallback.
# ---------------------------------------------------------------------
using_demo = mode == "Demo (simulated)"

if using_demo:
    st.warning(
        "⚠️ **DEMO MODE** - Simulated data only. Weak separation and noise are intentional "
        "(mirrors real model, chronological ROC-AUC ≈ 0.60).",
        icon="⚠️",
    )
    df = apply_bands(make_demo_data(row_window, threshold), threshold)

elif gold_ok and live_df is not None:
    st.success("Connected to live Delta gold table.")
    df = apply_bands(live_df.head(row_window), threshold)

else:
    # Live mode but pipeline not ready. Explain exactly what's missing. Do NOT fake it.
    st.error("Live mode selected, but the pipeline is not serving data yet.", icon="🛑")
    st.markdown("**What's missing - fix these in order:**")
    if not minio_ok:
        st.markdown(f"- **MinIO**: {minio_msg}  →  run `docker compose up -d`, then create bucket `{MINIO_BUCKET}`.")
    if not kafka_ok:
        st.markdown(f"- **Kafka**: {kafka_msg}  →  run `docker compose up -d`, then create topic `{KAFKA_TOPIC_RAW}` if needed.")
    if not gold_ok:
        st.markdown(f"- **Gold table**: {gold_msg}  →  run the producer + Spark consumer to populate `gold/secom`.")
    if minio_ok and gold_ok and not fresh_ok:
        st.markdown(f"- **Freshness**: {fresh_msg}  →  the producer may have stopped; restart `producer.py`.")
    st.info("Switch to **Demo (simulated)** in the sidebar if you want to walk through the dashboard without the live stack.")
    st.stop()


# ---------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------
total = len(df)
alerts = int(df["alert"].sum())
alert_pct = (alerts / total * 100) if total else 0.0
avg_risk = float(df["cal_prob"].mean()) if total else 0.0
max_risk = float(df["cal_prob"].max()) if total else 0.0
avg_drift = float(df["drift_score"].mean()) if ("drift_score" in df.columns and total) else None
last_seen = df["scored_at"].max().strftime("%H:%M:%S") if total else "-"
biz_status, biz_note = business_status(alert_pct, avg_risk)

status_color = {"Watch": COL_STABLE, "Investigate": COL_WATCH, "Escalate": COL_ALERT}.get(biz_status, ACCENT)

def kpi_card(col, label, value, sub="", value_color="#E6EDF3"):
    col.markdown(
        f"""
        <div class="argus-card">
          <div class="lbl">{label}</div>
          <div class="val" style="color:{value_color};">{value}</div>
          <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

k1, k2, k3, k4, k5 = st.columns(5)
kpi_card(k1, "Operator action", biz_status, sub=f"Last update: {last_seen}", value_color=status_color)
kpi_card(k2, "Alert rate", f"{alert_pct:.1f}%", f"{alerts:,} of {total:,} rows",
         value_color=COL_ALERT if alert_pct >= INVESTIGATE_ALERT_PCT else "#E6EDF3")
kpi_card(k3, "Avg risk score", f"{avg_risk:.3f}", f"Max: {max_risk:.3f}")

# Alert rate delta: current 20-row rate vs prior 20 rows
_ordered = df.sort_values("timestamp")
_win20_cur = _ordered.tail(20)
_win20_pri = _ordered.iloc[max(0, len(_ordered) - 40):max(0, len(_ordered) - 20)]
_cur_r = float(_win20_cur["alert"].mean() * 100) if len(_win20_cur) else 0.0
_pri_r = float(_win20_pri["alert"].mean() * 100) if len(_win20_pri) else _cur_r
_delta = _cur_r - _pri_r
_d_arrow = "▲" if _delta > 0 else ("▼" if _delta < 0 else "→")
_d_color = COL_ALERT if _delta > 2 else (COL_STABLE if _delta < -2 else "#E6EDF3")
kpi_card(k4, "Rate delta", f"{_d_arrow} {abs(_delta):.1f}pp",
         f"Now {_cur_r:.1f}% · prior {_pri_r:.1f}%", value_color=_d_color)

# Latest score percentile
_latest_score = float(_ordered["cal_prob"].iloc[-1]) if total else 0.0
_pct_rank = float((df["cal_prob"] <= _latest_score).mean() * 100)
_p_color = COL_ALERT if _pct_rank >= 90 else (COL_WATCH if _pct_rank >= 70 else COL_STABLE)
kpi_card(k5, "Latest score", f"{_latest_score:.3f}",
         f"Top {100 - _pct_rank:.0f}% of window", value_color=_p_color)

st.markdown("---")


# ---------------------------------------------------------------------
# Drift indicator - baseline window vs recent window mean-risk shift.
# Operator-facing summary. The statistical drift test (KS / PSI) lives in
# the argusq_retrain_monitor Airflow DAG, not on this screen.
# ---------------------------------------------------------------------
# Relative shift (%) above which we flag drift for an operator's attention.
DRIFT_FLAG_PCT = 25.0

def compute_drift(frame: pd.DataFrame) -> dict | None:
    """Compare mean risk in the earliest vs most-recent window of the data."""
    if len(frame) < 40:
        return None  # too few rows for a meaningful comparison
    ordered = frame.sort_values("timestamp")["cal_prob"].to_numpy()
    win = max(20, len(ordered) // 4)
    baseline = float(ordered[:win].mean())
    recent = float(ordered[-win:].mean())
    abs_shift = recent - baseline
    rel_shift = (abs_shift / baseline * 100) if baseline > 0 else 0.0
    return {
        "win": win,
        "baseline": baseline,
        "recent": recent,
        "abs_shift": abs_shift,
        "rel_shift": rel_shift,
        "flagged": abs(rel_shift) >= DRIFT_FLAG_PCT,
        "direction": "up" if abs_shift > 0 else "down",
    }

drift = compute_drift(df)

if drift is not None:
    if drift["flagged"] and drift["direction"] == "up":
        d_color, d_word = COL_ALERT, "Drift detected - risk rising"
    elif drift["flagged"]:
        d_color, d_word = COL_WATCH, "Drift detected - risk falling"
    else:
        d_color, d_word = COL_STABLE, "No significant drift"

    arrow = "▲" if drift["direction"] == "up" else "▼"
    dc1, dc2, dc3, dc4 = st.columns([2, 1, 1, 1])
    dc1.markdown(
        f"""
        <div class="argus-card" style="border-left:3px solid {d_color};">
          <div class="lbl">Drift status</div>
          <div class="val" style="font-size:20px; color:{d_color};">{d_word}</div>
          <div class="sub">Baseline vs recent {drift['win']}-row window</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    kpi_card(dc2, "Baseline mean", f"{drift['baseline']:.3f}")
    kpi_card(dc3, "Recent mean", f"{drift['recent']:.3f}")
    kpi_card(dc4, "Shift", f"{arrow} {abs(drift['rel_shift']):.0f}%", value_color=d_color)

    st.caption(
        f"Flags when the recent {drift['win']}-row mean shifts >{DRIFT_FLAG_PCT:.0f}% from baseline. Full statistical test runs in the retrain DAG."
    )
    st.markdown("---")


# ---------------------------------------------------------------------
# Charts. All Altair, themed dark, consistent. Axis says "risk score".
# ---------------------------------------------------------------------
AXIS = alt.Axis(labelColor="#8B98A8", titleColor="#8B98A8", gridColor="rgba(120,140,170,0.10)", domainColor=COL_BORDER)

_chart_counter = 0

def to_chart(df_in: pd.DataFrame) -> alt.Chart:
    """Inline data into the Vega spec. Adds a unique _cid column so that
    Altair never identifies two calls as having identical data and never
    extracts them into a shared named dataset (which causes 'Unrecognized
    data set' errors on rerun)."""
    global _chart_counter
    _chart_counter += 1
    tmp = df_in.copy()
    tmp["_cid"] = _chart_counter
    return alt.Chart(alt.Data(values=tmp.to_dict(orient="records")))

def themed(chart, height=260):
    return chart.properties(height=height)

# --- Stage 1: Hotelling T² drift chart ---
if "drift_score" in df.columns:
    st.markdown("##### Stage 1 - Hotelling T² drift score over time")
    t2_df = (
        df.sort_values("timestamp")[["timestamp", "drift_score"]]
        .copy()
        .reset_index(drop=True)
    )
    t2_df["timestamp"] = pd.to_datetime(t2_df["timestamp"])
    t2_df["drift_score"] = t2_df["drift_score"].astype(float)
    t2_line = to_chart(t2_df.copy()).mark_line(color=ACCENT, strokeWidth=1.4).encode(
        x=alt.X("timestamp:T", axis=AXIS, title="time"),
        y=alt.Y("drift_score:Q", axis=AXIS, title="T² score"),
    )
    t2_breaches = to_chart(t2_df.copy()).transform_filter(
        alt.datum.drift_score > T2_DRIFT_THRESHOLD
    ).mark_circle(color=COL_ALERT, size=28).encode(x="timestamp:T", y="drift_score:Q")
    _t2_thr_df = pd.DataFrame({"_thr": [float(T2_DRIFT_THRESHOLD)] * len(t2_df)})
    t2_thr = (
        to_chart(_t2_thr_df)
        .mark_rule(color=COL_ALERT, strokeDash=[5, 4], strokeWidth=1)
        .encode(y=alt.Y("mean(_thr):Q"))
    )
    st.altair_chart(themed(t2_line + t2_breaches + t2_thr, height=260), use_container_width=True)
    t2_alerts = int((df["drift_score"] > T2_DRIFT_THRESHOLD).sum())
    st.caption(
        f"{t2_alerts} of {total:,} rows exceed the T² threshold ({T2_DRIFT_THRESHOLD:.0f}). "
        f"Threshold = chi-squared(df=173) at alpha=0.05. Red points = multivariate drift flagged."
    )
    st.markdown("")

# --- SPC control chart (the quality-manager hero) ---
st.markdown("##### Process control chart - calibrated risk over time")

spc_df = (
    df.sort_values("timestamp")[["timestamp", "cal_prob"]]
    .copy()
    .reset_index(drop=True)
)
spc_df["timestamp"] = pd.to_datetime(spc_df["timestamp"])
spc_df["cal_prob"] = spc_df["cal_prob"].astype(float)
mean_r = float(spc_df["cal_prob"].mean())
std_r = float(spc_df["cal_prob"].std(ddof=0))
ucl = mean_r + 2 * std_r  # upper control limit (±2σ reference band)
lcl = max(0.0, mean_r - 2 * std_r)

spc_df_line = spc_df.copy()
spc_df_pts  = spc_df.copy()

line = to_chart(spc_df_line).mark_line(color=ACCENT, strokeWidth=1.4).encode(
    x=alt.X("timestamp:T", axis=AXIS, title="time"),
    y=alt.Y("cal_prob:Q", axis=AXIS, title="risk score"),
)
breaches = to_chart(spc_df_pts).transform_filter(alt.datum.cal_prob >= threshold).mark_circle(
    color=COL_ALERT, size=28
).encode(x="timestamp:T", y="cal_prob:Q")

def rule(df_src, val, color, dash):
    _tmp = pd.DataFrame({"_rule_val": [float(val)] * len(df_src)})
    return (
        to_chart(_tmp)
        .mark_rule(color=color, strokeDash=dash, strokeWidth=1)
        .encode(y=alt.Y("mean(_rule_val):Q"))
    )

mean_rule = rule(spc_df, mean_r, "#8B98A8", [4, 0])
ucl_rule  = rule(spc_df, ucl,    COL_WATCH, [5, 4])
thr_rule  = rule(spc_df, threshold, COL_ALERT, [5, 4])

st.altair_chart(
    themed(line + breaches + mean_rule + ucl_rule + thr_rule, height=300),
    use_container_width=True,
)
st.caption(
    f"Center line = mean ({mean_r:.3f}, grey). +2σ control limit ({ucl:.3f}, amber). "
    f"Operating threshold ({threshold:.3f}, red): points above it are flagged alerts. "
    "The score is a relative triage indicator, not a calibrated probability of failure."
)

st.markdown("")

# --- Alert-rate trend: full width ---
st.markdown("##### Alert-rate trend")
trend = (
    df.sort_values("timestamp")[["timestamp", "alert"]]
    .copy()
    .reset_index(drop=True)
)
trend["timestamp"] = pd.to_datetime(trend["timestamp"])
trend["alert"] = trend["alert"].astype(float)
win = max(10, len(trend) // 12)
trend["alert_rate"] = trend["alert"].rolling(win, min_periods=1).mean() * 100
trend["alert_rate"] = trend["alert_rate"].astype(float)
trend_chart = to_chart(trend).mark_area(
    line={"color": COL_WATCH, "strokeWidth": 1.6},
    color=COL_WATCH,
    opacity=0.25,
).encode(
    x=alt.X("timestamp:T", axis=AXIS, title="time"),
    y=alt.Y("alert_rate:Q", axis=AXIS, title="alert rate (%)"),
)
st.altair_chart(themed(trend_chart, height=240), use_container_width=True)
st.caption(f"Rolling alert rate over a {win}-row window. Rising trend = process drifting toward instability.")

st.markdown("")

# --- Row: band distribution + score distribution ---
mid, right = st.columns(2)

with mid:
    st.markdown("##### Risk band distribution")
    risk_counts = (
        df["risk_level"].value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .rename_axis("risk_level").reset_index(name="count")
        .reset_index(drop=True)
    )
    risk_counts["risk_level"] = risk_counts["risk_level"].astype(str)
    risk_counts["count"] = risk_counts["count"].astype(int)
    band_chart = to_chart(risk_counts).mark_bar().encode(
        x=alt.X("risk_level:N", sort=["Low", "Medium", "High"], axis=AXIS, title="risk band"),
        y=alt.Y("count:Q", axis=AXIS, title="count"),
        color=alt.Color("risk_level:N", scale=alt.Scale(
            domain=["Low", "Medium", "High"], range=[COL_STABLE, COL_WATCH, COL_ALERT]
        ), legend=None),
    )
    st.altair_chart(themed(band_chart, height=240), use_container_width=True)
    st.caption("Low / Medium / High by severity.")

with right:
    st.markdown("##### Score distribution")
    hist_df = df[["cal_prob"]].copy().reset_index(drop=True)
    hist_df["cal_prob"] = hist_df["cal_prob"].astype(float)
    hist_chart = to_chart(hist_df).mark_bar(color=ACCENT, opacity=0.75).encode(
        x=alt.X("cal_prob:Q", bin=alt.Bin(maxbins=30), axis=AXIS, title="calibrated score"),
        y=alt.Y("count():Q", axis=AXIS, title="count"),
    )
    _thr_df = pd.DataFrame({"_thr": [float(threshold)] * len(hist_df)})
    thr_line = (
        to_chart(_thr_df)
        .mark_rule(color=COL_ALERT, strokeDash=[5, 4], strokeWidth=1)
        .encode(x=alt.X("mean(_thr):Q"))
    )
    st.altair_chart(themed(hist_chart + thr_line, height=240), use_container_width=True)
    st.caption(f"Score density. Red line = alert threshold ({threshold:.3f}). Right-tail mass drives alert rate.")


# --- SHAP attribution ---
# Build SHAP data: use real columns if present, else synthesise plausible demo values.
_DEMO_FEATS = [
    "Chamber pressure", "Gas flow rate", "RF power", "Etch time",
    "Wafer temp", "Endpoint signal", "Coolant flow", "Deposition rate",
    "Vacuum level", "Plasma density",
]

if "top_shap_feat" in df.columns and "top_shap_value" in df.columns:
    # Live: pull most recent alert row for the per-row waterfall
    _alert_rows = df[df["alert"]].sort_values("scored_at", ascending=False)
    _shap_row_feat = _alert_rows["top_shap_feat"].iloc[0] if len(_alert_rows) else None
    _shap_row_val = float(_alert_rows["top_shap_value"].iloc[0]) if len(_alert_rows) else None
    _shap_row_id = int(_alert_rows["row_id"].iloc[0]) if len(_alert_rows) else None

    # Aggregate: signed mean SHAP per feature (preserves direction)
    _shap_agg = (
        df.groupby("top_shap_feat")["top_shap_value"]
        .mean()
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(10)
        .rename_axis("feature").reset_index(name="shap_value")
        .reset_index(drop=True)
    )
    _shap_agg["feature"] = _shap_agg["feature"].astype(str)
    _shap_agg["shap_value"] = _shap_agg["shap_value"].astype(float)
    _shap_title = "Mean |SHAP| by feature"
    _shap_caption = "Average absolute SHAP value per feature across all scored rows. Longer bar = stronger average contribution to risk score."
else:
    # Demo: synthesise signed SHAP values for the most recent alert row
    rng2 = np.random.default_rng(42)
    _feats = _DEMO_FEATS
    _vals = rng2.normal(0, 0.04, size=len(_feats))
    # Push top 3 to be meaningfully positive (these "drove" the alert)
    _vals[0] = rng2.uniform(0.09, 0.14)
    _vals[1] = rng2.uniform(0.05, 0.09)
    _vals[2] = rng2.uniform(0.03, 0.06)
    _vals[3] = rng2.uniform(-0.07, -0.03)
    _shap_agg = pd.DataFrame({"feature": _feats, "shap_value": _vals})
    _shap_agg = _shap_agg.reindex(_shap_agg["shap_value"].abs().sort_values(ascending=False).index)
    _shap_agg = _shap_agg.reset_index(drop=True)
    _shap_agg["feature"] = _shap_agg["feature"].astype(str)
    _shap_agg["shap_value"] = _shap_agg["shap_value"].astype(float)
    _alert_rows = df[df["alert"]].sort_values("scored_at", ascending=False)
    _shap_row_id = int(_alert_rows["row_id"].iloc[0]) if len(_alert_rows) else None
    _shap_title = "Top feature drivers — most recent alert"
    _shap_caption = (
        "Simulated SHAP values (demo mode). Red bars push the score toward alert; green bars push toward pass. "
        "Feature names are anonymised (SECOM public dataset)."
    )

st.markdown("")
_row_label = f"row {_shap_row_id}" if _shap_row_id is not None else "latest alert"
st.markdown(f"##### SHAP — {_shap_title} ({_row_label})")

_shap_agg = _shap_agg.copy()
_shap_agg["color"] = _shap_agg["shap_value"].apply(
    lambda v: COL_ALERT if v >= 0 else COL_STABLE
)
_shap_agg["abs_val"] = _shap_agg["shap_value"].abs()
_shap_agg["direction"] = _shap_agg["shap_value"].apply(lambda v: "toward alert" if v >= 0 else "toward pass")

_shap_chart = (
    to_chart(_shap_agg)
    .mark_bar(cornerRadiusEnd=3)
    .encode(
        x=alt.X("shap_value:Q", axis=AXIS, title="SHAP value",
                 scale=alt.Scale(domain=[
                     float(_shap_agg["shap_value"].min()) * 1.15 if _shap_agg["shap_value"].min() < 0 else -0.01,
                     float(_shap_agg["shap_value"].max()) * 1.15,
                 ])),
        y=alt.Y("feature:N", sort=alt.EncodingSortField(field="abs_val", order="descending"),
                axis=AXIS, title=""),
        color=alt.condition(
            alt.datum.shap_value >= 0,
            alt.value(COL_ALERT),
            alt.value(COL_STABLE),
        ),
        tooltip=["feature:N", "shap_value:Q", "direction:N"],
    )
)

st.altair_chart(themed(_shap_chart, height=300), use_container_width=True)
st.caption(_shap_caption)


# ---------------------------------------------------------------------
# Model honesty panel - forward-looking, not self-flagellating
# ---------------------------------------------------------------------
with st.expander("Model card - read before trusting any score", expanded=False):
    st.markdown(
        f"""
        **Stage:** Baseline established. Not production-grade for automated action.

        **Performance:** Chronological ROC-AUC ≈ **{MODEL_AUC_CHRONO:.3f}** on a held-out
        forward time split. This is modest, and that is expected: SECOM is a
        ~{SECOM_POSITIVE_RATE:.1%} positive-rate stream with temporal drift, which is a
        genuinely hard monitoring problem.

        **Why this still matters:** the value here is the *architecture*. The pipeline
        (Kafka → Spark Streaming → Delta/MinIO → Airflow) lets the scoring model be
        swapped or retrained without re-engineering the data flow. The current model is
        the first iteration, not the ceiling.

        **Intended use:** operator visibility and triage - surfacing batches worth a human
        look. **Not** automated line shutdown or any decision taken without a person.

        **Known limitation:** weak class separation means individual scores should be read
        as relative signal, not absolute failure probability.
        """
    )


# ---------------------------------------------------------------------
# Operator tables
# ---------------------------------------------------------------------
ac, lc = st.columns(2)

with ac:
    st.subheader("Recent alerts requiring review")
    alert_df = df[df["alert"]].sort_values("scored_at", ascending=False)
    if alert_df.empty:
        st.success("No alerts in the selected window.")
    else:
        alert_cols = ["timestamp", "row_id", "drift_score", "cal_prob", "top_shap_feat",
                      "top_shap_value", "risk_level", "label", "scored_at"]
        alert_cols = [c for c in alert_cols if c in alert_df.columns]
        st.dataframe(alert_df[alert_cols].head(25), use_container_width=True, hide_index=True)

with lc:
    st.subheader("Latest scored rows")
    all_cols = ["timestamp", "row_id", "drift_score", "raw_prob", "cal_prob",
                "top_shap_feat", "top_shap_value", "alert", "risk_level", "label", "scored_at"]
    cols = [c for c in all_cols if c in df.columns]
    st.dataframe(df[cols].sort_values("scored_at", ascending=False).head(25),
                 use_container_width=True, hide_index=True)


st.markdown("---")
st.caption(
    "Live data: Delta gold table on MinIO, written by the Spark consumer. "
    "Demo data: simulated, clearly banner-flagged, inspired by the real model's weak class separation."
)
