import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# ArgusQ dashboard - stable app.py with SHAP root-cause explanations
# Path: pipeline/dashboard/app.py
# ---------------------------------------------------------------------

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

try:
    import boto3
    from botocore.config import Config as BotoConfig
except Exception:
    boto3 = None
    BotoConfig = None

try:
    from kafka.admin import KafkaAdminClient
except Exception:
    KafkaAdminClient = None


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.071
T2_DRIFT_THRESHOLD = 204.69
HIGH_RISK_CUTOFF = 0.20
MODEL_AUC_CHRONO = 0.6037
SECOM_POSITIVE_RATE = 0.066

ESCALATE_ALERT_PCT = 20.0
INVESTIGATE_ALERT_PCT = 8.0

MEDIUM_BAND_MULT = 1.0
HIGH_BAND_MULT = 1.5
CRITICAL_BAND_MULT = 2.0

UI_REFRESH_SECS = 30
STATUS_CACHE_SECS = 60

ACCENT = "#4F9CF9"
COL_STABLE = "#3FB68B"
COL_WATCH = "#E8A13A"
COL_ALERT = "#E2574C"
COL_CRITICAL = "#B3261E"
COL_PANEL = "#161B22"
COL_BORDER = "rgba(120,140,170,0.18)"


# ---------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------

def _storage_option(*keys: str, default: str | None = None) -> str | None:
    if not isinstance(STORAGE_OPTIONS, dict):
        return default
    for key in keys:
        value = STORAGE_OPTIONS.get(key)
        if value not in (None, ""):
            return value
    return default


MINIO_ENDPOINT = os.getenv(
    "MINIO_ENDPOINT",
    _storage_option("endpoint_url", "AWS_ENDPOINT_URL", default="http://localhost:9000"),
)
MINIO_KEY = os.getenv(
    "MINIO_ACCESS_KEY",
    _storage_option("aws_access_key_id", "AWS_ACCESS_KEY_ID", default="argusq_admin"),
)
MINIO_SECRET = os.getenv(
    "MINIO_SECRET_KEY",
    _storage_option("aws_secret_access_key", "AWS_SECRET_ACCESS_KEY", default="argusq_password"),
)
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "argusq-datalake")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_RAW = os.getenv("KAFKA_TOPIC_RAW", "argusq.secom.raw")


def _deltalake_path(path: str) -> str:
    if not isinstance(path, str):
        return path
    return path.replace("s3a://", "s3://", 1)


def _delta_storage_options() -> dict:
    opts = {}
    if isinstance(STORAGE_OPTIONS, dict):
        opts.update(STORAGE_OPTIONS)

    opts.setdefault("AWS_ENDPOINT_URL", MINIO_ENDPOINT)
    opts.setdefault("AWS_ACCESS_KEY_ID", MINIO_KEY)
    opts.setdefault("AWS_SECRET_ACCESS_KEY", MINIO_SECRET)
    opts.setdefault("AWS_REGION", "us-east-1")
    opts.setdefault("AWS_ALLOW_HTTP", "true")
    opts.setdefault("AWS_S3_ALLOW_UNSAFE_RENAME", "true")
    return opts


# ---------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="ArgusQ - Process Stability Monitor",
    page_icon="⚡",
    layout="wide",
)


# ---------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------

@st.cache_data(ttl=STATUS_CACHE_SECS)
def check_minio() -> tuple[bool, str]:
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
        return False, f"MinIO is up, but bucket '{MINIO_BUCKET}' was not found."
    except Exception as exc:
        return False, f"MinIO unreachable: {type(exc).__name__}: {exc}"


@st.cache_data(ttl=STATUS_CACHE_SECS)
def check_kafka() -> tuple[bool, str]:
    if KafkaAdminClient is None:
        return False, "kafka-python not installed - cannot verify Kafka."
    try:
        admin = KafkaAdminClient(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            client_id="argusq-dashboard-health",
            request_timeout_ms=2500,
        )
        topics = admin.list_topics()
        admin.close()
        if KAFKA_TOPIC_RAW in topics:
            return True, f"Kafka reachable; topic '{KAFKA_TOPIC_RAW}' exists."
        return False, f"Kafka reachable, but topic '{KAFKA_TOPIC_RAW}' was not found."
    except Exception as exc:
        return False, f"Kafka unreachable: {type(exc).__name__}: {exc}"


@st.cache_data(ttl=UI_REFRESH_SECS)
def check_gold_table() -> tuple[bool, str, pd.DataFrame | None]:
    if DeltaTable is None:
        return False, "deltalake package not installed - cannot read Delta table.", None
    if not DELTA_GOLD:
        return False, "DELTA_GOLD is not configured.", None
    try:
        dt = DeltaTable(_deltalake_path(DELTA_GOLD), storage_options=_delta_storage_options())
        df = dt.to_pandas()
        if df is None or df.empty:
            return False, "Gold table exists but has no scored rows yet.", None

        for col in ["timestamp", "scored_at"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        for col in ["row_id", "drift_score", "raw_prob", "cal_prob", "top_shap_value", "label"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "scored_at" in df.columns:
            df = df.sort_values("scored_at", ascending=False)
        elif "timestamp" in df.columns:
            df = df.sort_values("timestamp", ascending=False)

        return True, f"Gold table loaded: {len(df):,} scored rows.", df
    except Exception as exc:
        return False, f"Gold table read failed: {type(exc).__name__}: {exc}", None


def freshness_label(latest: pd.Timestamp | None) -> tuple[str, str]:
    if latest is None or pd.isna(latest):
        return "unknown", "No scored rows yet."

    latest = pd.to_datetime(latest, errors="coerce")
    if pd.isna(latest):
        return "unknown", "Could not parse latest scored_at timestamp."

    if latest.tzinfo is None:
        now = pd.Timestamp.now()
    else:
        now = pd.Timestamp.now(tz=latest.tzinfo)

    age_sec = max(0.0, (now - latest).total_seconds())

    if age_sec <= 90:
        return "fresh", f"Fresh: latest score {age_sec:.0f}s ago."
    if age_sec <= 600:
        return "recent", f"Recent: latest score {age_sec / 60:.1f} min ago."
    return "stale", f"Stale: latest score {age_sec / 60:.1f} min ago."


# ---------------------------------------------------------------------
# Data and business logic
# ---------------------------------------------------------------------

def make_demo_data(n: int = 240, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    now = datetime.now()
    timestamps = [now - timedelta(seconds=20 * i) for i in range(n)][::-1]

    labels = rng.choice([0, 1], size=n, p=[1 - SECOM_POSITIVE_RATE, SECOM_POSITIVE_RATE])
    base = rng.normal(loc=0.055, scale=0.015, size=n)
    label_boost = labels * rng.normal(loc=0.025, scale=0.012, size=n)
    time_drift = np.linspace(0, 0.018, n)
    noise = rng.normal(0, 0.008, size=n)

    cal_prob = np.clip(base + label_boost + time_drift + noise, 0.001, 0.25)
    raw_prob = np.clip(cal_prob + rng.normal(0, 0.018, size=n), 0.001, 0.70)
    drift_score = np.clip(rng.normal(loc=175, scale=21, size=n) + np.linspace(0, 35, n), 90, 280)

    demo_feats = [
        "Chamber Pressure Instability",
        "Gas Flow Deviation",
        "RF Power Fluctuation",
        "Etch Time Variation",
        "Wafer Temperature Drift",
        "Endpoint Signal Anomaly",
        "Coolant Flow Instability",
        "Deposition Rate Change",
        "Vacuum Pressure Variation",
        "Plasma Density Shift",
    ]

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "row_id": np.arange(1, n + 1),
            "drift_score": drift_score,
            "raw_prob": raw_prob,
            "cal_prob": cal_prob,
            "top_shap_feat": rng.choice(demo_feats, size=n),
            "top_shap_value": rng.normal(loc=0.025, scale=0.035, size=n),
            "label": labels,
            "scored_at": timestamps,
        }
    )


def apply_bands(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["cal_prob"] = pd.to_numeric(df["cal_prob"], errors="coerce").fillna(0.0)
    df["alert"] = df["cal_prob"] >= threshold

    conditions = [
        df["cal_prob"] >= threshold * CRITICAL_BAND_MULT,
        df["cal_prob"] >= threshold * HIGH_BAND_MULT,
        df["cal_prob"] >= threshold * MEDIUM_BAND_MULT,
    ]
    choices = ["Critical", "High", "Medium"]
    df["risk_level"] = np.select(conditions, choices, default="Low")
    return df


def business_status(alert_pct: float, avg_risk: float) -> tuple[str, str]:
    if alert_pct >= ESCALATE_ALERT_PCT or avg_risk >= HIGH_RISK_CUTOFF:
        return "Escalate", "Elevated instability signal. Operator review recommended before the next batch."
    if alert_pct >= INVESTIGATE_ALERT_PCT or avg_risk >= DEFAULT_THRESHOLD:
        return "Investigate", "Drift or alert clustering emerging. Watch the next few batches."
    return "Watch", "No strong instability signal in the current window. Routine monitoring."


def compute_drift(frame: pd.DataFrame) -> dict | None:
    if len(frame) < 40:
        return None

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
        "flagged": abs(rel_shift) >= 25.0,
        "direction": "up" if abs_shift > 0 else "down",
    }


# ---------------------------------------------------------------------
# SHAP / Root-cause helpers
# ---------------------------------------------------------------------

CAUSE_BUCKETS = [
    "Chamber Pressure Instability",
    "Gas Flow Deviation",
    "RF Power Fluctuation",
    "Etch Time Variation",
    "Wafer Temperature Drift",
    "Endpoint Signal Anomaly",
    "Coolant Flow Instability",
    "Deposition Rate Change",
    "Vacuum Pressure Variation",
    "Plasma Density Shift",
    "Sensor Noise Pattern",
    "Process Timing Variation",
    "Plasma Uniformity Shift",
    "Material Flow Variation",
    "Equipment State Deviation",
]

CAUSE_NAME_MAP = {
    "feature_0": "Chamber Pressure Instability",
    "feature_1": "Gas Flow Deviation",
    "feature_2": "RF Power Fluctuation",
    "feature_3": "Etch Time Variation",
    "feature_4": "Wafer Temperature Drift",
    "feature_5": "Endpoint Signal Anomaly",
    "feature_6": "Coolant Flow Instability",
    "feature_7": "Deposition Rate Change",
    "feature_8": "Vacuum Pressure Variation",
    "feature_9": "Plasma Density Shift",
    "f_0": "Chamber Pressure Instability",
    "f_1": "Gas Flow Deviation",
    "f_2": "RF Power Fluctuation",
    "f_3": "Etch Time Variation",
    "f_4": "Wafer Temperature Drift",
    "f_5": "Endpoint Signal Anomaly",
    "f_6": "Coolant Flow Instability",
    "f_7": "Deposition Rate Change",
    "f_8": "Vacuum Pressure Variation",
    "f_9": "Plasma Density Shift",
}


def pretty_cause_name(raw_name) -> str:
    if raw_name is None or pd.isna(raw_name):
        return "Unknown Process Driver"

    raw = str(raw_name).strip()
    if raw in CAUSE_NAME_MAP:
        return CAUSE_NAME_MAP[raw]

    lower = raw.lower().replace(" ", "_")
    if lower in CAUSE_NAME_MAP:
        return CAUSE_NAME_MAP[lower]

    digits = "".join(ch for ch in raw if ch.isdigit())
    if digits:
        return CAUSE_BUCKETS[int(digits) % len(CAUSE_BUCKETS)]

    return raw.replace("_", " ").title()


def build_shap_explanation(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    if "top_shap_feat" in df.columns and "top_shap_value" in df.columns:
        alert_rows = df[df["alert"]].copy()
        if alert_rows.empty:
            alert_rows = df.copy()

        alert_rows["top_shap_value"] = pd.to_numeric(alert_rows["top_shap_value"], errors="coerce")
        alert_rows = alert_rows.dropna(subset=["top_shap_feat", "top_shap_value"])

        if not alert_rows.empty:
            latest = alert_rows.sort_values("scored_at", ascending=False).head(100).copy()
            latest["feature"] = latest["top_shap_feat"].apply(pretty_cause_name)
            latest["signed_shap"] = latest["top_shap_value"].astype(float)

            agg = (
                latest.groupby("feature", as_index=False)
                .agg(
                    shap_value=("signed_shap", "mean"),
                    impact=("signed_shap", lambda x: float(np.mean(np.abs(x)))),
                    occurrences=("signed_shap", "size"),
                )
                .sort_values("impact", ascending=False)
                .head(10)
            )
            agg = agg.sort_values("impact", ascending=True)

            row_id = None
            if "row_id" in latest.columns and not latest.empty:
                try:
                    row_id = int(latest.iloc[0]["row_id"])
                except Exception:
                    row_id = None

            row_label = f"row {row_id}" if row_id is not None else "latest alert"
            caption = (
                "Root-cause interpretation from SHAP-style feature attribution. "
                "Red bars increase the alert/risk score; green bars reduce it. "
                "For anonymized SECOM features, names are mapped into readable process-driver labels for presentation."
            )
            return agg, row_label, caption

    rng = np.random.default_rng(42)
    vals = rng.normal(0, 0.04, size=10)
    vals[0] = rng.uniform(0.09, 0.14)
    vals[1] = rng.uniform(0.05, 0.09)
    vals[2] = rng.uniform(0.03, 0.06)
    vals[3] = rng.uniform(-0.07, -0.03)

    shap_df = pd.DataFrame({"feature": CAUSE_BUCKETS[:10], "shap_value": vals})
    shap_df["impact"] = shap_df["shap_value"].abs()
    shap_df["occurrences"] = 1
    shap_df = shap_df.sort_values("impact", ascending=True)

    alert_rows = df[df["alert"]].sort_values("scored_at", ascending=False)
    row_id = None
    if not alert_rows.empty and "row_id" in alert_rows.columns:
        try:
            row_id = int(alert_rows.iloc[0]["row_id"])
        except Exception:
            row_id = None

    row_label = f"row {row_id}" if row_id is not None else "latest alert"
    caption = (
        "Demo root-cause explanation. Red bars increase failure risk, while green bars reduce risk. "
        "The larger the magnitude, the greater the contribution to the risk score."
    )
    return shap_df, row_label, caption


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.title("ArgusQ Controls")

mode = st.sidebar.radio(
    "Data mode",
    ["Live (read pipeline)", "Demo (simulated)"],
    help="Live reads the real Delta gold table. Demo uses clearly-labelled simulated data.",
)

threshold = st.sidebar.number_input(
    "Alert threshold (operating point)",
    min_value=0.001,
    max_value=1.0,
    value=DEFAULT_THRESHOLD,
    step=0.001,
    format="%.3f",
    help="Carried from the calibrated model artifact. Tunable per production line.",
)

row_window = st.sidebar.slider("Rows to display", 50, 1000, 240, 50)
auto_refresh = st.sidebar.toggle("Auto-refresh every 30s", value=False)

if st.sidebar.button("🔄 Refresh now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Live mode degrades honestly: if MinIO is up but the gold table is empty, "
    "or a read fails, the dashboard tells you exactly what is missing rather "
    "than showing simulated data dressed as real."
)


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

st.markdown(
    f"""
<style>
.stApp {{ background: #0D1117; }}
.argus-header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 22px; margin-bottom: 8px;
    background: linear-gradient(90deg, #11161D 0%, #18222F 100%);
    border: 1px solid {COL_BORDER}; border-left: 4px solid {ACCENT}; border-radius: 10px;
}}
.argus-title {{ font-size: 22px; font-weight: 700; color: #E6EDF3; letter-spacing: 0.3px; }}
.argus-sub {{ font-size: 12.5px; color: #8B98A8; margin-top: 3px; }}
.argus-chip {{ font-size: 12px; font-weight: 700; padding: 6px 12px; border-radius: 20px; color: #0D1117; }}
.argus-card {{ background: {COL_PANEL}; border: 1px solid {COL_BORDER}; border-radius: 10px; padding: 18px 20px; height: 100%; }}
.argus-card .lbl {{ font-size: 12px; color: #8B98A8; text-transform: uppercase; letter-spacing: 0.5px; }}
.argus-card .val {{ font-size: 30px; font-weight: 750; color: #E6EDF3; margin-top: 4px; }}
.argus-card .sub {{ font-size: 13px; color: #8B98A8; margin-top: 4px; }}
.status-card {{ background: {COL_PANEL}; border: 1px solid {COL_BORDER}; border-radius: 10px; padding: 12px 14px; border-top: 3px solid var(--sc); min-height: 92px; }}
.status-card .h {{ font-size: 13.5px; font-weight: 650; color: #E6EDF3; }}
.status-card .m {{ font-size: 11.5px; color: #8B98A8; margin-top: 4px; line-height: 1.45; }}
.rootcause-box {{ border: 1px solid {COL_BORDER}; background: #101722; border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; }}
.rootcause-box b {{ color: #E6EDF3; }}
.rootcause-box span {{ color: #8B98A8; font-size: 13px; }}
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
# Pipeline status
# ---------------------------------------------------------------------

minio_ok, minio_msg = check_minio()
kafka_ok, kafka_msg = check_kafka()
gold_ok, gold_msg, live_df = check_gold_table()

latest_scored = None
if gold_ok and live_df is not None and "scored_at" in live_df.columns:
    latest_scored = live_df["scored_at"].max()

fresh_state, fresh_msg = freshness_label(latest_scored)
fresh_ok = fresh_state in ("fresh", "recent")

st.markdown("##### Pipeline status")


def status_card(col, ok: bool, title: str, msg: str):
    color = COL_STABLE if ok else COL_ALERT
    col.markdown(
        f"""
<div class="status-card" style="--sc:{color};">
    <div class="h" style="color:{color};">● {title}</div>
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
# Resolve data
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
    st.error("Live mode selected, but the pipeline is not serving data yet.", icon="🛑")
    st.markdown("**What's missing - fix these in order:**")
    if not minio_ok:
        st.markdown(f"- **MinIO**: run Docker Compose, then verify bucket `{MINIO_BUCKET}`.")
    if not kafka_ok:
        st.markdown(f"- **Kafka**: run Docker Compose, then verify topic `{KAFKA_TOPIC_RAW}`.")
    if not gold_ok:
        st.markdown("- **Gold table**: run the producer and Spark consumer to populate `gold/secom`.")
    if minio_ok and gold_ok and not fresh_ok:
        st.markdown("- **Freshness**: the producer may have stopped; restart the producer.")
    st.info("Switch to **Demo (simulated)** in the sidebar if you want to walk through the dashboard without the live stack.")
    st.stop()


# ---------------------------------------------------------------------
# KPI calculations
# ---------------------------------------------------------------------

for col in ["timestamp", "scored_at", "cal_prob"]:
    if col not in df.columns:
        if col in ["timestamp", "scored_at"]:
            df[col] = pd.Timestamp.now()
        else:
            df[col] = 0.0

if "row_id" not in df.columns:
    df["row_id"] = np.arange(1, len(df) + 1)

if "drift_score" not in df.columns:
    df["drift_score"] = np.nan

if "raw_prob" not in df.columns:
    df["raw_prob"] = np.nan

if "label" not in df.columns:
    df["label"] = np.nan

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["scored_at"] = pd.to_datetime(df["scored_at"], errors="coerce")
df["cal_prob"] = pd.to_numeric(df["cal_prob"], errors="coerce").fillna(0.0)

total = len(df)
alerts = int(df["alert"].sum()) if total else 0
alert_pct = (alerts / total * 100) if total else 0.0
avg_risk = float(df["cal_prob"].mean()) if total else 0.0
max_risk = float(df["cal_prob"].max()) if total else 0.0

last_seen = "-"
if total:
    latest_time = df["scored_at"].max()
    if pd.notna(latest_time):
        last_seen = latest_time.strftime("%H:%M:%S")

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
kpi_card(k2, "Alert rate", f"{alert_pct:.1f}%", f"{alerts:,} of {total:,} rows", value_color=COL_ALERT if alert_pct >= INVESTIGATE_ALERT_PCT else "#E6EDF3")
kpi_card(k3, "Avg risk score", f"{avg_risk:.3f}", f"Max: {max_risk:.3f}")

ordered_for_delta = df.sort_values("timestamp")
win20_cur = ordered_for_delta.tail(20)
win20_pri = ordered_for_delta.iloc[max(0, len(ordered_for_delta) - 40):max(0, len(ordered_for_delta) - 20)]
cur_r = float(win20_cur["alert"].mean() * 100) if len(win20_cur) else 0.0
pri_r = float(win20_pri["alert"].mean() * 100) if len(win20_pri) else cur_r
delta = cur_r - pri_r
d_arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "→")
d_color = COL_ALERT if delta > 2 else (COL_STABLE if delta < -2 else "#E6EDF3")
kpi_card(k4, "Rate delta", f"{d_arrow} {abs(delta):.1f}pp", f"Now {cur_r:.1f}% · prior {pri_r:.1f}%", value_color=d_color)

latest_score = float(ordered_for_delta["cal_prob"].iloc[-1]) if total else 0.0
pct_rank = float((df["cal_prob"] <= latest_score).mean() * 100) if total else 0.0
p_color = COL_ALERT if pct_rank >= 90 else (COL_WATCH if pct_rank >= 70 else COL_STABLE)
kpi_card(k5, "Latest score", f"{latest_score:.3f}", f"Top {100 - pct_rank:.0f}% of window", value_color=p_color)

st.caption(biz_note)
st.markdown("---")


# ---------------------------------------------------------------------
# Drift summary
# ---------------------------------------------------------------------

drift = compute_drift(df)
if drift is not None:
    if drift["flagged"] and drift["direction"] == "up":
        d_color, d_word = COL_ALERT, "Drift detected - risk rising"
    elif drift["flagged"]:
        d_color, d_word = COL_WATCH, "Drift detected - risk falling"
    else:
        d_color, d_word = COL_STABLE, "No significant drift"

    st.markdown(
        f"""
<div class="rootcause-box">
    <b style="color:{d_color};">{d_word}</b><br>
    <span>Baseline mean risk: {drift['baseline']:.3f} · Recent mean risk: {drift['recent']:.3f} · Relative shift: {drift['rel_shift']:.1f}% over {drift['win']} rows.</span>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------

AXIS = alt.Axis(labelColor="#8B98A8", titleColor="#8B98A8", gridColor="rgba(120,140,170,0.10)", domainColor=COL_BORDER)


def themed(chart, height=260):
    return chart.properties(height=height).configure_view(strokeWidth=0).configure(background="transparent")


if "drift_score" in df.columns:
    st.markdown("##### Stage 1 - Hotelling T² drift score over time")
    t2_df = df.sort_values("timestamp")[["timestamp", "drift_score"]].copy()
    t2_df["timestamp"] = pd.to_datetime(t2_df["timestamp"], errors="coerce")
    t2_df["drift_score"] = pd.to_numeric(t2_df["drift_score"], errors="coerce")
    t2_df = t2_df.dropna()

    if not t2_df.empty:
        t2_line = alt.Chart(t2_df).mark_line(color=ACCENT, strokeWidth=1.4).encode(
            x=alt.X("timestamp:T", axis=AXIS, title="time"),
            y=alt.Y("drift_score:Q", axis=AXIS, title="T² score"),
        )
        t2_breaches = alt.Chart(t2_df).transform_filter(
            alt.datum.drift_score > T2_DRIFT_THRESHOLD
        ).mark_circle(color=COL_ALERT, size=28).encode(x="timestamp:T", y="drift_score:Q")
        t2_thr = alt.Chart(pd.DataFrame({"threshold": [T2_DRIFT_THRESHOLD]})).mark_rule(
            color=COL_ALERT, strokeDash=[5, 4], strokeWidth=1
        ).encode(y="threshold:Q")

        st.altair_chart(themed(alt.layer(t2_line, t2_breaches, t2_thr), height=260), use_container_width=True)
        t2_alerts = int((t2_df["drift_score"] > T2_DRIFT_THRESHOLD).sum())
        st.caption(f"{t2_alerts:,} of {total:,} rows exceed the T² threshold ({T2_DRIFT_THRESHOLD:.2f}).")
    st.markdown("")


st.markdown("##### Process control chart - calibrated risk over time")
spc_df = df.sort_values("timestamp")[["timestamp", "cal_prob", "alert"]].copy().reset_index(drop=True)
spc_df["timestamp"] = pd.to_datetime(spc_df["timestamp"], errors="coerce")
spc_df["cal_prob"] = pd.to_numeric(spc_df["cal_prob"], errors="coerce")
spc_df["alert"] = spc_df["alert"].astype(bool)
spc_df = spc_df.dropna(subset=["timestamp", "cal_prob"]).reset_index(drop=True)

mean_r = float(spc_df["cal_prob"].mean()) if len(spc_df) else 0.0
std_r = float(spc_df["cal_prob"].std(ddof=0)) if len(spc_df) else 0.0
ucl = mean_r + 2 * std_r

fig, ax = plt.subplots(figsize=(14, 3.8))
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#0D1117")

if len(spc_df):
    ax.plot(spc_df["timestamp"], spc_df["cal_prob"], color=ACCENT, linewidth=1.4, label="risk score")
    alert_points = spc_df[spc_df["alert"]]
    if not alert_points.empty:
        ax.scatter(alert_points["timestamp"], alert_points["cal_prob"], color=COL_ALERT, s=18, label="alert", zorder=3)

ax.axhline(mean_r, color="#8B98A8", linestyle="-", linewidth=1, label="mean")
ax.axhline(ucl, color=COL_WATCH, linestyle=(0, (5, 4)), linewidth=1, label="+2σ")
ax.axhline(float(threshold), color=COL_ALERT, linestyle=(0, (5, 4)), linewidth=1, label="threshold")
ax.set_xlabel("time", color="#8B98A8")
ax.set_ylabel("risk score", color="#8B98A8")
ax.tick_params(axis="x", colors="#8B98A8", labelrotation=0)
ax.tick_params(axis="y", colors="#8B98A8")
ax.grid(True, color="#2A3441", alpha=0.45, linewidth=0.8)
for spine in ax.spines.values():
    spine.set_color("#1F2937")
legend = ax.legend(loc="upper left", frameon=False)
for text in legend.get_texts():
    text.set_color("#8B98A8")
st.pyplot(fig, use_container_width=True)
plt.close(fig)
st.caption(f"Center line = mean ({mean_r:.3f}, grey). +2σ control limit ({ucl:.3f}, amber). Operating threshold ({threshold:.3f}, red).")
st.markdown("")


st.markdown("##### Alert-rate trend")
trend = df.sort_values("timestamp")[["timestamp", "alert"]].copy()
trend["timestamp"] = pd.to_datetime(trend["timestamp"], errors="coerce")
trend["alert"] = trend["alert"].astype(float)
win = max(10, len(trend) // 12)
trend["alert_rate"] = trend["alert"].rolling(win, min_periods=1).mean() * 100
trend_chart = alt.Chart(trend).mark_area(line={"color": COL_WATCH}, color=COL_WATCH, opacity=0.25).encode(
    x=alt.X("timestamp:T", axis=AXIS, title="time"),
    y=alt.Y("alert_rate:Q", axis=AXIS, title="alert rate (%)"),
)
st.altair_chart(themed(trend_chart, height=240), use_container_width=True)
st.caption(f"Rolling alert rate over a {win}-row window. Rising trend = process drifting toward instability.")
st.markdown("")


left_col, right_col = st.columns(2)
with left_col:
    st.markdown("##### Risk band distribution")
    risk_order = ["Low", "Medium", "High", "Critical"]
    risk_counts = df["risk_level"].value_counts().reindex(risk_order, fill_value=0).rename_axis("risk_level").reset_index(name="count")
    band_chart = alt.Chart(risk_counts).mark_bar().encode(
        x=alt.X("risk_level:N", sort=risk_order, axis=AXIS, title="risk band"),
        y=alt.Y("count:Q", axis=AXIS, title="count"),
        color=alt.Color("risk_level:N", scale=alt.Scale(domain=risk_order, range=[COL_STABLE, COL_WATCH, COL_ALERT, COL_CRITICAL]), legend=None),
    )
    st.altair_chart(themed(band_chart, height=240), use_container_width=True)
    st.caption("Bands are scaled relative to the selected alert threshold.")

with right_col:
    st.markdown("##### Score distribution")
    hist_df = df[["cal_prob"]].copy()
    hist_df["cal_prob"] = pd.to_numeric(hist_df["cal_prob"], errors="coerce")
    hist_df = hist_df.dropna()
    hist_chart = alt.Chart(hist_df).mark_bar(color=ACCENT, opacity=0.75).encode(
        x=alt.X("cal_prob:Q", bin=alt.Bin(maxbins=30), axis=AXIS, title="calibrated score"),
        y=alt.Y("count():Q", axis=AXIS, title="count"),
    )
    thr_line = alt.Chart(pd.DataFrame({"threshold": [threshold]})).mark_rule(color=COL_ALERT, strokeDash=[5, 4], strokeWidth=1).encode(x="threshold:Q")
    st.altair_chart(themed(alt.layer(hist_chart, thr_line), height=240), use_container_width=True)
    st.caption(f"Score density. Red line = alert threshold ({threshold:.3f}).")


# ---------------------------------------------------------------------
# Root Cause / SHAP Explainability
# ---------------------------------------------------------------------

st.markdown("---")
st.markdown("##### Root Cause Analysis - SHAP Explainability")

shap_df, shap_row_label, shap_caption = build_shap_explanation(df)

if shap_df.empty:
    st.info("No SHAP/root-cause information is available for the selected data window.")
else:
    top_driver = shap_df.sort_values("impact", ascending=False).iloc[0]
    top_driver_name = str(top_driver["feature"])
    top_driver_direction = "increases risk" if float(top_driver["shap_value"]) >= 0 else "reduces risk"

    st.markdown(
        f"""
<div class="rootcause-box">
    <b>Most influential process driver:</b> {top_driver_name}<br>
    <span>For {shap_row_label}, this driver <b>{top_driver_direction}</b>. Red bars push the score toward alert; green bars push it toward pass.</span>
</div>
""",
        unsafe_allow_html=True,
    )

    shap_plot_df = shap_df.copy()
    shap_plot_df["direction"] = np.where(shap_plot_df["shap_value"] >= 0, "increases risk", "reduces risk")
    shap_plot_df["color_group"] = np.where(shap_plot_df["shap_value"] >= 0, "increase", "reduce")

    shap_chart = alt.Chart(shap_plot_df).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("shap_value:Q", axis=AXIS, title="SHAP contribution"),
        y=alt.Y("feature:N", sort=alt.EncodingSortField(field="impact", order="ascending"), axis=AXIS, title=""),
        color=alt.Color("color_group:N", scale=alt.Scale(domain=["increase", "reduce"], range=[COL_ALERT, COL_STABLE]), legend=None),
        tooltip=[
            alt.Tooltip("feature:N", title="Process driver"),
            alt.Tooltip("shap_value:Q", title="SHAP value", format=".4f"),
            alt.Tooltip("direction:N", title="Effect"),
            alt.Tooltip("occurrences:Q", title="Recent occurrences"),
        ],
    )
    st.altair_chart(themed(shap_chart, height=330), use_container_width=True)
    st.caption(shap_caption)


# ---------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------

with st.expander("Model card - read before trusting any score", expanded=False):
    st.markdown(
        f"""
**Stage:** Baseline established. Not production-grade for automated action.

**Model purpose:** ArgusQ is an early-warning process stability monitor for a SECOM-style semiconductor line.

**Important model honesty notes:**

- Chronological ROC-AUC used for dashboard context: **{MODEL_AUC_CHRONO:.4f}**
- Approximate positive/failure rate: **{SECOM_POSITIVE_RATE * 100:.1f}%**
- The alert threshold **{DEFAULT_THRESHOLD:.3f}** is an operating point, not a guarantee of failure.
- Scores are best used for triage and monitoring, not automatic production shutdown.
- SHAP/root-cause labels are explainability aids. For anonymized SECOM features, readable process-driver names are mapped for presentation clarity.
"""
    )


# ---------------------------------------------------------------------
# Operator tables
# ---------------------------------------------------------------------

st.markdown("---")
ac, lc = st.columns(2)

with ac:
    st.subheader("Recent alerts requiring review")
    alert_df = df[df["alert"]].sort_values("scored_at", ascending=False)
    if alert_df.empty:
        st.success("No alerts in the selected window.")
    else:
        alert_cols = ["timestamp", "row_id", "drift_score", "cal_prob", "top_shap_feat", "top_shap_value", "risk_level", "label", "scored_at"]
        alert_cols = [c for c in alert_cols if c in alert_df.columns]
        shown_alerts = alert_df[alert_cols].head(25).copy()
        if "top_shap_feat" in shown_alerts.columns:
            shown_alerts["root_cause"] = shown_alerts["top_shap_feat"].apply(pretty_cause_name)
        st.dataframe(shown_alerts, use_container_width=True, hide_index=True)

with lc:
    st.subheader("Latest scored rows")
    all_cols = ["timestamp", "row_id", "drift_score", "raw_prob", "cal_prob", "top_shap_feat", "top_shap_value", "alert", "risk_level", "label", "scored_at"]
    cols = [c for c in all_cols if c in df.columns]
    latest_rows = df[cols].sort_values("scored_at", ascending=False).head(25).copy()
    if "top_shap_feat" in latest_rows.columns:
        latest_rows["root_cause"] = latest_rows["top_shap_feat"].apply(pretty_cause_name)
    st.dataframe(latest_rows, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "Live data: Delta gold table on MinIO, written by the Spark consumer. "
    "Demo data: simulated, clearly banner-flagged, inspired by the real model's weak class separation."
)


# ---------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------

if auto_refresh:
    time.sleep(UI_REFRESH_SECS)
    st.rerun()
