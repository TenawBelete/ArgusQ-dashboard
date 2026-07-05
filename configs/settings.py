import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# MinIO
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "argusq_admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "argusq_password")
MINIO_BUCKET     = os.getenv("MINIO_BUCKET",     "argusq-datalake")

STORAGE_OPTIONS = {
    "aws_access_key_id":          MINIO_ACCESS_KEY,
    "aws_secret_access_key":      MINIO_SECRET_KEY,
    "endpoint_url":               MINIO_ENDPOINT,
    "region_name":                "us-east-1",
    "aws_allow_http":             "true",
    "aws_s3_allow_unsafe_rename": "true",
}

DELTA_BRONZE = f"s3a://{MINIO_BUCKET}/bronze"
DELTA_SILVER = f"s3a://{MINIO_BUCKET}/silver"
DELTA_GOLD   = f"s3a://{MINIO_BUCKET}/gold"

# Kafka
KAFKA_BROKER       = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC_RAW    = "argusq.secom.raw"
KAFKA_TOPIC_SCORED = "argusq.secom.scored"
KAFKA_GROUP_ID     = "argusq-spark-consumer"
KAFKA_AUTO_OFFSET  = "earliest"

# Spark
SPARK_APP_NAME      = "ArgusQ"
SPARK_MASTER        = os.getenv("SPARK_MASTER", "local[*]")
SPARK_SHUFFLE_PARTS = 8

# Data paths
RAW_SECOM          = BASE_DIR / "data" / "processed" / "secom" / "secom_scaled_stream.csv"
MODELS_DIR         = BASE_DIR / "models"
SECOM_DRIFT_MODEL  = MODELS_DIR / "secom_t2.joblib"
SECOM_RISK_MODEL   = MODELS_DIR / "secom_xgb.joblib"
SECOM_PREPROCESSOR = MODELS_DIR / "secom_preprocessor.joblib"

# Streaming
STREAM_INTERVAL_SECS = 2
STREAM_TRIGGER_MS    = 5000

# Domain
DOMAIN        = "secom"
LABEL_COL     = "label"
TIMESTAMP_COL = "timestamp"
SPLIT_COL     = "split"