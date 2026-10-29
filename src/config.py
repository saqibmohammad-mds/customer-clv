from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"

TRANSACTIONS_PATH = RAW_DIR / "transactions.csv"
DATASET_PATH = PROCESSED_DIR / "customer_features.csv"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"

RANDOM_SEED = 42
CURRENCY = "USD"
PREDICTION_HORIZON_DAYS = 90
FEATURE_LOOKBACK_DAYS = 180

# Exclusive end of the period for which transaction coverage is complete.
# When using real data, set this to the correct coverage boundary.
DATA_END_EXCLUSIVE = "2024-07-01"

SNAPSHOTS = {
    "train": [
        "2023-01-01",
        "2023-04-01",
        "2023-07-01",
        "2023-09-01",
    ],
    "validation": ["2024-01-01"],
    "test": ["2024-04-01"],
}

FEATURE_COLUMNS = [
    "recency_days",
    "tenure_days",
    "orders_180d",
    "spend_180d",
    "avg_order_value_180d",
    "active_months_180d",
    "spend_last_30d",
    "spend_previous_30d",
    "spend_change_30d",
]

TARGET_COLUMN = "clv_90d"


def ensure_directories() -> None:
    for directory in (
        RAW_DIR,
        PROCESSED_DIR,
        ARTIFACT_DIR,
        REPORT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)