import numpy as np
import pandas as pd

from src.config import (
    CURRENCY,
    FEATURE_COLUMNS,
    FEATURE_LOOKBACK_DAYS,
    PREDICTION_HORIZON_DAYS,
    TARGET_COLUMN,
)


REQUIRED_COLUMNS = [
    "customer_id",
    "order_id",
    "order_date",
    "amount",
    "currency",
]


def validate_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Missing transaction columns: {sorted(missing)}"
        )

    result = frame[REQUIRED_COLUMNS].copy()

    if result.empty:
        raise ValueError("The transaction dataset is empty.")

    if result.isna().any().any():
        raise ValueError("Transactions contain missing values.")

    for column in ("customer_id", "order_id", "currency"):
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise ValueError(f"{column} contains empty values.")

    if result["order_id"].duplicated().any():
        raise ValueError(
            "order_id must be unique. Aggregate line items "
            "into one row per order before using this pipeline."
        )

    result["order_date"] = pd.to_datetime(
        result["order_date"],
        errors="raise",
        utc=True,
    ).dt.tz_convert(None)

    result["amount"] = pd.to_numeric(
        result["amount"],
        errors="raise",
    )

    if not np.isfinite(result["amount"].to_numpy()).all():
        raise ValueError("Amounts must be finite numbers.")

    if result["amount"].le(0).any():
        raise ValueError(
            "This schema accepts positive completed purchases only."
        )

    if not result["currency"].eq(CURRENCY).all():
        raise ValueError(
            f"All purchases must be in {CURRENCY}. "
            "Convert currencies before preparing the dataset."
        )

    return result.sort_values("order_date").reset_index(drop=True)


def build_features(
    transactions: pd.DataFrame,
    snapshot_date: str | pd.Timestamp,
) -> pd.DataFrame:
    cutoff = pd.Timestamp(snapshot_date)

    history = transactions.loc[
        transactions["order_date"] < cutoff
    ].copy()

    if history.empty:
        raise ValueError(
            f"No purchase history exists before {cutoff.date()}."
        )

    # Every customer observed before the cutoff is eligible,
    # including customers inactive during the recent window.
    customers = pd.Index(
        sorted(history["customer_id"].unique()),
        name="customer_id",
    )
    features = pd.DataFrame(index=customers)

    lifetime = history.groupby("customer_id")["order_date"]

    features["recency_days"] = (
        cutoff - lifetime.max()
    ).dt.total_seconds() / 86400

    features["tenure_days"] = (
        cutoff - lifetime.min()
    ).dt.total_seconds() / 86400

    recent_start = cutoff - pd.Timedelta(
        days=FEATURE_LOOKBACK_DAYS
    )

    recent = history.loc[
        history["order_date"] >= recent_start
    ].copy()

    recent["purchase_month"] = (
        recent["order_date"].dt.to_period("M").astype(str)
    )

    grouped = recent.groupby("customer_id")

    features["orders_180d"] = grouped["order_id"].nunique()
    features["spend_180d"] = grouped["amount"].sum()
    features["avg_order_value_180d"] = grouped["amount"].mean()
    features["active_months_180d"] = (
        grouped["purchase_month"].nunique()
    )

    last_30_start = cutoff - pd.Timedelta(days=30)
    previous_30_start = cutoff - pd.Timedelta(days=60)

    last_30 = history.loc[
        history["order_date"] >= last_30_start
    ]

    previous_30 = history.loc[
        (history["order_date"] >= previous_30_start)
        & (history["order_date"] < last_30_start)
    ]

    features["spend_last_30d"] = (
        last_30.groupby("customer_id")["amount"].sum()
    )
    features["spend_previous_30d"] = (
        previous_30.groupby("customer_id")["amount"].sum()
    )

    features = features.fillna(0.0)

    features["spend_change_30d"] = (
        features["spend_last_30d"]
        - features["spend_previous_30d"]
    )

    features = features[FEATURE_COLUMNS].astype(float)

    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Generated features contain invalid values.")

    return features.reset_index()


def build_training_snapshot(
    transactions: pd.DataFrame,
    snapshot_date: str,
    coverage_end: str,
) -> pd.DataFrame:
    cutoff = pd.Timestamp(snapshot_date)
    target_end = cutoff + pd.Timedelta(
        days=PREDICTION_HORIZON_DAYS
    )

    if target_end > pd.Timestamp(coverage_end):
        raise ValueError(
            f"Snapshot {snapshot_date} has an incomplete "
            f"{PREDICTION_HORIZON_DAYS}-day outcome window."
        )

    features = build_features(transactions, cutoff)

    future = transactions.loc[
        (transactions["order_date"] >= cutoff)
        & (transactions["order_date"] < target_end)
    ]

    targets = future.groupby("customer_id")["amount"].sum()

    features[TARGET_COLUMN] = (
        features["customer_id"]
        .map(targets)
        .fillna(0.0)
    )

    features["snapshot_date"] = cutoff.strftime("%Y-%m-%d")
    features["target_end_date"] = target_end.strftime("%Y-%m-%d")

    return features