import hashlib
import json

import pandas as pd

from src.config import (
    CURRENCY,
    DATA_END_EXCLUSIVE,
    DATASET_PATH,
    FEATURE_COLUMNS,
    MANIFEST_PATH,
    PREDICTION_HORIZON_DAYS,
    SNAPSHOTS,
    TARGET_COLUMN,
    TRANSACTIONS_PATH,
    ensure_directories,
)
from src.features import (
    build_training_snapshot,
    validate_transactions,
)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_split_boundaries() -> None:
    horizon = pd.Timedelta(days=PREDICTION_HORIZON_DAYS)

    for earlier, later in (
        ("train", "validation"),
        ("validation", "test"),
    ):
        earlier_labels_available = (
            max(pd.Timestamp(date) for date in SNAPSHOTS[earlier])
            + horizon
        )
        later_start = min(
            pd.Timestamp(date) for date in SNAPSHOTS[later]
        )

        if earlier_labels_available > later_start:
            raise ValueError(
                f"{earlier} labels extend beyond the start of {later}."
            )


def main() -> None:
    ensure_directories()
    check_split_boundaries()

    if not TRANSACTIONS_PATH.exists():
        raise FileNotFoundError(
            "Transaction data is missing. Run: "
            "python -m scripts.generate_sample_data"
        )

    raw = pd.read_csv(
        TRANSACTIONS_PATH,
        dtype={
            "customer_id": "string",
            "order_id": "string",
            "currency": "string",
        },
    )
    transactions = validate_transactions(raw)

    if transactions["order_date"].ge(
        pd.Timestamp(DATA_END_EXCLUSIVE)
    ).any():
        raise ValueError(
            "Transactions extend beyond DATA_END_EXCLUSIVE. "
            "Update the coverage configuration."
        )

    snapshots = []

    for split, dates in SNAPSHOTS.items():
        for date in dates:
            snapshot = build_training_snapshot(
                transactions,
                snapshot_date=date,
                coverage_end=DATA_END_EXCLUSIVE,
            )
            snapshot["split"] = split
            snapshots.append(snapshot)

    dataset = pd.concat(snapshots, ignore_index=True)

    if dataset.duplicated(
        subset=["customer_id", "snapshot_date"]
    ).any():
        raise ValueError("Duplicate customer snapshots detected.")

    dataset.to_csv(DATASET_PATH, index=False)

    manifest = {
        "schema_version": 1,
        "currency": CURRENCY,
        "prediction_horizon_days": PREDICTION_HORIZON_DAYS,
        "coverage_end_exclusive": DATA_END_EXCLUSIVE,
        "source_sha256": sha256_file(TRANSACTIONS_PATH),
        "dataset_sha256": sha256_file(DATASET_PATH),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "snapshots": SNAPSHOTS,
        "row_counts": {
            key: int(value)
            for key, value in dataset["split"].value_counts().items()
        },
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    summary = dataset.groupby("split").agg(
        rows=("customer_id", "size"),
        customers=("customer_id", "nunique"),
        mean_clv=(TARGET_COLUMN, "mean"),
    )

    print(f"Saved dataset: {DATASET_PATH}")
    print(f"Saved manifest: {MANIFEST_PATH}")
    print()
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()