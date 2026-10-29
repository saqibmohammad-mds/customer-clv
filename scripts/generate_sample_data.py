import numpy as np
import pandas as pd

from src.config import (
    CURRENCY,
    DATA_END_EXCLUSIVE,
    RANDOM_SEED,
    TRANSACTIONS_PATH,
    ensure_directories,
)


def generate_transactions(number_of_customers: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp(DATA_END_EXCLUSIVE)

    records = []
    order_number = 1

    for customer_number in range(1, number_of_customers + 1):
        customer_id = f"C{customer_number:05d}"

        joined = start + pd.Timedelta(
            days=int(rng.integers(0, 650))
        )

        purchase_probability = float(rng.uniform(0.015, 0.10))
        typical_order_value = float(
            rng.lognormal(mean=4.0, sigma=0.6)
        )
        spending_trend = float(rng.uniform(-0.20, 0.35))

        for date in pd.date_range(
            joined,
            end - pd.Timedelta(days=1),
            freq="D",
        ):
            # A modest seasonal effect.
            seasonal_multiplier = (
                1.35 if date.month in (11, 12) else 1.0
            )

            probability = min(
                purchase_probability * seasonal_multiplier,
                0.95,
            )

            if rng.random() >= probability:
                continue

            elapsed_years = (date - joined).days / 365.25
            trend_multiplier = max(
                0.30,
                1.0 + spending_trend * elapsed_years,
            )

            amount = (
                typical_order_value
                * trend_multiplier
                * rng.lognormal(mean=0.0, sigma=0.35)
            )

            records.append(
                {
                    "customer_id": customer_id,
                    "order_id": f"O{order_number:08d}",
                    "order_date": date.strftime("%Y-%m-%d"),
                    "amount": round(max(amount, 1.0), 2),
                    "currency": CURRENCY,
                }
            )
            order_number += 1

    return (
        pd.DataFrame(records)
        .sort_values(["order_date", "order_id"])
        .reset_index(drop=True)
    )


def main() -> None:
    ensure_directories()

    # Avoid accidentally replacing a real dataset.
    if TRANSACTIONS_PATH.exists():
        raise FileExistsError(
            f"{TRANSACTIONS_PATH} already exists. "
            "Move or remove it before generating sample data."
        )

    transactions = generate_transactions()
    transactions.to_csv(TRANSACTIONS_PATH, index=False)

    print(f"Saved: {TRANSACTIONS_PATH}")
    print(f"Transactions: {len(transactions):,}")
    print(
        f"Customers: "
        f"{transactions['customer_id'].nunique():,}"
    )


if __name__ == "__main__":
    main()