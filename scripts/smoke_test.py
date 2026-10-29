import os
import time

import httpx


def main():
    base_url = os.environ["CLV_BASE_URL"].rstrip("/")
    api_key = os.environ["CLV_API_KEY"]

    payload = {
        "customer_id": "deployment-check",
        "features": {
            "recency_days": 10,
            "tenure_days": 400,
            "orders_180d": 12,
            "spend_180d": 1200,
            "avg_order_value_180d": 100,
            "active_months_180d": 6,
            "spend_last_30d": 250,
            "spend_previous_30d": 200,
            "spend_change_30d": 50,
        },
    }

    last_error = None

    with httpx.Client(timeout=60) as client:
        for attempt in range(12):
            try:
                health = client.get(f"{base_url}/health")
                health.raise_for_status()

                response = client.post(
                    f"{base_url}/predict",
                    headers={"X-API-Key": api_key},
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()

                if result["predicted_clv_90d"] < 0:
                    raise ValueError("Negative prediction.")

                if result["prediction_horizon_days"] != 90:
                    raise ValueError("Unexpected prediction horizon.")

                unauthorized = client.post(
                    f"{base_url}/predict",
                    json=payload,
                )

                if unauthorized.status_code != 401:
                    raise ValueError("Authentication check failed.")

                print("Deployed API and authentication checks passed.")
                return

            except (httpx.HTTPError, ValueError, KeyError) as error:
                last_error = error
                if attempt < 11:
                    time.sleep(10)

    raise RuntimeError(
        "Deployed API check failed after retries."
    ) from last_error


if __name__ == "__main__":
    main()