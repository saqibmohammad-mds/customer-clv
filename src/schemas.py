from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


NonNegativeFloat = Annotated[float, Field(ge=0)]


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    recency_days: NonNegativeFloat
    tenure_days: NonNegativeFloat
    orders_180d: int = Field(ge=0)
    spend_180d: NonNegativeFloat
    avg_order_value_180d: NonNegativeFloat
    active_months_180d: int = Field(ge=0, le=7)
    spend_last_30d: NonNegativeFloat
    spend_previous_30d: NonNegativeFloat
    spend_change_30d: float

    @model_validator(mode="after")
    def validate_consistency(self):
        if self.recency_days > self.tenure_days:
            raise ValueError(
                "recency_days cannot exceed tenure_days."
            )

        if (
            self.spend_last_30d + self.spend_previous_30d
            > self.spend_180d + 0.01
        ):
            raise ValueError(
                "Combined spending in the last 60 days "
                "cannot exceed spending in the last 180 days."
            )

        expected_change = (
            self.spend_last_30d - self.spend_previous_30d
        )

        if abs(self.spend_change_30d - expected_change) > 0.01:
            raise ValueError(
                "spend_change_30d must equal spend_last_30d "
                "minus spend_previous_30d."
            )

        return self


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1, max_length=100)
    features: CustomerFeatures


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customers: list[PredictionRequest] = Field(
        min_length=1,
        max_length=100,
    )


class PredictionResponse(BaseModel):
    customer_id: str
    predicted_clv_90d: float = Field(ge=0)
    currency: str
    prediction_horizon_days: int
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]