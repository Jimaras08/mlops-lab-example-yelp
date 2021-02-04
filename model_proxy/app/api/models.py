from datetime import datetime

from pydantic import BaseModel, Field


class PredictRequestSchema(BaseModel):
    text: str = Field(..., min_length=5, max_length=1024 * 5)
    is_positive_user_answered: bool


class PredictResponseSchema(PredictRequestSchema):
    is_positive_model_answered: bool
    model_run_id: str = Field(..., min_length=32, max_length=32)
    timestamp: datetime

class PredictDB(PredictResponseSchema):
    id: int
