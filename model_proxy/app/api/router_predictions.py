from typing import List

from fastapi import APIRouter, HTTPException, Path

from app.api import crud
from app.api.models import PredictResponseSchema, PredictRequestSchema, PredictDB
from app.ml import model_predict

router = APIRouter()


@router.post("", status_code=201)
async def create_prediction(payload: PredictRequestSchema):
    text = payload.text
    output = model_predict(text)
    prediction = PredictResponseSchema(
        text=text,
        is_positive_user_answered=payload.is_positive_user_answered,
        is_positive_model_answered=output["is_positive_review"] == 1,
        mlflow_run_id=output["details"]["mlflow_run_id"],
    )
    pred_id = await crud.post(prediction)

    response_object = {
        "id": pred_id,
        "text": text,
        "is_positive": {
            "user_answered": prediction.is_positive_user_answered,
            "model_answered": prediction.is_positive_model_answered,
        },
        "details": {
            "mlflow_run_id": prediction.mlflow_run_id,
            "timestamp": prediction.timestamp,
        }
    }
    return response_object


@router.get("/{id}", response_model=PredictDB)
async def get_prediction(
    id: int = Path(..., gt=0),
):
    note = await crud.get(id)
    if not note:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return note


@router.get("", response_model=List[PredictDB])
async def get_all_predictions():
    return await crud.get_all()
