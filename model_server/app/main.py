import logging

import uvicorn
from fastapi import FastAPI, HTTPException

from app.constants import MLFLOW_MODEL_ARTIFACTS_PATH, MLFLOW_MODEL_RUN_ID
from app.ml import load_model
from app.models import ModelInput

logger = logging.getLogger(__file__)

app = FastAPI()


MODEL = None


@app.on_event("startup")
async def startup():
    global MODEL
    MODEL = load_model(MLFLOW_MODEL_ARTIFACTS_PATH)


@app.get("/ping")
def ping():
    return {"ping": "pong"}


@app.post("/predict")
async def predict(input: ModelInput):
    try:
        model_output = MODEL.predict(input.text)
    except BaseException as e:
        detail = f"Unknown model prediction error: {e}"
        logger.exception(detail)
        raise HTTPException(status_code=500, detail=detail)
    else:
        return {
            "text": input.text,
            "is_positive_review": model_output,
            "details": {
                "mlflow_run_id": MLFLOW_MODEL_RUN_ID,
            },
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
