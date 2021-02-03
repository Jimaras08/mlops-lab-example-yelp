import logging

import uvicorn
from fastapi import FastAPI, HTTPException

from app.constants import MODEL_ARTIFACTS_PATH
from app.ml import load_model
from app.models import ModelInput

logger = logging.getLogger(__file__)

app = FastAPI()


MODEL = load_model(MODEL_ARTIFACTS_PATH)


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
            "text": input.dict(exclude_unset=True),
            "is_positive_review": model_output,
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
