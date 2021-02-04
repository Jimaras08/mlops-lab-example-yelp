import logging

import requests

logger = logging.getLogger()

from app.constants import MODEL_URI


def model_predict(text: str):
    r = requests.post(f"{MODEL_URI}/predict", json={"text": text})
    r.raise_for_status()
    output = r.json()
    logger.debug(f"Received model output: '{output}'")
    return {
            "text": output["text"],
            "is_positive_review": output["is_positive_review"],
            "details": {
                "mlflow_run_id": output["details"]["mlflow_run_id"],
            },
        }