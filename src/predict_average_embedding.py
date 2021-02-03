import os
from pathlib import Path

import torch
import logging
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
import mlflow.pytorch

logger = logging.getLogger(__file__)

# TODO: expose this as env var option
NGRAMS = 1

CURRENT_FILE = Path(__file__)
SRC_DIR = CURRENT_FILE.parent
REQUIREMENTS_PATH = SRC_DIR.parent / "requirements.txt"

MODEL_NAME = os.environ.get("MODEL_NAME", "yelp-model")
MODEL_ARTIFACT_PATH = "model"

PREDICT_FILE_NAME = __file__
MODEL_FILE_NAME = "model.pth"
VOCAB_FILE_NAME = "vocab.pkl"


def log_model(model):
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=MODEL_ARTIFACT_PATH,
        registered_model_name=MODEL_NAME,
        code_paths=[str(SRC_DIR / PREDICT_FILE_NAME)],
        requirements_file=str(REQUIREMENTS_PATH),
    )


class ModelWrapper:
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab

    def predict(self, text: str):
        tokenizer = get_tokenizer("basic_english")
        with torch.no_grad():
            text = torch.tensor(
                [
                    self.vocab[token]
                    for token in ngrams_iterator(tokenizer(text), NGRAMS)
                ]
            )
            output = self.model(text, torch.tensor([0]))
            return output.argmax(1).item()
