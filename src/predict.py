import os
from time import time
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

MODEL_FILE_NAME = "model.pth"
VOCAB_FILE_NAME = "vocab.pkl"


def log_model(model):
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=MODEL_ARTIFACT_PATH,
        registered_model_name=MODEL_NAME,
        requirements_file=str(REQUIREMENTS_PATH),
    )


class ModelWrapper:
    def __init__(self, model, vocab):
        logger.info(f"ModelWrapper.__init__: model of type {type(model)}: {model}")
        logger.info(f"ModelWrapper.__init__: vocab of type {type(vocab)}: {vocab}")
        self.model = model
        self.vocab = vocab

    def predict(self, text: str):
        time_started = time()
        tokenizer = get_tokenizer("basic_english")
        with torch.no_grad():
            text_tensor = torch.tensor(
                [
                    self.vocab[token]
                    for token in ngrams_iterator(tokenizer(text), NGRAMS)
                ]
            )
            output_tensor = self.model(text_tensor, torch.tensor([0]))
            output = output_tensor.argmax(1).item()
            elapsed = time() - time_started
            logger.info(
                f"ModelWrapper.predict: [elapsed {elapsed:.2f}s]: "
                f"len(text)={len(text)}, len(tokens)={len(text_tensor)}, answer={output}"
            )
            return output
