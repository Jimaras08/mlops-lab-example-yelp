import importlib
import logging
import pickle
import shutil
import sys
from pathlib import Path
from typing import Union
from time import time

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

# TODO: should be stored in mlflow
NGRAMS = 1
MODEL_ARTIFACT_PATH = "model"

MODEL_FILE_NAME = "model.pth"
VOCAB_FILE_NAME = "vocab.pkl"
PREDICT_MODULE = "src.predict_average_embedding"
MODEL_WRAPPER_CLASS_NAME = "ModelWrapper"


def _rename_dir(directory: Path, new_name: str, skip_if_exists=True):
    if not skip_if_exists:
        raise NotImplementedError("skip_if_exists=False")

    new_directory = directory.parent / new_name
    if not new_directory.is_dir():
        logger.info(f"Copying dir {directory} -> {new_directory}")
        shutil.copytree(str(directory), str(new_directory))
    else:
        logger.info(f"Dir already exists, skipping copying: {new_directory}")
    return new_directory


def load_model(model_artifacts_path: Union[str, Path]):
    time_started = time()

    model_artifacts_path = Path(model_artifacts_path)
    ls = list(model_artifacts_path.iterdir())
    logger.info(f"Using model artifacts directory: {model_artifacts_path}: {ls}")

    data_dir = model_artifacts_path / "data"
    code_dir = model_artifacts_path / "code"
    model_file = data_dir / MODEL_FILE_NAME
    vocab_file = data_dir / VOCAB_FILE_NAME

    if not data_dir.is_dir():
        raise ValueError(f"Model data directory not found: {data_dir}")
    logger.info(f"Model data directory: {data_dir}: {list(data_dir.iterdir())}")

    if not code_dir.is_dir():
        raise ValueError(f"Model code directory not found: {code_dir}")
    logger.info(f"Model code directory: {code_dir}: {list(code_dir.iterdir())}")

    if not model_file.is_file():
        raise ValueError(f"Model file not found: {model_file}")

    if not vocab_file.is_file():
        raise ValueError(f"Vocab file not found: {vocab_file}")

    code_dir = _rename_dir(code_dir, new_name=PREDICT_MODULE.split(".")[0])
    logger.info(f"Adding code dir root to sys.path: {code_dir.parent}")
    sys.path.insert(0, str(code_dir.parent))

    logger.info(f"Importing class {PREDICT_MODULE}.{MODEL_WRAPPER_CLASS_NAME}")
    predict_module = importlib.import_module(PREDICT_MODULE)
    ModelWrapper = getattr(predict_module, MODEL_WRAPPER_CLASS_NAME)

    logger.info(f"Loading vocabulary file {vocab_file}")
    vocab = pickle.load(vocab_file.open("rb"))

    logger.info(f"Loading model file {model_file}")
    model = torch.load(model_file, map_location="cpu")
    model.eval()

    result = ModelWrapper(model, vocab)

    elapsed = time() - time_started
    logger.info(f"[Elapsed {elapsed:.2f}s]: Model loaded from {model_artifacts_path}")
    return result
