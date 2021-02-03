import argparse
import logging
import pickle
import tempfile
from pathlib import Path

import mlflow.pytorch
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from torchtext.datasets import text_classification

from src.model import TextSentiment
from src.predict import log_model
from src.utils import setup_logging

logger = logging.getLogger(__file__)

VOCAB_DUMP_PATH = Path(f"{tempfile.mkdtemp()}/vocab.pkl")


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_grams", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=32),
    parser.add_argument("--max_epochs", type=int, default=10)
    return parser


NGRAMS = 1
EMBED_DIM = 32
MAX_EPOCHS = 1


def main(n_grams, batch_size, embed_dim, max_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset = text_classification.DATASETS["YelpReviewPolarity"](
        root="../data", ngrams=n_grams, vocab=None
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=generate_batch,
        num_workers=5,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=generate_batch,
        num_workers=5,
    )

    vocab = train_dataset.get_vocab()
    VOCAB_SIZE = len(vocab)
    NUM_CLASS = len(train_dataset.get_labels())

    logger.info("Creating average embedding model")
    model = TextSentiment(VOCAB_SIZE, embed_dim, NUM_CLASS)
    model.to(device)

    logger.info(f"Saving vocabulary to {VOCAB_DUMP_PATH}")
    with open(VOCAB_DUMP_PATH, "wb") as f:
        pickle.dump(vocab, f)

    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", verbose=True, patience=3
    )
    lr_logger = LearningRateMonitor()

    # Loss and optimizer
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=max_epochs,
        progress_bar_refresh_rate=20,
        callbacks=[lr_logger, early_stopping],
    )

    # Auto log all MLflow entities
    mlflow.pytorch.autolog(log_models=False)

    with mlflow.start_run() as run:
        logger.info(f"run_id: {run.info.run_id}")
        mlflow.log_artifact(VOCAB_DUMP_PATH, artifact_path="model/data")
        trainer.fit(model, train_dataloader, test_dataloader)
        log_model(model)


if __name__ == "__main__":
    logger = setup_logging()
    args = get_parser().parse_args()
    main(args.n_grams, args.batch_size, args.embed_dim, args.max_epochs)
