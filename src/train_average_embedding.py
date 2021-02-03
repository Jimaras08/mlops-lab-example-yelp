from torchtext.datasets import text_classification
from torch.utils.data import DataLoader
import mlflow.pytorch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import pytorch_lightning as pl
import pickle
import logging

from src.average_embedding import TextSentiment
from src.predict_average_embedding import log_model
from src.utils import setup_logging
import tempfile
from pathlib import Path


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


NGRAMS = 1
BATCH_SIZE = 32
EMBED_DIM = 32
MAX_EPOCHS = 1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset = text_classification.DATASETS['YelpReviewPolarity'](
        root='../data', ngrams=NGRAMS, vocab=None)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=generate_batch, num_workers = 5)

    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=generate_batch, num_workers = 5)

    vocab = train_dataset.get_vocab()
    VOCAB_SIZE = len(vocab)
    NUN_CLASS = len(train_dataset.get_labels())
    model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS)
    model.to(device)

    logger.info(f"Saving vocabulary to {VOCAB_DUMP_PATH}")
    with open(VOCAB_DUMP_PATH, 'wb') as f:
        pickle.dump(vocab, f)

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=3)
    lr_logger = LearningRateMonitor()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    # Loss and optimizer
    trainer = pl.Trainer(gpus=1, max_epochs=MAX_EPOCHS, progress_bar_refresh_rate=20,
                         callbacks=[lr_logger, early_stopping])

    # Auto log all MLflow entities
    mlflow.pytorch.autolog(log_models=False)

    with mlflow.start_run() as run:
        logger.info(f"run_id: {run.info.run_id}")
        mlflow.log_param("max_epochs", MAX_EPOCHS)
        mlflow.log_artifact(VOCAB_DUMP_PATH, artifact_path="model/data")
        log_model(model)
        trainer.fit(model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    logger = setup_logging()
    main()