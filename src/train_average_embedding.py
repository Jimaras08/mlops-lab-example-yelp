import torch
import torchtext
from torchtext.datasets import text_classification
from torch.utils.data import DataLoader
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import pytorch_lightning as pl

from average_embedding import TextSentiment

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


def main(args):

    train_dataset, test_dataset = text_classification.DATASETS['YelpReviewPolarity'](
        root='../data', ngrams=NGRAMS, vocab=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=generate_batch)

    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=generate_batch)

    VOCAB_SIZE = len(train_dataset.get_vocab())
    NUN_CLASS = len(train_dataset.get_labels())
    model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS)

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=3)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(), save_top_k=1, verbose=True, monitor="val_loss", mode="min", prefix="",
    )
    lr_logger = LearningRateMonitor()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    # Loss and optimizer
    trainer = pl.Trainer(gpus=1, max_epochs=1, progress_bar_refresh_rate=20)
                    #   callbacks=[lr_logger, early_stopping],
                    #  checkpoint_callback=True)

    # Auto log all MLflow entities
    mlflow.pytorch.autolog(log_models = True)

    with mlflow.start_run() as run:
        trainer.fit(model, train_dataloader, test_dataloader)