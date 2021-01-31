"""
Train LSTM.
"""
import torch
from torchtext import data
import torch.nn as nn
import pandas as pd

import dill 

import logging
import time

import mlflow.pytorch
from mlflow.tracking import MlflowClient
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
import argparse
import logging

import os

import os
import sys
from pathlib import Path
from src.utils import DataFrameDataset, setup_logging
from src.lstm import LSTM_net


logger = logging.getLogger(__file__)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=Path, default=Path('/data/dataset_split.pkl'))
    parser.add_argument('--cuda_device', type=str, default="cuda:2")
    return parser


MAX_VOCAB_SIZE = 20000
BATCH_SIZE = 128

# Declare hyperparameters
# TODO: use and log these params
# num_epochs = 25
# learning_rate = 0.001

EMBEDDING_DIM = 200
HIDDEN_DIM = 128
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.2
SEED = 42

# export MLFLOW_TRACKING_URI=http://34.91.123.207:5000/


def main(args):
    logger.info(f"Args: {args}")
    dataset_path = args.dataset_path

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create text and label fields
    TEXT = data.Field(tokenize = 'spacy', include_lengths = True, batch_first=True)
    LABEL = data.LabelField(dtype = torch.float, batch_first=True)

   # TEXT = data.Field(tokenize = 'spacy', include_lengths = True)
   # LABEL = data.LabelField(dtype = torch.float)

    # Load dataset
    logging.info("Reading dataset and splitting into train/val")

    dataset = pd.read_pickle(dataset_path)
    train_df = dataset['X_train'].loc[:, ['text', 'stars']][0:100000]
    train_df['text'] = train_df['text'].str.lower()
    train_df['stars'] = (train_df['stars'] >= 3.0).astype(int)
    train_df.columns = ['text', 'target']

    val_df = dataset['X_val'].loc[:, ['text', 'stars']][0:50000]
    val_df['text'] = val_df['text'].str.lower()
    val_df['stars'] = (val_df['stars'] >= 3.0).astype(int)
    val_df.columns = ['text', 'target']

    logging.info("Finished reading dataset and splitting into train/val")

    # Tokenize data and create a data loader for the batch iterators
    fields = [('text', TEXT), ('label', LABEL)]

    logging.info("Tokenizing train and validation data")

    train_ds, val_ds = DataFrameDataset.splits(fields, train_df=train_df, val_df=val_df)

    logging.info("Finished tokenizing train and validation data")

    logging.info("Building vocabulary")

    TEXT.build_vocab(train_ds,
                 max_size = MAX_VOCAB_SIZE,
                 vectors = 'glove.6B.200d',
                 unk_init = torch.Tensor.zero_)

    LABEL.build_vocab(train_ds)

    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] # padding
    INPUT_DIM = len(TEXT.vocab)

    logging.info("Finished building vocabulary")

    device = torch.device(args.cuda_device if torch.cuda.is_available() else 'cpu')
    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_ds, val_ds),
        batch_size = BATCH_SIZE,
        sort_within_batch = True,
        device = device)
    
    # Create model
    model = LSTM_net(INPUT_DIM,
            EMBEDDING_DIM, 
            HIDDEN_DIM, 
            OUTPUT_DIM,
            N_LAYERS, 
            BIDIRECTIONAL, 
            DROPOUT,
            PAD_IDX)

    # Add pretrained embeddings
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    logging.info("Transferring model to device")

    # Save tokenizer
    with open("TEXT.Field", "wb") as f:
        dill.dump(TEXT, f)

    #  NN to GPU
    model.to(device)

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=3)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(), save_top_k=1, verbose=True, monitor="val_loss", mode="min", prefix="",
    )
    lr_logger = LearningRateMonitor()

    # Loss and optimizer
    trainer = pl.Trainer(gpus=1, max_epochs=20, progress_bar_refresh_rate=20,
                         callbacks=[lr_logger, early_stopping, checkpoint_callback],
                         checkpoint_callback=True)

    # Auto log all MLflow entities
    mlflow.pytorch.autolog()

    # Train the model
    logging.info("Training pytorch model")
    with mlflow.start_run() as run:
        mlflow.log_artifact("TEXT.Field")
        trainer.fit(model, train_iterator, valid_iterator)


if __name__ == "__main__":
    logger = setup_logging()
    main(get_parser().parse_args())