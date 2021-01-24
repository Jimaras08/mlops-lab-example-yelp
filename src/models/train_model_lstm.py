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

import os

import os
import sys

import sys
sys.path.append('..')
os.chdir('..') # FIXME: Relative paths

from src.data.utils import DataFrameDataset, setup_logging

DATASET_PATH = '../notebooks/dataset_split.pkl'
MAX_VOCAB_SIZE = 20000
BATCH_SIZE = 128
CUDA_DEVICE = 'cuda:2'

# Declare hyperparameters
num_epochs = 25
learning_rate = 0.001

EMBEDDING_DIM = 200
HIDDEN_DIM = 128
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.2

TRACKING_URI = 'http://34.91.123.207:5000/'

mlflow.set_tracking_uri(TRACKING_URI)


if __name__ == "__main__":
    SEED = 42

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger = setup_logging()

    print(os.getcwd())

    # Create text and label fields
    TEXT = data.Field(tokenize = 'spacy', include_lengths = True)
    LABEL = data.LabelField(dtype = torch.float)

    # Load dataset
    logging.info("Reading dataset and splitting into train/val")

    dataset = pd.read_pickle(DATASET_PATH)
    train_df = dataset['X_train'].loc[:, ['text', 'stars']][0:100000]
    train_df['stars'] = (train_df['stars'] >= 3.0).astype(int)
    train_df.columns = ['text', 'target']

    val_df = dataset['X_val'].loc[:, ['text', 'stars']][0:50000]
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

    device = torch.device(f'{CUDA_DEVICE}' if torch.cuda.is_available() else 'cpu')
    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_ds, val_ds), 
        batch_size = BATCH_SIZE,
        sort_within_batch = True,
        device = device)
    
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

    # Loss and optimizer
    trainer = pl.Trainer(gpus=1, max_epochs=20, progress_bar_refresh_rate=20)

    # Auto log all MLflow entities
    mlflow.pytorch.autolog()

    # Train the model
    logging.info("Training pytorch model")
    with mlflow.start_run() as run:
        trainer.fit(model, train_iterator, valid_iterator)
