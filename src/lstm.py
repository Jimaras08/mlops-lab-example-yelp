"""
Bidirectional LSTM.
Adapted from https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/
"""

import torch.nn as nn
import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy

class LSTM_net(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)
        
        self.fc1 = nn.Linear(hidden_dim * 2, output_dim)
        
        #activation function
        self.act = nn.Sigmoid()

       # self.fc2 = nn.Linear(hidden_dim, 1)
        
       #  self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):

        # text Dim: (sent len, batch size)
        # embedded dim: (sent len, batch size, emb dim)
        embedded = self.embedding(text) 
                
        # Pack padded sequences sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        
        # hidden = (num layers * num directions, batch size, hid dim)
        # cell = (num layers * num directions, batch size, hid dim)
        # packed_output = [sent len, batch size, hid dim * num directions]

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        # hidden = (batch size, hid dim * num directions)

        #hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)

        # activation function

        dense_outputs=self.fc1(hidden)
        output=self.act(dense_outputs)
        #output = self.fc1(hidden)
        #output = self.dropout(self.fc2(output))

        return output

    def training_step(self, batch, batch_nb):
        text, text_lengths = batch.text

        criterion = nn.BCEWithLogitsLoss()

        predictions = self(text, text_lengths).squeeze(1)

        loss = criterion(predictions, batch.label)
        acc = accuracy(predictions, batch.label)

        # Use the current PyTorch logger
        # self.log("train_loss", loss, on_epoch=True)
        # self.log("acc", acc, on_epoch = True)

        return loss
    
    def test_step(self, batch, batch_nb):

        with torch.no_grad():

            text, text_lengths = batch.text

            criterion = nn.BCEWithLogitsLoss()

            predictions = self(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)
            acc = accuracy(predictions, batch.label)
            return loss

    def validation_step(self, batch, batch_nb):

        with torch.no_grad():

            text, text_lengths = batch.text

            criterion = nn.BCEWithLogitsLoss()

            predictions = self(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)
            acc = accuracy(predictions, batch.label)
            return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
