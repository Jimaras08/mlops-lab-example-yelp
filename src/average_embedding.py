import torch.nn as nn
import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy, auroc

class TextSentiment(pl.LightningModule):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
    
    def training_step(self, batch, batch_nb):
        text, offsets, target = batch

        criterion = nn.CrossEntropyLoss()

        predictions = self(text, offsets)

        loss = criterion(predictions, target)

        predictions = predictions.squeeze(1)

        acc = accuracy(predictions.argmax(1), target)

        self.log("train_acc", acc, on_epoch=True)
        self.log("train_loss", loss, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_nb):
        text, offsets, target = batch

        criterion = nn.CrossEntropyLoss()

        predictions = self(text, offsets)

        loss = criterion(predictions, target)

        predictions = predictions.squeeze(1)

        acc = accuracy(predictions.argmax(1), target)
        # NOTE: auroc fails with ValueError: No positive samples in targets, true positive value should be meaningless
        #auc = auroc(predictions, target)

        self.log("val_acc", acc, on_epoch=True)
        self.log("val_loss", loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        return [optimizer], [scheduler]