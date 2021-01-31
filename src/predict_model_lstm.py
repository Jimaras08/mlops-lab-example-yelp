"""
Script for prediction
"""

import spacy
import dill

import torch
nlp = spacy.load('en')
MODEL_PATH = '....'
DEVICE = 'cuda:2'
TOKENIZER_PATH = 'TEXT.Field'

with open(TOKENIZER_PATH, "rb") as f:
     TEXT = dill.load(f)


def predict(model, sentence):

    # Tokenize the sentence 
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]

    # Convert to integer sequence
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]

    # Compute no. of words
    length = [len(indexed)]

    # Convert to tensor
    tensor = torch.LongTensor(indexed).to(device)

    # Reshape in form of batch, no. of words
    tensor = tensor.unsqueeze(1).T

    # Convert to tensor
    length_tensor = torch.LongTensor(length)

    # Prediction
    prediction = model(tensor, length_tensor)

    return prediction.item()


if __name__ == "main":

    model.eval()
    device = torch.device(f'{DEVICE}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    predict(model, "terrible horrible restaurant")
