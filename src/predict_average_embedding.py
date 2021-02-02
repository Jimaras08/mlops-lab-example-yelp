import torch
import logging
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import text_classification
from src.average_embedding import TextSentiment
import pandas as pd
import numpy as np

logger = logging.getLogger(__file__)


class MLflowModel:
    # TODO
    def __init__(self, model_path, vocab_path):
        pass
    def predict(self, model_input: pd.DataFrame): # -> [np.ndarray | pd.DataFrame]
        pass


def _load_pyfunc(path):
    return MLflowModel(...)


# -- Old code below


def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item()

if __name__ == "__main__":

    ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
        enduring the season’s worst weather conditions on Sunday at The \
        Open on his way to a closing 75 at Royal Portrush, which \
        considering the wind and the rain was a respectable showing. \
        Thursday’s first round at the WGC-FedEx St. Jude Invitational \
        was another story. With temperatures in the mid-80s and hardly any \
        wind, the Spaniard was 13 strokes better in a flawless round. \
        Thanks to his best putting performance on the PGA Tour, Rahm \
        finished with an 8-under 62 for a three-stroke lead, which \
        was even more impressive considering he’d never played the \
        front nine at TPC Southwind."

    train_dataset, test_dataset = text_classification.DATASETS['YelpReviewPolarity'](
            root='../data', ngrams=NGRAMS, vocab=None)

    vocab = train_dataset.get_vocab()
    VOCAB_SIZE = len(train_dataset.get_vocab())
    NUN_CLASS = len(train_dataset.get_labels())
    model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS)
    model = model.to("cpu")

    predict(ex_text_str, model, vocab, 1)