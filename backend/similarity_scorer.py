import numpy as np
from gensim.models import KeyedVectors
import gensim.downloader as api
import logging as log
from pathlib import Path

model_name = "glove-twitter-25"
model_path = Path(__file__).resolve().parent / model_name


def download_model():
    if not model_path.exists():
        log.info(f"{model_name} not found. Downloading...")
        model = api.load(model_name)
        print(f"{model_name} downloaded successfully.")
        model.save(str(model_path))
        log.info(f"{model_name} saved to {model_path}.")
    else:
        print("Model already exists. Skipping download.")



class SimilarityScorer:
    def __init__(self):
        log.info('Load word2vec model')
        download_model()
        self.model = KeyedVectors.load(str(model_path))
    
    def embed(self, tokens):
        if isinstance(tokens, str):
            tokens = [tokens]
        vecs = [self.model[t] for t in tokens if t in self.model]
        return np.mean(vecs, axis=0) if vecs else np.zeros(self.model.vector_size)

    def similarity(self, word, guess):
        v_word, v_guess = self.embed(word), self.embed(guess)
        denom = np.linalg.norm(v_word) * np.linalg.norm(v_guess)
        return float(v_word @ v_guess / denom) if denom else 0.0
