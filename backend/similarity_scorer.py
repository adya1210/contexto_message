import logging as log
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import STOPWORDS

# Source model: GloVe trained on Wikipedia + Gigaword news (400k words, 100 dims).
# We only keep the most common words, which keeps memory small and makes
# rankings feel like real English instead of being full of junk tokens.
SOURCE_MODEL = "glove-wiki-gigaword-100"
VOCAB_SIZE = 50_000
MIN_WORD_LENGTH = 2
model_path = Path(__file__).resolve().parent / f"glove-wiki-100-{VOCAB_SIZE // 1000}k.kv"

# Words that are too common to be meaningful guesses (also excluded from rankings).
RANK_STOP_WORDS = set(STOPWORDS)


def _is_rankable(word):
    return (
        len(word) >= MIN_WORD_LENGTH
        and word.isascii()
        and word.isalpha()
        and word not in RANK_STOP_WORDS
    )


def download_model():
    """Download the source model once and save a compact, common-words-only copy."""
    if model_path.exists():
        log.info("%s already exists. Skipping download.", model_path.name)
        return

    import gensim.downloader as api

    log.info("%s not found. Downloading %s...", model_path.name, SOURCE_MODEL)
    full = api.load(SOURCE_MODEL)
    # index_to_key is sorted by frequency, so the first words are the most common.
    words = []
    for word in full.index_to_key:
        if _is_rankable(word):
            words.append(word)
            if len(words) == VOCAB_SIZE:
                break

    compact = KeyedVectors(vector_size=full.vector_size)
    compact.add_vectors(words, np.stack([full[word] for word in words]))
    compact.save(str(model_path))
    log.info("Saved %d words to %s.", len(words), model_path)


class SimilarityScorer:
    def __init__(self, cache_size=4):
        log.info("Loading word vectors")
        download_model()
        self.model = KeyedVectors.load(str(model_path))
        vectors = self.model.vectors.astype(np.float32)
        self.unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.total = len(self.model.index_to_key)
        self._rank_cache = OrderedDict()
        self._cache_size = cache_size

    def knows(self, word):
        return word in self.model.key_to_index

    def is_too_common(self, word):
        return word in RANK_STOP_WORDS

    def _ranks_for(self, secret):
        """Return an array where ranks[i] is the 1-based rank of word i (secret = #1)."""
        cached = self._rank_cache.get(secret)
        if cached is not None:
            self._rank_cache.move_to_end(secret)
            return cached

        secret_vector = self.unit_vectors[self.model.key_to_index[secret]]
        similarities = self.unit_vectors @ secret_vector
        order = np.argsort(-similarities)
        ranks = np.empty(self.total, dtype=np.int32)
        ranks[order] = np.arange(1, self.total + 1, dtype=np.int32)
        # Make sure the secret is always #1, even with floating point ties.
        secret_index = self.model.key_to_index[secret]
        if ranks[secret_index] != 1:
            ranks[ranks < ranks[secret_index]] += 1
            ranks[secret_index] = 1

        self._rank_cache[secret] = ranks
        if len(self._rank_cache) > self._cache_size:
            self._rank_cache.popitem(last=False)
        return ranks

    def rank(self, secret, guess):
        """1-based rank of `guess` among all known words, ordered by closeness to `secret`.

        Returns None if either word is unknown.
        """
        if not self.knows(secret) or not self.knows(guess):
            return None
        return int(self._ranks_for(secret)[self.model.key_to_index[guess]])

    def score_from_rank(self, rank):
        """Map a rank to 0..1 for the progress bar (log scale, #1 = 1.0)."""
        if rank is None:
            return 0.0
        return max(0.0, 1.0 - math.log(rank) / math.log(self.total))
