import logging as log
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
from lemminflect import getAllLemmas

# Prebuilt by backend/build_vectors.py (GloVe-300 + fastText blend, base word forms only).
MODEL_FILE = Path(__file__).resolve().parent / "word_vectors.npz"
VOCAB_SIZE = 50_000
MIN_WORD_LENGTH = 2

# Only true function words are left out. Contexto still ranks words like
# "computer", "fire" or "across", so we do too.
STOP_WORDS = set("""
a an the and or but nor of to in on at by for with from as into onto upon
is are was were be been being am i you he she it we they me him her us them
my your his its our their yours hers ours theirs this that these those
do does did has have had not no if so than then there here who whom which
what when where why how will would should can could
s t ll d re ve m o y don doesn didn isn aren wasn weren hasn hadn won wouldn shouldn couldn
""".split())

POS_ORDER = ("NOUN", "VERB", "ADJ", "ADV")


def is_base_form(word):
    """True unless the word is an inflection of another word (e.g. "claws", "went")."""
    lemmas = getAllLemmas(word)
    return not lemmas or any(word in forms for forms in lemmas.values())


def is_rankable(word):
    return (
        len(word) >= MIN_WORD_LENGTH
        and word.isascii()
        and word.isalpha()
        and word not in STOP_WORDS
        and is_base_form(word)
    )


def parts_of_speech(word):
    """Dictionary parts of speech for a word; empty for names and unknown words."""
    return set(getAllLemmas(word).keys())


class SimilarityScorer:
    def __init__(self, cache_size=4):
        log.info("Loading word vectors from %s", MODEL_FILE.name)
        data = np.load(MODEL_FILE)
        self.words = [str(word) for word in data["words"]]
        self.key_to_index = {word: index for index, word in enumerate(self.words)}
        vectors = data["vectors"].astype(np.float32)
        self.unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.total = len(self.words)
        self._rank_cache = OrderedDict()
        self._cache_size = cache_size

    def knows(self, word):
        return word in self.key_to_index

    def is_too_common(self, word):
        return word in STOP_WORDS

    def frequency_rank(self, word):
        """0 = most common word in the vocabulary."""
        return self.key_to_index.get(word)

    def normalize(self, word):
        """Map a guess to the form we rank ("claws" -> "claw", "went" -> "go")."""
        word = word.strip().lower()
        if word in self.key_to_index:
            return word
        lemmas = getAllLemmas(word)
        for pos in POS_ORDER:
            for lemma in lemmas.get(pos, ()):
                if lemma.lower() in self.key_to_index:
                    return lemma.lower()
        return word

    def _ranks_for(self, secret):
        """(ranks, order): ranks[i] is the 1-based rank of word i; order[r-1] is the word at rank r."""
        cached = self._rank_cache.get(secret)
        if cached is not None:
            self._rank_cache.move_to_end(secret)
            return cached

        secret_index = self.key_to_index[secret]
        similarities = self.unit_vectors @ self.unit_vectors[secret_index]
        similarities[secret_index] = np.inf  # the secret is always #1
        order = np.argsort(-similarities, kind="stable")
        ranks = np.empty(self.total, dtype=np.int32)
        ranks[order] = np.arange(1, self.total + 1, dtype=np.int32)

        self._rank_cache[secret] = (ranks, order)
        if len(self._rank_cache) > self._cache_size:
            self._rank_cache.popitem(last=False)
        return ranks, order

    def rank(self, secret, guess):
        """1-based rank of `guess` by closeness to `secret`, or None if either is unknown."""
        if not self.knows(secret) or not self.knows(guess):
            return None
        return int(self._ranks_for(secret)[0][self.key_to_index[guess]])

    def word_at_rank(self, secret, rank):
        if not self.knows(secret) or not 1 <= rank <= self.total:
            return None
        return self.words[self._ranks_for(secret)[1][rank - 1]]

    def score_from_rank(self, rank):
        """Map a rank to 0..1 for the progress bar (log scale, #1 = 1.0)."""
        if rank is None:
            return 0.0
        return max(0.0, 1.0 - math.log(rank) / math.log(self.total))
