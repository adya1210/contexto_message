"""Build the compact word-vector file the game ships with (backend/word_vectors.npz).

Run once locally (needs ~3 GB of downloads the first time):

    python -m backend.build_vectors

Why this setup? We scored candidate models against Contexto's real closest-word
lists for 30 past games. A blend of GloVe-300 (Wikipedia + news) and fastText
(Wikipedia news, subword) matched Contexto best. The vocabulary keeps only base
word forms ("claw", not "claws"), like Contexto does.
"""
import logging as log

import numpy as np

if __package__:
    from .similarity_scorer import MODEL_FILE, VOCAB_SIZE, is_rankable
else:
    from similarity_scorer import MODEL_FILE, VOCAB_SIZE, is_rankable

PRIMARY = "glove-wiki-gigaword-300"
SECONDARY = "fasttext-wiki-news-subwords-300"
# Cosine of the concatenated vectors = 2/3 GloVe similarity + 1/3 fastText similarity.
PRIMARY_WEIGHT = np.sqrt(2.0)
SECONDARY_WEIGHT = 1.0


def unit(matrix):
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def build():
    import gensim.downloader as api
    from gensim.models import KeyedVectors

    log.info("Loading %s", PRIMARY)
    primary = api.load(PRIMARY)
    log.info("Loading %s", SECONDARY)
    secondary = KeyedVectors.load_word2vec_format(
        api.load(SECONDARY, return_path=True), binary=False, limit=400_000
    )

    words = []
    for word in primary.index_to_key:  # sorted most-common first
        if is_rankable(word) and word in secondary.key_to_index:
            words.append(word)
            if len(words) == VOCAB_SIZE:
                break

    a = unit(np.stack([primary[w] for w in words]).astype(np.float32)) * PRIMARY_WEIGHT
    b = unit(np.stack([secondary[w] for w in words]).astype(np.float32)) * SECONDARY_WEIGHT
    vectors = unit(np.hstack([a, b]))
    # int8 keeps the file small; ranking quality is unchanged in testing.
    quantized = np.clip(np.round(vectors * 127), -127, 127).astype(np.int8)
    np.savez_compressed(MODEL_FILE, words=np.array(words), vectors=quantized)
    log.info("Saved %d words to %s", len(words), MODEL_FILE)


if __name__ == "__main__":
    log.basicConfig(level=log.INFO)
    build()
