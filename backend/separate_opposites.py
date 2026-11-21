"""Find opposite word pairs (warm/cool, win/lose, buy/sell) to keep out of the green zone.

Word-vector models learn from which words appear in similar sentences, and
opposites appear in almost identical ones ("the water is warm/cold"), so the
model ranks them as near-synonyms (87% of opposite pairs landed in each other's
green zone). The scorer moves a secret word's opposites to just after
OPPOSITE_RANK in its ranking (see SimilarityScorer), i.e. the yellow zone:
"related, but not it". Every other word keeps its normal rank.

We also tried moving the vectors themselves apart ("counter-fitting"). It did
not carry over to pairs missing from the list and shifted other rankings, so
the exact rank-time rule won.

Pairs come from WordNet: direct opposites (warm/cool, hot/cold) plus its "see
also" links, which catch indirect ones (warm/cold, happy/sad). Only adjectives,
adverbs and verbs are used; noun pairs such as king/queen or day/night stay close
because players expect them to be.

Update the pairs in the shipped vector file (no model downloads needed):

    python -m backend.separate_opposites
"""
import logging as log

import numpy as np

OPPOSITE_POS = {"a", "s", "r", "v"}  # adjective, satellite adjective, adverb, verb


def antonym_pairs(words):
    """Opposite pairs as (index, index) into `words`, from WordNet."""
    import nltk

    try:
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
    except LookupError:
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import wordnet as wn

    index = {word: i for i, word in enumerate(words)}
    pairs = set()

    def add(a, b):
        a, b = a.lower(), b.lower()
        if a != b and a in index and b in index:
            pairs.add((min(index[a], index[b]), max(index[a], index[b])))

    for synset in wn.all_synsets():
        if synset.pos() not in OPPOSITE_POS:
            continue
        for lemma in synset.lemmas():
            for antonym in lemma.antonyms():
                add(lemma.name(), antonym.name())
                # warm "see also" hot, and hot's opposite is cold -> warm/cold
                for related in synset.also_sees():
                    for related_lemma in related.lemmas():
                        for far in related_lemma.antonyms():
                            add(lemma.name(), far.name())
                # cool "see also" cold -> warm/cold
                for related in antonym.synset().also_sees():
                    for related_lemma in related.lemmas():
                        add(lemma.name(), related_lemma.name())
    return np.array(sorted(pairs), dtype=np.int32).reshape(-1, 2)


def update_model_file():
    if __package__:
        from .similarity_scorer import MODEL_FILE
    else:
        from similarity_scorer import MODEL_FILE

    data = dict(np.load(MODEL_FILE))
    words = [str(word) for word in data["words"]]
    data["opposites"] = antonym_pairs(words)
    np.savez_compressed(MODEL_FILE, **data)
    log.info("Saved %d opposite pairs to %s", len(data["opposites"]), MODEL_FILE)


if __name__ == "__main__":
    log.basicConfig(level=log.INFO)
    update_model_file()
