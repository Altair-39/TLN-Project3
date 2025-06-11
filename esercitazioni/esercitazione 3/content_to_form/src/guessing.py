from typing import List, Optional, Tuple

from nltk.corpus import wordnet as en_wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from functools import lru_cache

lemmatizer = WordNetLemmatizer()


@lru_cache(maxsize=None)
def cached_synsets(word: str, pos='n') -> Tuple[..., ...]:
    return tuple(en_wordnet.synsets(word, pos=pos))


def extract_genus_candidates(defn_en: str) -> List[str]:
    tokens = word_tokenize(defn_en.lower())
    tokens_lem = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    return tokens_lem


def guess_synset(term_en: str, defn_en: str) -> Optional[en_wordnet.synset]:
    candidate_genus = extract_genus_candidates(defn_en)
    term_synsets = cached_synsets(term_en, pos='n')
    if not term_synsets or not candidate_genus:
        return None

    for genus_word in candidate_genus:
        genus_synsets = cached_synsets(genus_word, pos='n')
        if not genus_synsets:
            continue
        genus_synset = genus_synsets[0]

        for syn in term_synsets:
            if any(genus_synset in path for path in syn.hypernym_paths()):
                return syn

    return None
