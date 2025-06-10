from nltk.corpus import wordnet as en_wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import List, Optional


def extract_genus_candidates(defn_en: str) -> List[str]:
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(defn_en.lower())
    tokens_lem = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    return tokens_lem


def guess_synset(
    term_en: str,
    defn_en: str,
) -> Optional[en_wordnet.synset]:
    candidate_genus = extract_genus_candidates(defn_en)
    term_synsets = en_wordnet.synsets(term_en, pos='n')
    if not term_synsets:
        return None

    if not candidate_genus:
        return None

    for genus_word in candidate_genus:
        genus_synsets = en_wordnet.synsets(genus_word, pos='n')
        if not genus_synsets:
            continue
        genus_synset = genus_synsets[0]

        filtered_synsets: List[en_wordnet.synset] = []
        for syn in term_synsets:
            for path in syn.hypernym_paths():
                if genus_synset in path:
                    filtered_synsets.append(syn)
                    break

        if filtered_synsets:
            return filtered_synsets[0]

    return None
