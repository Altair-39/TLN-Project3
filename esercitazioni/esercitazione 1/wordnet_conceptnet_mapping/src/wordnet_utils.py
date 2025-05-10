
import nltk
from nltk.corpus import wordnet as wn
from src.conceptnet_utils import get_conceptnet_entries

nltk.download('wordnet')


def wordnet_to_conceptnet(word):

    synsets = wn.synsets(word)
    conceptnet_data = get_conceptnet_entries(word)

    relations = []
    for edge in conceptnet_data:
        try:
            if edge["start"]["language"] == "en" and edge["end"]["language"] == "en":
                rel = edge["rel"]["@id"]
                target = edge["end"]["label"] if edge["end"]["label"].lower(
                ) != word.lower() else edge["start"]["label"]
                relations.append({
                    "relation": rel,
                    "target": target
                })
        except KeyError:
            continue

    return {
        "synsets": [
            {"name": syn.name(), "definition": syn.definition()}
            for syn in synsets
        ],
        "relations": relations
    }
