from typing import List, Optional, TypedDict
from nltk.corpus.reader import Synset
from nltk.corpus import wordnet as wn
from src.conceptnet_utils import get_conceptnet_entries
import logging

# Define type aliases for better code documentation


class SynsetData(TypedDict):
    name: str
    definition: str


class ConceptNetRelation(TypedDict):
    relation: str
    target: str


class WordNetToConceptNetResult(TypedDict):
    synsets: List[SynsetData]
    relations: List[ConceptNetRelation]


def wordnet_to_conceptnet(word: str) -> Optional[WordNetToConceptNetResult]:
    try:
        synsets: List[Synset] = wn.synsets(word)
        if not synsets:
            logging.warning(f"No WordNet synsets found for '{word}'")
            return None

        conceptnet_data = get_conceptnet_entries(word)
        if not conceptnet_data:
            logging.warning(f"No ConceptNet edges found for '{word}'")

        relations: List[ConceptNetRelation] = []
        for edge in conceptnet_data:
            try:
                if (edge["start"]["language"] == "en" and
                        edge["end"]["language"] == "en"):

                    rel = edge["rel"]["@id"].split("/")[-1]
                    start_label = edge["start"]["label"].lower()
                    end_label = edge["end"]["label"].lower()
                    word_lower = word.lower()

                    if end_label != word_lower:
                        target = edge["end"]["label"]
                    elif start_label != word_lower:
                        target = edge["start"]["label"]
                    else:
                        continue

                    relations.append({
                        "relation": rel,
                        "target": target
                    })
            except (KeyError, AttributeError):
                continue

        return {
            "synsets": [
                {
                    "name": syn.name(),
                    "definition": syn.definition(),
                }
                for syn in synsets
            ],
            "relations": relations
        }

    except Exception as e:
        logging.error(f"Failed to process word '{word}': {str(e)}")
        return None
