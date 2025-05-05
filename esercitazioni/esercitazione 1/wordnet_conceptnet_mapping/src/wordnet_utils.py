
import nltk
from nltk.corpus import wordnet as wn
from src.conceptnet_utils import get_conceptnet_entries

nltk.download('wordnet')


def wordnet_to_conceptnet(word):
    synsets = wn.synsets(word)
    conceptnet_data = get_conceptnet_entries(word)

    for syn in synsets:
        print(f"\nWordNet Synset: {syn.name()} - {syn.definition()}")
        print("ConceptNet edges:")
        for edge in conceptnet_data[:5]:  # Limit to first 5 for brevity
            print(f"  - {edge['rel']['label']} → {edge['end']['label']}")
