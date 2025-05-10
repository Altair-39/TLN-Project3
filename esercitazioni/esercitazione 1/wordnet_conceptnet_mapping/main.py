

import json
from nltk.corpus import wordnet as wn
from src.wordnet_utils import wordnet_to_conceptnet


def main():
    all_words = set()

    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            all_words.add(lemma.name())

    results = {}
    for word in sorted(all_words):
        try:
            results[word] = wordnet_to_conceptnet(word)
        except Exception as e:
            print(f"Failed to process word '{word}': {e}")

    # Save results to JSON
    with open("wordnet_to_conceptnet.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
