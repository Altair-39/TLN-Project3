import json
import random
import logging
from nltk.corpus import wordnet as wn
from src.wordnet_utils import wordnet_to_conceptnet


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def main() -> None:
    setup_logging()

    try:
        all_words = set()
        logging.info("Collecting words from WordNet...")
        for synset in wn.all_synsets():
            for lemma in synset.lemmas():
                all_words.add(lemma.name())
        word_list = list(all_words)
        random.shuffle(word_list)
        selected_words = word_list[:min(100, len(word_list))]
        results = {}
        logging.info(f"Processing {len(selected_words)} random words...")
        for word in sorted(selected_words):
            try:
                results[word] = wordnet_to_conceptnet(word)
                logging.info(f"Processed: {word}")
            except Exception as e:
                logging.warning(f"Failed to process word '{word}': {e}")

        with open("wordnet_to_conceptnet_sample.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info(f"Successfully processed {len(results)} words")

    except KeyboardInterrupt:
        logging.info("\nInterrupt received, saving partial results...")
        if results:
            with open("wordnet_to_conceptnet_partial.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved partial results for {len(results)} words")
        logging.info("Exiting!")


if __name__ == "__main__":
    main()
