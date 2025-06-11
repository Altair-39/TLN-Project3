import os
import logging
import argparse
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from nltk.corpus import wordnet as en_wordnet
from deep_translator import GoogleTranslator

from src.load_data import extract_definitions_to_word
from src.guessing import guess_synset


def setup_logging() -> None:
    level = logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def translate_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source='it', target='en').translate(text)
    except Exception as e:
        logging.warning(f"Translation failed for text: {text}. Error: {e}")
        return text


def load_environment() -> Optional[str]:
    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)
        logging.info(f"Loaded environment from: {dotenv_path}")
        return dotenv_path
    else:
        logging.error("No .env file found.")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synset guessing script")
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable detailed debug logging'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    load_environment()

    try:
        definitions_csv = os.getenv("DEFINITIONS_CSV", "rsrc/definizioni.csv")
        definitions_dict = extract_definitions_to_word(definitions_csv)

        for term, definitions in definitions_dict.items():
            term_en = translate_to_english(term)
            term_synsets = en_wordnet.synsets(term_en, pos='n')

            if args.debug:
                logging.info(f"\nTerm: {term} ({term_en})"
                             f"→ Possible Synsets: {term_synsets}")

            total = 0
            correct = 0

            for defn in definitions:
                defn_translated = translate_to_english(defn)
                predicted_synset = guess_synset(term_en, defn_translated)

                if args.debug:
                    logging.info(f"\nDefinition (IT): {defn}")
                    logging.info(f"Definition (EN): {defn_translated}")
                    logging.info(f"Predicted Synset: {predicted_synset}")

                total += 1

                if predicted_synset and term_synsets:
                    if predicted_synset in term_synsets:
                        correct += 1
                        if args.debug:
                            logging.info("Synset Match!")
                    else:
                        if args.debug:
                            logging.info("Synset Mismatch.")
                else:
                    if args.debug:
                        logging.warning("Synset could not be resolved.")

            if total > 0:
                accuracy = (correct / total) * 100
                logging.info(f"\nAccuracy for '{term}' ({term_en}):"
                             f"{accuracy:.2f}% ({correct}/{total})")
            else:
                logging.info(f"\nNo definitions to evaluate for '{term}'.")

    except KeyboardInterrupt:
        logging.info("Exiting!")


if __name__ == "__main__":
    main()
