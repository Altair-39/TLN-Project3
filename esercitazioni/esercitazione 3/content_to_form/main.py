import os
import logging
from nltk.corpus import wordnet as en_wordnet
from deep_translator import GoogleTranslator
from typing import Optional
from dotenv import load_dotenv, find_dotenv
from src.load_data import extract_definitions_to_word
from src.guessing import guess_synset


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def check_dotenv(dotenv_path: Optional[str]) -> None:
    if dotenv_path:
        load_dotenv(dotenv_path)
        logging.info(f"Loaded environment variables from: {dotenv_path}")
    else:
        logging.error("No .env file found.")


def translate_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source='it', target='en').translate(text)
    except Exception as e:
        logging.warning(f"Translation failed for text: {text}. Error: {e}")
        return text


def main() -> None:
    setup_logging()
    dotenv_path = find_dotenv()
    check_dotenv(dotenv_path)

    definitions_csv = os.getenv("DEFINITIONS_CSV", "rsrc/definizioni.csv")
    definitions_dict = extract_definitions_to_word(definitions_csv)

    try:
        for term, definitions in definitions_dict.items():
            term_en = translate_to_english(term)
            term_synsets = en_wordnet.synsets(term_en, pos='n')

            logging.info(f"\nTerm: {term} ({term_en}) →"
                         "Possible Synsets: {term_synsets}")

            total = 0
            correct = 0

            for defn in definitions:
                defn_translated = translate_to_english(defn)
                predicted_synset = guess_synset(term_en, defn_translated)

                logging.info(f"\nDefinition (IT): {defn}")
                logging.info(f"Definition (EN): {defn_translated}")
                logging.info(f"Predicted Synset: {predicted_synset}")

                total += 1

                if predicted_synset and term_synsets:
                    if predicted_synset in term_synsets:
                        correct += 1
                        logging.info("Synset Match!")
                    else:
                        logging.info("Synset Mismatch.")
                else:
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
