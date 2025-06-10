import os
import csv
import logging
from dotenv import load_dotenv, find_dotenv
from typing import Optional, Tuple, List
from src.babelnet import get_sense, find_synset_language_sets
from src.saving import save_ambiguities, save_pseudoword


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


def load_word_pairs(filepath: str) -> List[Tuple[str, str]]:
    pairs = []
    try:
        with open(filepath, newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    pairs.append((row[0].strip(), row[1].strip()))
    except FileNotFoundError:
        logging.error(f"Input file not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error reading input file: {e}")
        raise
    return pairs


def process_word_pair(en_word: str, it_word: str, api_key: str) -> Optional[dict]:
    synsets = get_sense(en_word, ["EN", "IT"], api_key)
    if not synsets:
        logging.warning(f"No synsets found for pair: {en_word} - {it_word}")
        return None
    s_en, s_it, common_synsets = find_synset_language_sets(synsets)
    if (s_en + s_it) == 0:
        ambiguity_reduction = 0.0
    else:
        ambiguity_reduction = ((s_en + s_it) - (len(common_synsets) << 1)
                               ) / (s_en + s_it)
    save_pseudoword(en_word, it_word, synsets, common_synsets)
    return {
        'pseudoword': f'{en_word}-{it_word}',
        'ambiguity_reduction': round(ambiguity_reduction, 3)
    }


def main() -> None:
    setup_logging()
    dotenv_path = find_dotenv()
    check_dotenv(dotenv_path)
    API_KEY = os.getenv('BABELNET_API_KEY')
    input_file = os.getenv('WORD_PAIRS')

    if not API_KEY or not input_file:
        logging.error("Required environment variables missing.")
        return

    try:
        word_pairs = load_word_pairs(input_file)
    except Exception:
        return

    ambiguity_scores = []

    try:
        for en_word, it_word in word_pairs:
            logging.info(f"Processing pair: {en_word} - {it_word}")
            result = process_word_pair(en_word, it_word, API_KEY)
            if result:
                logging.info(
                    f"  Ambiguity reduction: {result['ambiguity_reduction']:.3f}"
                )
                ambiguity_scores.append(result)

    except KeyboardInterrupt:
        logging.info("\nKeyboard interrupt received. Saving partial results...")
    finally:
        save_ambiguities(ambiguity_scores)
        logging.info("Results saved. Exiting now.")


if __name__ == "__main__":
    main()
