
import os
import csv
import logging
from dotenv import load_dotenv, find_dotenv
from typing import Optional, Tuple, List, Dict, Set

from src.babelnet import get_sense
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


def load_word_tuples(filepath: str) -> List[Tuple[str, ...]]:
    tuples = []
    try:
        with open(filepath, newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                words = tuple(word.strip() for word in row if word.strip())
                if len(words) >= 2:
                    tuples.append(words)
    except FileNotFoundError:
        logging.error(f"Input file not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error reading input file: {e}")
        raise
    return tuples


def find_synset_language_dict(synsets: List[dict]) -> Dict[str, Set[str]]:
    """
    Extract sets of synset IDs for each language from synsets.
    """
    lang_synsets: Dict[str, Set[str]] = {}
    for synset in synsets:
        props = synset.get('properties', {})
        synset_id = props.get('synsetID', {}).get('id')
        lang = props.get('language', '').upper()
        if synset_id and lang:
            lang_synsets.setdefault(lang, set()).add(synset_id)
    return lang_synsets


def extract_lemma_for_lang(synsets: List[dict], synset_id: str, lang: str) -> str:
    """
    Extract lemma from synsets for given language and synset_id.
    """
    lang = lang.upper()
    for synset in synsets:
        props = synset.get('properties', {})
        synset_id_prop = props.get('synsetID', {}).get('id')
        synset_lang = props.get('language', '').upper()
        if synset_id_prop == synset_id and synset_lang == lang:
            # Return the lemma property if exists
            lemma = props.get('lemma')
            if lemma:
                return lemma
    return ''


def save_pseudoword_multi(pseudoword: str, words: Tuple[str, ...], synsets: List[dict],
                          common_synsets: Set[str], langs: List[str]) -> None:
    """
    Save pseudoword info to CSV or delegate to your existing save_pseudoword if suitable.
    Here, we extend it for multiple languages.
    """
    save_pseudoword(pseudoword, '-'.join(words), synsets, common_synsets)


def process_word_tuple(words: Tuple[str, ...], langs: List[str], api_key: str) -> Optional[dict]:
    """
    Process a tuple of words for multiple languages and return ambiguity reduction info.
    """
    if len(words) != len(langs):
        logging.error(f"Word tuple and language list length mismatch: {words}, {langs}")
        return None

    # We pick the first word as lemma to query BabelNet (assuming get_sense searches for it and langs)
    synsets = get_sense(words[0], langs, api_key)
    if not synsets:
        logging.warning(f"No synsets found for words: {words}")
        return None

    lang_synsets = find_synset_language_dict(synsets)
    synsets_sets = [lang_synsets.get(lang.upper(), set()) for lang in langs]

    if not all(synsets_sets):
        logging.info(f"Missing synsets for some languages in {words}")

    # Intersection of all synset sets
    common_synsets = set.intersection(*synsets_sets) if synsets_sets else set()

    total_synsets_count = sum(len(s) for s in synsets_sets)
    if total_synsets_count == 0:
        ambiguity_reduction = 0.0
    else:
        # Generalized ambiguity reduction formula:
        ambiguity_reduction = (total_synsets_count -
                               len(common_synsets) * len(langs)) / total_synsets_count

    pseudoword = '-'.join(words)
    save_pseudoword_multi(pseudoword, words, synsets, common_synsets, langs)

    return {
        'pseudoword': pseudoword,
        'ambiguity_reduction': round(ambiguity_reduction, 3)
    }


def main() -> None:
    setup_logging()
    dotenv_path = find_dotenv()
    check_dotenv(dotenv_path)

    API_KEY = os.getenv('BABELNET_API_KEY')
    input_file = os.getenv('WORD_PAIRS')
    langs_env = os.getenv('LANGUAGES')  # e.g., "EN,IT,FR,DE"

    if not API_KEY or not input_file or not langs_env:
        logging.error(
            "Required environment variables missing (BABELNET_API_KEY, WORD_PAIRS, LANGUAGES).")
        return

    langs = [lang.strip().upper() for lang in langs_env.split(',')]

    try:
        word_tuples = load_word_tuples(input_file)
    except Exception:
        return

    ambiguity_scores = []

    try:
        for words in word_tuples:
            if len(words) != len(langs):
                logging.warning(f"Skipping tuple due to length mismatch: {words}")
                continue

            logging.info(f"Processing tuple: {words}")
            result = process_word_tuple(words, langs, API_KEY)
            if result:
                logging.info(f"  Ambiguity reduction: {
                             result['ambiguity_reduction']:.3f}")
                ambiguity_scores.append(result)

    except KeyboardInterrupt:
        logging.info("\nKeyboard interrupt received. Saving partial results...")
    finally:
        save_ambiguities(ambiguity_scores)
        logging.info("Results saved. Exiting now.")


if __name__ == "__main__":
    main()
