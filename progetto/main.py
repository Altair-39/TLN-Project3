import csv
import concurrent.futures
import logging
import os
from functools import partial
from typing import Dict, List, Optional, Set, Tuple

from dotenv import find_dotenv, load_dotenv
import matplotlib.pyplot as plt
import numpy as np

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
    lang_synsets: Dict[str, Set[str]] = {}
    for synset in synsets:
        props = synset.get('properties', {})
        synset_id = props.get('synsetID', {}).get('id')
        lang = props.get('language', '').upper()
        if synset_id and lang:
            lang_synsets.setdefault(lang, set()).add(synset_id)
    return lang_synsets


def extract_lemma_for_lang(synsets: List[dict], synset_id: str, lang: str) -> str:
    lang = lang.upper()
    for synset in synsets:
        props = synset.get('properties', {})
        synset_id_prop = props.get('synsetID', {}).get('id')
        synset_lang = props.get('language', '').upper()
        if synset_id_prop == synset_id and synset_lang == lang:
            lemma = props.get('lemma')
            if lemma:
                return lemma
    return ''


def save_pseudoword_multi(pseudoword: str, words: Tuple[str, ...], synsets: List[dict],
                          common_synsets: Set[str], langs: List[str]) -> None:
    save_pseudoword(pseudoword, '-'.join(words), synsets, common_synsets)


def process_word_tuple(words: Tuple[str, ...], langs: List[str], api_key: str
                       ) -> Optional[dict]:
    if len(words) != len(langs):
        logging.error(f"Word tuple and language list length mismatch: {words}, {langs}")
        return None

    synsets = get_sense(words[0], langs, api_key)
    if not synsets:
        logging.warning(f"No synsets found for words: {words}")
        return None

    lang_synsets = find_synset_language_dict(synsets)
    synsets_sets = [lang_synsets.get(lang.upper(), set()) for lang in langs]

    if not all(synsets_sets):
        logging.info(f"Missing synsets for some languages in {words}")

    common_synsets = set.intersection(*synsets_sets) if synsets_sets else set()

    total_synsets_count = sum(len(s) for s in synsets_sets)
    if total_synsets_count == 0:
        ambiguity_reduction = 0.0
    else:
        ambiguity_reduction = 1 - (len(common_synsets) * len(langs)
                                   ) / total_synsets_count

    pseudoword = '-'.join(words)
    save_pseudoword_multi(pseudoword, words, synsets, common_synsets, langs)

    return {
        'pseudoword': pseudoword,
        'ambiguity_reduction': round(ambiguity_reduction, 3)
    }


def plot_results(ambiguity_scores: List[dict]) -> None:
    if not ambiguity_scores:
        logging.warning("No data available for plotting.")
        return

    pseudowords = [result['pseudoword'] for result in ambiguity_scores]
    scores = [result['ambiguity_reduction'] for result in ambiguity_scores]

    sorted_indices = np.argsort(scores)
    sorted_pseudowords = [pseudowords[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]

    plt.figure(figsize=(12, 8))

    y_pos = np.arange(len(sorted_pseudowords))
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_pseudowords)))

    bars = plt.barh(y_pos, sorted_scores, color=colors)
    plt.yticks(y_pos, sorted_pseudowords)
    plt.xlabel('Ambiguity Reduction Score')
    plt.title('Pseudoword Ambiguity Reduction Results')

    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                 f'{width:.2f}',
                 ha='left', va='center')

    plt.tight_layout()

    plot_filename = 'ambiguity_reduction_plot.png'
    plt.savefig(plot_filename)
    logging.info(f"Saved plot as {plot_filename}")


def process_word_tuple_wrapper(words: Tuple[str, ...], langs: List[str], api_key: str
                               ) -> Optional[dict]:
    try:
        return process_word_tuple(words, langs, api_key)
    except Exception as e:
        logging.error(f"Error processing {words}: {str(e)}")
        return None


def main() -> None:
    setup_logging()
    dotenv_path = find_dotenv()
    check_dotenv(dotenv_path)

    API_KEY = os.getenv('BABELNET_API_KEY')
    input_file = os.getenv('WORD_PAIRS')
    langs_env = os.getenv('LANGUAGES')

    if not API_KEY or not input_file or not langs_env:
        logging.error("Required environment variables missing")
        return

    langs = [lang.strip().upper() for lang in langs_env.split(',')]

    try:
        word_tuples = load_word_tuples(input_file)
        valid_tuples = [words for words in word_tuples if len(words) == len(langs)]
        max_workers = min(4, os.cpu_count() or 1)
        chunk_size = 50
        ambiguity_scores = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            process_func = partial(process_word_tuple_wrapper,
                                   langs=langs, api_key=API_KEY)

            for i in range(0, len(valid_tuples), chunk_size):
                chunk = valid_tuples[i:i + chunk_size]
                future_to_tuple = {
                    executor.submit(process_func, words): words
                    for words in chunk
                }

                for future in concurrent.futures.as_completed(future_to_tuple):
                    words = future_to_tuple[future]
                    try:
                        result = future.result()
                        if result:
                            ambiguity_scores.append(result)
                            logging.info(f"Completed {words} → Score:"
                                         f"{result['ambiguity_reduction']:.3f}")
                    except Exception as e:
                        logging.error(f"Error processing {words}: {str(e)}")

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        save_ambiguities(ambiguity_scores)
        plot_results(ambiguity_scores)
        logging.info(f"Processed {len(ambiguity_scores)}/{len(word_tuples)} tuples")


if __name__ == "__main__":
    main()
