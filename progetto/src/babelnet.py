import logging
from typing import Optional, Tuple, List, Set, Dict, Any
import requests


def get_sense(
    lemma: str,
    targetLang: List[str],
    key: str,
    source: str = "WIKI"
) -> Optional[List[Dict[str, Any]]]:
    url = 'https://babelnet.io/v9/getSenses'
    searchLang = targetLang[0]
    params = {
        'lemma': lemma,
        'searchLang': searchLang,
        'targetLang': targetLang,
        'key': key,
        'source': source
    }

    response = requests.get(url, params=params, timeout=10)
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching synsets: {e}")
        return None


def find_synset_language_dict(
    synsets: List[Dict[str, Any]],
    langs: List[str]
) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """
    For a list of synsets, returns:
    - A dict mapping each language to its set of synset IDs
    - The set of synsets common to all languages
    """
    lang_synsets: Dict[str, Set[str]] = {lang.upper(): set() for lang in langs}

    for synset in synsets:
        props = synset.get('properties', {})
        synset_id = props.get('synsetID', {}).get('id')
        lang = props.get('language', '').upper()

        if synset_id and lang in lang_synsets:
            lang_synsets[lang].add(synset_id)

    synset_sets = [s for s in lang_synsets.values() if s]
    if synset_sets:
        common_synsets = set.intersection(*synset_sets)
    else:
        common_synsets = set()

    return lang_synsets, common_synsets
