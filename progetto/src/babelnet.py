import logging
from typing import Optional, Tuple, List, Set, Dict, Any
import requests


def get_sense(
    lemma: str,
    targetLang: List[str],
    key: str,
    searchLang: str = "EN",
    source: str = "WIKI"
) -> Optional[List[Dict[str, Any]]]:
    url = 'https://babelnet.io/v9/getSenses'
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


def find_synset_language_sets(
    synsets: List[Dict[str, Any]]
) -> Tuple[int, int, Set[str]]:
    en_synsets = set()
    it_synsets = set()

    for synset in synsets:
        props = synset.get('properties', {})
        synset_id = props.get('synsetID', {}).get('id')
        lang = props.get('language')

        if synset_id and lang:
            if lang.upper() == 'EN':
                en_synsets.add(synset_id)
            elif lang.upper() == 'IT':
                it_synsets.add(synset_id)

    common = en_synsets.intersection(it_synsets)
    return len(en_synsets), len(it_synsets), common
