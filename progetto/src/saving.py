import csv
import json
from typing import List, Set, Dict, Any


def extract_lemma_for_lang(
    synsets: List[Dict[str, Any]],
    synset_id: str,
    lang: str
) -> str:
    for synset in synsets:
        props = synset.get('properties', {})
        sid = props.get('synsetID', {}).get('id')
        language = props.get('language', '').upper()
        if sid == synset_id and language == lang.upper():
            return props.get('fullLemma') or props.get('simpleLemma') or "N/A"
    return "N/A"


def save_pseudoword(
    en_word: str,
    it_word: str,
    synsets: List[Dict[str, Any]],
    common_synsets: Set[str]
) -> None:
    filename = f'rsrc/pseudowords_{en_word}_{it_word}.csv'

    # Extract all synsets for each language
    en_synsets = {s['properties']['synsetID']['id'] for s in synsets
                  if s['properties']['language'].upper() == 'EN'}
    it_synsets = {s['properties']['synsetID']['id'] for s in synsets
                  if s['properties']['language'].upper() == 'IT'}

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['pseudoword', 'language', 'word',
                        'sense', 'synset_id', 'is_common'])

        # Save English-exclusive senses
        for synset_id in en_synsets - common_synsets:
            sense = extract_lemma_for_lang(synsets, synset_id, 'EN')
            writer.writerow([f"{en_word}-{it_word}", 'EN', en_word,
                             sense, synset_id, False])

        # Save Italian-exclusive senses
        for synset_id in it_synsets - common_synsets:
            sense = extract_lemma_for_lang(synsets, synset_id, 'IT')
            writer.writerow([f"{en_word}-{it_word}", 'IT', it_word,
                             sense, synset_id, False])

        # Save common senses
        for synset_id in common_synsets:
            en_sense = extract_lemma_for_lang(synsets, synset_id, 'EN')
            it_sense = extract_lemma_for_lang(synsets, synset_id, 'IT')
            writer.writerow([f"{en_word}-{it_word}", 'BOTH', f"{en_word}/{it_word}",
                             f"{en_sense}/{it_sense}", synset_id, True])


def save_ambiguities(
    data: List[Dict[str, Any]],
    filename: str = 'rsrc/ambiguity_scores.json'
) -> None:
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
