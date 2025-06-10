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
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['pseudoword', 'en_word', 'it_word',
                        'en_sense', 'it_sense', 'common_synset_id'])

        for synset_id in common_synsets:
            en_sense = extract_lemma_for_lang(synsets, synset_id, 'EN')
            it_sense = extract_lemma_for_lang(synsets, synset_id, 'IT')
            pseudoword = f"{en_word}-{it_word}"
            writer.writerow([pseudoword, en_word, it_word,
                            en_sense, it_sense, synset_id])


def save_ambiguities(
    data: List[Dict[str, Any]],
    filename: str = 'rsrc/ambiguity_scores.json'
) -> None:
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
