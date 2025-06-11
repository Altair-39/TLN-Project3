import logging
from typing import Dict, List

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from deep_translator import GoogleTranslator

nltk_packages = ["punkt", "punkt_tab", "stopwords", "wordnet"]
for pkg in nltk_packages:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def translate_texts(texts: List[str], target: str = "en") -> List[str]:
    try:
        return GoogleTranslator(source="auto", target=target).translate_batch(texts)
    except Exception as e:
        logging.warning(f"Batch translation failed: {e}")
        return texts


def preprocess_text(text: str) -> str:
    tokens = word_tokenize(text.lower())
    return " ".join(
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words
    )


def extract_definitions_to_word(csv_path: str) -> Dict[str, List[str]]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Failed to load CSV file at '{csv_path}': {e}")
        return {}

    if "Termine" not in df.columns:
        logging.error("Missing 'Termine' column in CSV.")
        return {}

    definitions_dict = {}

    for _, row in df.iterrows():
        term = str(row.get("Termine", "")).strip()
        if not term:
            continue

        raw_definitions = [
            str(cell).strip()
            for cell in row[2:].dropna().tolist()
            if str(cell).strip()
        ]

        if not raw_definitions:
            logging.info(f"No definitions found for term '{term}'")
            continue

        translated_defs = translate_texts(raw_definitions)
        processed_defs = [preprocess_text(d) for d in translated_defs]

        definitions_dict[term] = processed_defs

    return definitions_dict
