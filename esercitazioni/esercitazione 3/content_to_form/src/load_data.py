import nltk
from typing import Dict, List

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk_packages = ["punkt", "punkt_tab", "stopwords", "wordnet"]
for pkg in nltk_packages:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    tokens = word_tokenize(text.lower())
    return " ".join(
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words
    )


def extract_definitions_to_word(csv_path: str) -> Dict[str, List[str]]:
    df: pd.DataFrame = pd.read_csv(csv_path)
    definitions_dict = {}

    for index, row in df.iterrows():
        term = row['Termine']
        definitions = row[2:].dropna().tolist()

        processed_definitions = [
            preprocess_text(defn) for defn in definitions]
        definitions_dict[term] = processed_definitions

    return definitions_dict
