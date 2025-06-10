import pandas as pd
import nltk
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from deep_translator import GoogleTranslator
from typing import Dict, List

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


def translate_text(text: str, target: str = 'en') -> str:
    try:
        return GoogleTranslator(source='auto', target=target).translate(text)
    except Exception as e:
        logging.warning(f"Translation failed for text: {text}. Error: {e}")
        return text


def preprocess_text(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(text.lower())
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
        and word.isalpha()
    ]

    return " ".join(processed_tokens)


def extract_definitions_to_word(csv_path: str) -> Dict[str, List[str]]:
    df = pd.read_csv(csv_path)
    definitions_dict = {}

    for index, row in df.iterrows():
        term = row['Termine']
        definitions = row[2:].dropna().tolist()

        translated_definitions = [translate_text(defn) for defn in definitions]
        processed_definitions = [preprocess_text(
            defn) for defn in translated_definitions]
        definitions_dict[term] = processed_definitions

    return definitions_dict
