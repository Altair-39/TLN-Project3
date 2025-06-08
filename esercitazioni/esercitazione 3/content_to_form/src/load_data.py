
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required resources if not already present
nltk.download('punkt')
nltk.download('punkt-tab')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    # Initialize the Lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    # You can change the language if needed
    stop_words = set(stopwords.words('italian'))

    tokens = word_tokenize(text.lower())  # Convert to lowercase for case insensitivity
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()
    ]

    return " ".join(processed_tokens)


def extract_definitions_to_word(csv_path: str):
    df = pd.read_csv(csv_path)
    definitions_dict = {}

    for index, row in df.iterrows():
        term = row['Termine']
        definitions = row[2:].dropna().tolist()

        processed_definitions = [preprocess_text(defn) for defn in definitions]
        definitions_dict[term] = processed_definitions

    return definitions_dict
