
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet as en_wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from deep_translator import GoogleTranslator

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def translate_to_english(text):
    return GoogleTranslator(source='it', target='en').translate(text)


def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('italian'))
    tokens = word_tokenize(text.lower())
    return " ".join([
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ])


def extract_definitions_to_word(csv_path: str):
    df = pd.read_csv(csv_path)
    definitions_dict = {}
    for _, row in df.iterrows():
        term = row['Termine']
        definitions = row[2:].dropna().tolist()
        processed_definitions = [preprocess_text(defn) for defn in definitions]
        definitions_dict[term] = processed_definitions
    return definitions_dict


def find_genus_candidates(defn_en):
    """
    Extract genus candidate word and synset from the definition text by 
    looking for genus-differentia patterns or fallback to first noun.
    """
    patterns = [
        r'is a kind of ([\w\s]+)',
        r'type of ([\w\s]+)',
        r'is a type of ([\w\s]+)',
        r'is a form of ([\w\s]+)',
        r'is a ([\w\s]+)',  # fallback pattern
    ]
    defn_lower = defn_en.lower()

    for pattern in patterns:
        match = re.search(pattern, defn_lower)
        if match:
            genus_phrase = match.group(1).strip()
            tokens = word_tokenize(genus_phrase)
            for token in tokens:
                synsets = en_wordnet.synsets(token, pos='n')
                if synsets:
                    return token, synsets[0]  # genus word + first synset

    # fallback: first noun in definition
    tokens = word_tokenize(defn_lower)
    for token in tokens:
        synsets = en_wordnet.synsets(token, pos='n')
        if synsets:
            return token, synsets[0]

    return None, None


def guess_synset_from_definition(term_en, defn_en):
    """
    Use genus to filter synsets of the term and pick the synset
    that matches the genus hypernym.
    """
    genus_word, genus_synset = find_genus_candidates(defn_en)
    term_synsets = en_wordnet.synsets(term_en, pos='n')
    if not term_synsets:
        return None

    if genus_synset is None:
        # no genus found: fallback to first synset
        return term_synsets[0]

    # Filter synsets that have genus_synset in their hypernym path
    filtered_synsets = []
    for syn in term_synsets:
        for path in syn.hypernym_paths():
            if genus_synset in path:
                filtered_synsets.append(syn)
                break

    if filtered_synsets:
        return filtered_synsets[0]  # pick first matching synset
    else:
        # fallback if none matched genus
        return term_synsets[0]


if __name__ == "__main__":
    csv_path = "rsrc/definizioni.csv"
    definitions_dict = extract_definitions_to_word(csv_path)

    for term, definitions in definitions_dict.items():
        term_en = translate_to_english(term)
        term_synsets = en_wordnet.synsets(term_en, pos='n')

        print(f"\nTerm: {term} ({term_en}) → Possible Synsets: {term_synsets}")

        for defn in definitions:
            defn_translated = translate_to_english(defn)
            predicted_synset = guess_synset_from_definition(term_en, defn_translated)

            print(f"\nDefinition (IT): {defn}")
            print(f"Definition (EN): {defn_translated}")
            print(f"Predicted Synset: {predicted_synset}")

            if predicted_synset and term_synsets:
                if predicted_synset in term_synsets:
                    print("✅ Synset Match!")
                else:
                    print("❌ Synset Mismatch.")
            else:
                print("⚠️ Synset could not be resolved.")
