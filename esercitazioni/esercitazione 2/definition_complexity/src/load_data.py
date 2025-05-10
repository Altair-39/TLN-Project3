
import pandas as pd


def extract_definitions_to_word(csv_path: str):

    df = pd.read_csv(csv_path)

    definitions_dict = {}

    for index, row in df.iterrows():
        term = row['Termine']
        definitions = row[2:].dropna().tolist()
        definitions_dict[term] = definitions

    return definitions_dict
