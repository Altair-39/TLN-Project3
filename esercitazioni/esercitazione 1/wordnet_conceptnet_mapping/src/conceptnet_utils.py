
import requests


def get_conceptnet_entries(word):
    word = word.lower().replace(" ", "_")
    url = f"http://api.conceptnet.io/c/en/{word}"
    obj = requests.get(url).json()
    return obj['edges']
