
import requests


def get_conceptnet_entries(word):
    url = f"http://api.conceptnet.io/c/en/{word}"
    obj = requests.get(url).json()
    return obj['edges']
