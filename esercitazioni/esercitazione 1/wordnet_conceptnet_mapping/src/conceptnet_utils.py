from typing import List, Dict, Any
import logging
import requests
from requests.exceptions import RequestException


def get_conceptnet_entries(word: str) -> List[Dict[str, Any]]:
    try:
        normalized_word = word.lower().replace(" ", "_")
        url = f"http://api.conceptnet.io/c/en/{normalized_word}"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        if 'edges' not in data:
            raise ValueError("ConceptNet response missing 'edges' key")
        return data['edges']

    except RequestException as e:
        logging.error(f"ConceptNet API request failed for '{word}': {str(e)}")
        raise
    except ValueError as e:
        logging.error(f"Invalid ConceptNet response for '{word}': {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error processing '{word}': {str(e)}")
        raise
