import os
import logging
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from rich.console import Console
import questionary

from src.load_data import extract_definitions_to_word
from src.similarity import (
    compute_semantic_similarities,
    compute_lexical_similarities,
    create_similarity_table,
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def check_dotenv(dotenv_path: Optional[str]) -> None:
    if dotenv_path:
        load_dotenv(dotenv_path)
        logging.info(f"Loaded environment variables from: {dotenv_path}")
    else:
        logging.error("No .env file found.")


def load_environment() -> Optional[str]:
    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)
        logging.info(f"Loaded environment from: {dotenv_path}")
        return dotenv_path
    else:
        logging.error("No .env file found.")
        return None


def main() -> None:
    setup_logging()
    load_environment()

    try:

        definitions_csv = os.getenv("DEFINITIONS_CSV", "rsrc/definizioni.csv")

        console = Console()
        definitions = extract_definitions_to_word(definitions_csv)

        semantic_results = compute_semantic_similarities(definitions)
        lexical_results = compute_lexical_similarities(definitions)

        if not definitions:
            logging.warning("No definitions found.")
            return

        selected_term = questionary.select(
            "Select a term to view its definition similarities:",
            choices=list(definitions.keys())
        ).ask()

        if selected_term not in semantic_results:
            logging.warning(f"Term '{selected_term}' not found!")
            return

        table, avg_sem, avg_syn = create_similarity_table(selected_term,
                                                          semantic_results,
                                                          lexical_results)
        console.print(table)

    except KeyboardInterrupt:
        logging.info("Exiting!")


if __name__ == "__main__":
    main()
