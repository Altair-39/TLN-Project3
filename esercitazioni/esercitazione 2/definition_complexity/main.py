import os
import questionary
import logging
from rich.console import Console
from typing import Optional
from dotenv import load_dotenv, find_dotenv
from src.load_data import extract_definitions_to_word
from src.similarity import compute_semantic_similarities, compute_syntactic_similarities
from src.similarity import create_similarity_table


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


def main() -> None:
    setup_logging()

    try:
        dotenv_path = find_dotenv()
        check_dotenv(dotenv_path)

        definitions_csv = os.getenv("DEFINITIONS_CSV", "rsrc/definizioni.csv")

        console = Console()
        definitions = extract_definitions_to_word(definitions_csv)

        semantic_results = compute_semantic_similarities(definitions)
        syntactic_results = compute_syntactic_similarities(definitions)

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
                                                          syntactic_results)
        console.print(table)

    except KeyboardInterrupt:
        logging.info("Exiting!")


if __name__ == "__main__":
    main()
