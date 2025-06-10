import os
import questionary
import logging
from typing import Optional, List
from dotenv import load_dotenv, find_dotenv
from transformers import Pipeline
from src.guess import (
    guess_terms_from_definitions_zero_shot,
    guess_terms_from_definitions_one_shot,
    guess_terms_from_definitions_one_shot_with_clues
)
from src.topic import topic_zero_shot, topic_one_shot
from src.pipeline import load_pipeline


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


def safe_select(message: str, choices: List[str]) -> str:
    try:
        result = questionary.select(message, choices=choices).ask()
    except KeyboardInterrupt:
        exit(0)

    if result is None:
        exit(0)

    return result


def run_topic_task(pipe: Pipeline, topic_csv: str) -> None:
    mode = safe_select("Select mode:", ["zero-shot", "one-shot"])

    if mode == "zero-shot":
        topic_zero_shot(topic_csv, pipe)
    elif mode == "one-shot":
        topic_one_shot(topic_csv, pipe)


def run_guess_task(pipe: Pipeline, definitions_csv: str) -> None:
    mode = safe_select("Select mode:", ["zero-shot", "one-shot"])

    if mode == "zero-shot":
        guess_terms_from_definitions_zero_shot(definitions_csv, pipe)
    elif mode == "one-shot":
        clues = safe_select("With clues?", ["yes", "no"])
        if clues == "yes":
            guess_terms_from_definitions_one_shot_with_clues(definitions_csv, pipe)
        else:
            guess_terms_from_definitions_one_shot(definitions_csv, pipe)


def main() -> None:
    setup_logging()

    try:
        dotenv_path = find_dotenv()
        check_dotenv(dotenv_path)

        pipe = load_pipeline()

        topic_csv = os.getenv("TOPIC_CSV", "topic_info.csv")
        definitions_csv = os.getenv("DEFINITIONS_CSV", "rsrc/definizioni.csv")

        task_choice = safe_select("Select the task to run:", [
                                  "Label topics from keywords",
                                  "Guess terms from definitions"])

        if task_choice == "Label topics from keywords":
            run_topic_task(pipe, topic_csv)
        elif task_choice == "Guess terms from definitions":
            run_guess_task(pipe, definitions_csv)

    except KeyboardInterrupt:
        logging.info("Exiting!")


if __name__ == "__main__":
    main()
