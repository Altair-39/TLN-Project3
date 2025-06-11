import sys
import logging
import os
from typing import List, Optional
import argparse

import questionary
from dotenv import load_dotenv, find_dotenv
from transformers import Pipeline


from src.guess import (
        guess_terms_from_definitions,
        zero_shot_prompt,
        one_shot_prompt,
        one_shot_with_clues_prompt,
        )
from src.pipeline import load_pipeline
from src.topic import (
        label_topics,
        zero_shot_prompt_topic,
        one_shot_prompt_topic,
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


def safe_select(message: str, choices: List[str]) -> str:
    try:
        result = questionary.select(message, choices=choices).ask()
    except KeyboardInterrupt:
        logging.info("User interrupted. Exiting.")
        sys.exit(0)

    if result is None:
        logging.info("No selection made. Exiting.")
        sys.exit(0)

    return result


def run_topic_task(pipe: Pipeline, topic_csv: str) -> None:
    mode = safe_select("Select mode:", ["zero-shot", "one-shot"])

    if mode == "zero-shot":
        label_topics(topic_csv, pipe, zero_shot_prompt_topic)
    else:
        label_topics(topic_csv, pipe, one_shot_prompt_topic)


def run_guess_task(pipe: Pipeline, definitions_csv: str, debug: bool = False) -> None:
    mode = safe_select("Select mode:", ["zero-shot", "one-shot"])

    if mode == "zero-shot":
        guess_terms_from_definitions(definitions_csv, pipe,
                                     zero_shot_prompt, debug=debug)
    elif mode == "one-shot":
        clues = safe_select("With clues?", ["yes", "no"])
        if clues == "yes":
            guess_terms_from_definitions(definitions_csv, pipe,
                                         one_shot_with_clues_prompt, debug=debug)
        else:
            guess_terms_from_definitions(definitions_csv, pipe,
                                         one_shot_prompt, debug=debug)


def load_environment() -> None:
    dotenv_path = find_dotenv()
    check_dotenv(dotenv_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode to print prompts")
    args = parser.parse_args()
    setup_logging()
    load_environment()

    try:
        pipe = load_pipeline()
        topic_csv = os.getenv("TOPIC_CSV", "topic_info.csv")
        definitions_csv = os.getenv("DEFINITIONS_CSV", "rsrc/definizioni.csv")

        task_choice = safe_select("Select the task to run:", [
            "Label topics from keywords",
            "Guess terms from definitions"
        ])

        if task_choice == "Label topics from keywords":
            run_topic_task(pipe, topic_csv)
        elif task_choice == "Guess terms from definitions":
            run_guess_task(pipe, definitions_csv, debug=args.debug)

    except KeyboardInterrupt:
        logging.info("Exiting!")


if __name__ == "__main__":
    main()
