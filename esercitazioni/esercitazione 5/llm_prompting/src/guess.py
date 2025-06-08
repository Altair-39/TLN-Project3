import logging
import pandas as pd
from transformers import Pipeline
from typing import List


def get_definitions(df: pd.DataFrame, row: pd.Series) -> List[str]:
    cols = [col for col in df.columns if col.startswith("P")]

    return [
        str(row[col]).strip()
        for col in cols
        if pd.notna(row[col])
    ]


def generate_text(pipe: Pipeline, prompt: str) -> str:
    output = pipe(prompt)[0]["generated_text"]
    guess = output.replace(prompt, "").strip().split("\n")[0]

    if guess.lower().startswith(("term:", "guess:")):
        guess = guess.split(":", 1)[1].strip()

    if not guess:
        guess = "Unclear"

    return guess


def saving(df: pd.DataFrame, guesses: List[str]) -> None:
    df["LLM_Guess"] = guesses
    df.to_csv("guessed_terms.csv", index=False)
    logging.info("Saved guessed terms to guessed_terms.csv")


def guess_terms_from_definitions_zero_shot(csv_path: str, pipe: Pipeline) -> None:
    df = pd.read_csv(csv_path)
    guesses = []

    for idx, row in df.iterrows():
        term = row["Termine"]
        definitions = get_definitions(df, row)

        if not definitions:
            guesses.append("No definitions provided")
            continue

        context = "\n".join(f"- {definition}" for definition in definitions)

        prompt = (
            "Based only on the following definitions,"
            "what single Italian noun do they all describe?\n"
            "Your answer must be a single word. Return only that word:\n\n"
            f"{context}"
        )

        try:
            guess = generate_text(pipe, prompt)
        except Exception as e:
            logging.error(f"Error guessing for row {idx}: {e}")
            guess = "Error generating guess"

        logging.info(f"Expected: {term} | LLM Guess: {guess}")
        guesses.append(guess)

    saving(df, guesses)


def guess_terms_from_definitions_one_shot(csv_path: str, pipe: Pipeline) -> None:
    df = pd.read_csv(csv_path)
    guesses = []

    for idx, row in df.iterrows():
        term = row["Termine"]
        definitions = get_definitions(df, row)

        if not definitions:
            guesses.append("No definitions provided")
            continue

        context = "\n".join(f"- {definition}" for definition in definitions)

        prompt = (
            "Example:\n"
            "Definitions: Struttura per abitare, composta da stanze."
            "Risponde al bisogno di rifugio.\n"
            "Answer: casa\n\n"
            "Definitions:\n"
            f"{context}\n"
            "Answer:"
        )

        try:
            guess = generate_text(pipe, prompt)
        except Exception as e:
            logging.error(f"Error guessing for row {idx}: {e}")
            guess = "Error generating guess"

        logging.info(f"Expected: {term} | LLM Guess: {guess}")
        guesses.append(guess)

    saving(df, guesses)


def guess_terms_from_definitions_one_shot_with_clues(csv_path: str, pipe: Pipeline) -> None:
    df = pd.read_csv(csv_path)
    guesses = []

    clues = [
        (
            "Al plur. (al sing., solo in usi region.), indumento (detto anche calzoni) "
            "maschile e femminile, che copre la persona dalla cintola in giù, "
            "dividendosi all’apertura delle gambe e avvolgendole in modo più o meno "
            "aderente, arrivando fino alla caviglia o a un’altezza variabile a seconda "
            "del modello."
        ),
        (
            "Strumento atto a dare immagini ingrandite di oggetti molto piccoli."
        ),
        (
            "Circostanza o complesso di circostanze da cui si teme che possa derivare "
            "grave danno."
        ),
        (
            "Aspetto del metodo scientifico che comprende un insieme di strategie, "
            "tecniche e procedimenti inventivi per ricercare un argomento, un concetto "
            "o una teoria adeguati a risolvere un problema dato."
        )
    ]

    for idx, row in df.iterrows():
        term = row["Termine"]
        definitions = get_definitions(df, row)

        if not definitions:
            guesses.append("No definitions provided")
            continue

        context = "\n".join(f"- {definition}" for definition in definitions)

        clue = clues[idx] if idx < len(clues) else ""

        prompt = (
            "Example:\n"
            "Definitions: Struttura per abitare, composta da stanze."
            "Risponde al bisogno di rifugio.\n"
            "Answer: casa\n\n"
            "Definitions:\n"
            f"{context}\n"
            f"Additional clue: {clue}\n"
            "Answer:"
        )

        try:
            guess = generate_text(pipe, prompt)
        except Exception as e:
            logging.error(f"Error guessing for row {idx}: {e}")
            guess = "Error generating guess"

        logging.info(f"Expected: {term} | LLM Guess: {guess}")
        guesses.append(guess)

    saving(df, guesses)
