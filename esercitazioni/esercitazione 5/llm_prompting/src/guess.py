import logging
from typing import Callable, List

import pandas as pd
from transformers import Pipeline


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


def guess_terms_from_definitions(
    csv_path: str,
    pipe: Pipeline,
    prompt_fn: Callable[[int, str, str], str],
    debug: bool = False
) -> None:
    df = pd.read_csv(csv_path)
    guesses = []

    for idx, row in df.iterrows():
        term = row["Termine"]
        definitions = get_definitions(df, row)

        if not definitions:
            guesses.append("No definitions provided")
            continue

        context = "\n".join(f"- {definition}" for definition in definitions)
        prompt = prompt_fn(idx, term, context)

        if debug:
            print(f"\n[DEBUG] Prompt for row {idx}:\n{prompt}\n")

        try:
            guess = generate_text(pipe, prompt)
        except Exception as e:
            logging.error(f"Error guessing for row {idx}: {e}")
            guess = "Error generating guess"

        logging.info(f"Expected: {term} | LLM Guess: {guess}")
        guesses.append(guess)

    saving(df, guesses)


def zero_shot_prompt(idx: int, term: str, context: str) -> str:
    return (
        "Based only on the following definitions,"
        " what single Italian noun do they all describe?\n"
        "Your answer must be a single word. Return only that word:\n\n"
        f"{context}"
    )


def one_shot_prompt(idx: int, term: str, context: str) -> str:
    return (
        "Example:\n"
        "Definitions: Struttura per abitare, composta da stanze."
        " Risponde al bisogno di rifugio.\n"
        "Answer: casa\n\n"
        "Definitions:\n"
        f"{context}\n"
        "Answer:"
    )


def one_shot_with_clues_prompt(idx: int, term: str, context: str) -> str:
    clues = [
        (
            "Al plurale (al singolare, solo in usi regionali), indumento (detto anche "
            "calzoni) maschile e femminile, che copre la persona dalla cintura in giù, "
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
        ),
    ]

    clue = clues[idx] if idx < len(clues) else ""

    return (
        "You will be given a list of definitions and an additional clue describing a"
        "concept. "
        "Based ONLY on the information provided, guess the single Italian noun these"
        "definitions describe. "
        "Return ONLY that single word, with no explanation, examples, or punctuation.\n"
        "Example:\n"
        "Definitions: Struttura per abitare, composta da stanze. Risponde al bisogno"
        "di rifugio.\n"
        "Answer: casa\n\n"
        "Definitions:\n"
        f"{context}\n"
        f"Additional clue: {clue}\n"
        "Answer:"
    )
