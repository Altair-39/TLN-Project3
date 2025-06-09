from datasets import load_dataset
from typing import List, Tuple


def dataset() -> Tuple[List[str], List[str]]:
    dataset = load_dataset("zou-lab/MedCaseReasoning")["train"]
    title = dataset["title"]
    text = dataset["text"]
    return (text, title)
