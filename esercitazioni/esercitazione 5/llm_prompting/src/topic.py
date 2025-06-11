from typing import Callable
import logging
import pandas as pd

from transformers import Pipeline


def generate_text(pipe: Pipeline, prompt: str) -> str:
    response = pipe(prompt)[0]["generated_text"]
    text = response.replace(prompt, "").strip()

    label = text.split("\n")[0]

    if label.lower().startswith(("label:", "topic:")):
        return label.split(":", 1)[1].strip()

    if not label:
        return "Unclear Topic"

    return label


def saving(info, labels) -> None:
    info["LLM_Label"] = labels
    info.to_csv("labeled_topics.csv", index=False)
    logging.info("Saved labeled topics to labeled_topics.csv")


def label_topics(
    csv_path: str,
    pipe: Pipeline,
    prompt_fn: Callable[[int, str], str]
) -> None:
    topic_info = pd.read_csv(csv_path)
    topic_labels = []

    for idx, row in topic_info.iterrows():
        topic_id = row["Topic"]
        words = row["Representation"]

        if topic_id == -1 or pd.isna(words):
            topic_labels.append("Outlier or Undefined")
            continue

        prompt = prompt_fn(topic_id, words)

        try:
            label = generate_text(pipe, prompt)
        except Exception as e:
            logging.error(f"Error processing topic {topic_id}: {e}")
            label = "Error generating label"

        logging.info(f"Topic {topic_id}: {label}")
        topic_labels.append(label)

    saving(topic_info, topic_labels)


def zero_shot_prompt_topic(topic_id: int, words: str) -> str:
    return (
        f"Given the keywords: {words}, generate a concise label for this topic. "
        "Return only the label, no explanation, no punctuation:"
    )


def one_shot_prompt_topic(topic_id: int, words: str) -> str:
    return (
        "Given a list of keywords related to a topic,"
        "provide a short label that describes the topic.\n\n"
        "Example:\n"
        "Words: tumor, cancer, leucemia, benign, malignant\n"
        "Topic: types of tumors\n\n"
        f"Words: {words}\n"
        "Topic:"
    )
