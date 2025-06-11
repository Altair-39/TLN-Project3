from typing import List, Optional, Tuple

import numpy as np
from bertopic import BERTopic
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def dataset() -> Tuple[List[str], List[str]]:
    ds = load_dataset("zou-lab/MedCaseReasoning")["train"]
    title: List[str] = ds["title"]
    text: List[str] = ds["text"]
    return text, title


def generate_embeddings(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    return model.encode(texts, show_progress_bar=True)


def create_topic_model(
    model: SentenceTransformer,
    texts: List[str],
    embeddings: np.ndarray
) -> Tuple[BERTopic, List[int], List[Optional[float]]]:
    topic_model = BERTopic(embedding_model=model, min_topic_size=10, verbose=True)
    topics, probs = topic_model.fit_transform(texts, embeddings)
    return topic_model, topics, probs
