from itertools import combinations
from statistics import mean
from difflib import SequenceMatcher
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rich.table import Table
from rich.box import HEAVY

model = SentenceTransformer('all-MiniLM-L6-v2')


model = SentenceTransformer('all-MiniLM-L6-v2')


def compute_semantic_for_term(term_defs: Tuple[str, List[str]]
                              ) -> Tuple[str, List[Tuple[str, str, float]]]:
    term, defs = term_defs
    if len(defs) < 2:
        return term, []

    embeddings = model.encode(defs, show_progress_bar=False)
    results = [
        (defs[i], defs[j], cosine_similarity([embeddings[i]], [embeddings[j]])[0][0])
        for i, j in combinations(range(len(defs)), 2)
    ]
    return term, results


def compute_syntactic_for_term(term_defs: Tuple[str, List[str]]
                               ) -> Tuple[str, List[Tuple[str, str, float]]]:
    term, defs = term_defs
    if len(defs) < 2:
        return term, []

    results = [
        (def1, def2, SequenceMatcher(None, def1, def2).ratio())
        for def1, def2 in combinations(defs, 2)
    ]
    return term, results


def compute_semantic_similarities(
    definitions_dict: Dict[str, List[str]],
    max_workers: int = 4
) -> Dict[str, List[Tuple[str, str, float]]]:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(compute_semantic_for_term, definitions_dict.items())
    return dict(results)


def compute_syntactic_similarities(
    definitions_dict: Dict[str, List[str]],
    max_workers: int = 4
) -> Dict[str, List[Tuple[str, str, float]]]:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(compute_syntactic_for_term, definitions_dict.items())
    return dict(results)


def create_similarity_table(
    term: str,
    semantic_results: Dict[str, List[Tuple[str, str, float]]],
    syntactic_results: Dict[str, List[Tuple[str, str, float]]]
) -> Tuple[Table, float, float]:
    sem_scores = []
    syn_scores = []

    table = Table(
        title=f"Semantic & Syntactic Similarity for: {term}",
        show_lines=True,
        box=HEAVY
    )

    table.add_column("Definition 1", style="dim", width=40)
    table.add_column("Definition 2", style="dim", width=40)
    table.add_column("Semantic", justify="center")
    table.add_column("Syntactic", justify="center")

    for (def1, def2, sem_score), (_, _, syn_score) in zip(
        semantic_results[term],
        syntactic_results[term]
    ):
        sem_scores.append(sem_score)
        syn_scores.append(syn_score)
        table.add_row(def1, def2, f"{sem_score:.4f}", f"{syn_score:.4f}")

    avg_sem = mean(sem_scores)
    avg_syn = mean(syn_scores)

    table.add_section()
    table.add_row("Mean", "", f"{avg_sem:.4f}", f"{avg_syn:.4f}")

    return table, avg_sem, avg_syn
