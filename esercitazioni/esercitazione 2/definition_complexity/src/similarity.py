
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from difflib import SequenceMatcher
from typing import Dict, List, Tuple
from rich.table import Table
from rich.box import HEAVY
from statistics import mean

model = SentenceTransformer('all-MiniLM-L6-v2')


def compute_semantic_similarities(
    definitions_dict: Dict[str, List[str]]
) -> Dict[str, List[Tuple[str, str, float]]]:
    results = {}

    for term, defs in definitions_dict.items():
        term_results = []
        for def1, def2 in combinations(defs, 2):
            embeddings = model.encode([def1, def2], show_progress_bar=False)
            cosine_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            term_results.append((def1, def2, cosine_sim))

        results[term] = term_results

    return results


def compute_syntactic_similarities(
        definitions_dict: Dict[str, List[str]]
) -> Dict[str, List[Tuple[str, str, float]]]:
    results = {}

    for term, defs in definitions_dict.items():
        term_results = []
        for i in range(len(defs)):
            for j in range(i + 1, len(defs)):
                def1, def2 = defs[i], defs[j]
                score = SequenceMatcher(None, def1, def2).ratio()
                term_results.append((def1, def2, score))
        results[term] = term_results

    return results


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
