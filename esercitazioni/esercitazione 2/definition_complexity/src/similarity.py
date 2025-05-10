
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from difflib import SequenceMatcher

model = SentenceTransformer('all-MiniLM-L6-v2')


def compute_semantic_similarities(definitions_dict):
    """
    Compute cosine semantic similarities between all definition pairs.
    Returns: {term: [(def1, def2, cosine_similarity)]}
    """
    results = {}

    for term, defs in definitions_dict.items():
        term_results = []
        for def1, def2 in combinations(defs, 2):
            embeddings = model.encode([def1, def2])
            cosine_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            term_results.append((def1, def2, cosine_sim))

        results[term] = term_results

    return results


def compute_syntactic_similarities(definitions_dict):
    """
    Compute syntactic similarity (based on SequenceMatcher ratio) between all definition pairs.
    Returns: {term: [(def1, def2, similarity_score)]}
    """
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
