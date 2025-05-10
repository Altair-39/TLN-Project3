
from src.load_data import extract_definitions_to_word
from src.similarity import compute_semantic_similarities, compute_syntactic_similarities
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.box import HEAVY
from statistics import mean


def main():
    console = Console()
    sem_scores = []
    syn_scores = []

    definitions = extract_definitions_to_word('rsrc/definizioni.csv')
    semantic_results = compute_semantic_similarities(definitions)
    syntactic_results = compute_syntactic_similarities(definitions)

    console.print("\n[bold cyan]Available Terms:[/bold cyan]")
    for term in definitions.keys():
        console.print(f"- {term}")

    selected_term = Prompt.ask("\nEnter a term to view its definition similarities")

    if selected_term not in semantic_results:
        console.print(f"[bold red]Term '{selected_term}' not found![/bold red]")
        return

    table = Table(
        title=f"Semantic & Syntactic Similarity for: {selected_term}",
        show_lines=True,
        box=HEAVY
    )

    table.add_column("Definition 1", style="dim", width=40)
    table.add_column("Definition 2", style="dim", width=40)
    table.add_column("Semantic", justify="center")
    table.add_column("Syntactic", justify="center")

    for (def1, def2, sem_score), (_, _, syn_score) in zip(
        semantic_results[selected_term],
        syntactic_results[selected_term]
    ):
        sem_scores.append(sem_score)
        syn_scores.append(syn_score)
        table.add_row(def1, def2, f"{sem_score:.4f}", f"{syn_score:.4f}")

    avg_sem = mean(sem_scores)
    avg_syn = mean(syn_scores)

    table.add_section()  # visually separates the mean row
    table.add_row(
        "[bold]Mean[/bold]", "", 
        f"[bold]{avg_sem:.4f}[/bold]", 
        f"[bold]{avg_syn:.4f}[/bold]"
    )

    console.print(table)


if __name__ == "__main__":
    main()

