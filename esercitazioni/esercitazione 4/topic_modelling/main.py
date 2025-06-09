import logging
import questionary
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from typing import Tuple, List
from src.pipeline import dataset, generate_embeddings, create_topic_model


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_data() -> Tuple[List[str], List[str]]:
    return dataset()


def save_topic_info(topic_model: BERTopic) -> None:
    topic_info = topic_model.get_topic_info()
    file_format: str = questionary.select(
        "Choose a format to save topic info:",
        choices=["CSV", "JSON", "None"]
    ).ask()

    if file_format == "CSV":
        topic_info.to_csv("topic_info.csv", index=False)
        logging.info("Saved as topic_info.csv")
    elif file_format == "JSON":
        topic_info.to_json("topic_info.json", orient="records", lines=True)
        logging.info("Saved as topic_info.json")
    else:
        logging.warning("Skipped saving topic info")


def handle_visualizations(topic_model: BERTopic) -> None:
    save_visuals: bool = questionary.confirm(
        "Do you want to save the visualizations as HTML?").ask()
    show_visuals: bool = questionary.confirm(
        "Do you want to display the visualizations now?").ask()

    fig_overview = topic_model.visualize_topics()
    fig_barchart = topic_model.visualize_barchart(top_n_topics=10)

    if save_visuals:
        fig_overview.write_html("topics_overview.html")
        fig_barchart.write_html("top_topics_barchart.html")
        logging.info("Visualizations saved as HTML.")
    else:
        logging.warning("Skipped saving visualizations.")

    if show_visuals:
        fig_overview.show()
        fig_barchart.show()
    else:
        logging.warning("Skipped displaying visualizations.")


def main() -> None:
    setup_logging()

    text, title = load_data()
    model: SentenceTransformer = SentenceTransformer("thenlper/gte-small")
    titles_subset: List[str] = title[:1000]

    embeddings = generate_embeddings(model, titles_subset)
    topic_model, topics, probs = create_topic_model(model, titles_subset, embeddings)

    save_topic_info(topic_model)
    handle_visualizations(topic_model)


if __name__ == "__main__":
    main()
