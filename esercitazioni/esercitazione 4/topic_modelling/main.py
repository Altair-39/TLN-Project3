import questionary
import logging
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from src.pipeline import dataset


def main():
    logging.basicConfig(
        level=logging.INFO,
        format=(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
    )
    text, title = dataset()
    model = SentenceTransformer("thenlper/gte-small")
    embeddings = model.encode(title[:1000], show_progress_bar=True)

    topic_model = BERTopic(embedding_model=model, min_topic_size=10, verbose=True)
    topics, probs = topic_model.fit_transform(title[:1000], embeddings)

    topic_info = topic_model.get_topic_info()

    file_format = questionary.select(
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
        logging.warning("❎ Skipped saving topic info")

    save_visuals = questionary.confirm("Do you want to save the visualizations as HTML?").ask()
    show_visuals = questionary.confirm("Do you want to display the visualizations now?").ask()

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


if __name__ == "__main__":
    main()
