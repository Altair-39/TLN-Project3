import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from umap.umap_ import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic


def dataset():
    dataset = load_dataset("zou-lab/MedCaseReasoning")["train"]
    title = dataset["title"]
    text = dataset["text"]
    return (text, title)


def convert(title):
    model = SentenceTransformer("thenlper/gte-small")
    embeddings = model.encode(title, show_progress_bar=True)
    embeddings.shape
    return embeddings


def reduce(embeddings):
    umap_model = UMAP(n_components=2, min_dist=0.0, metric="cosine",
                      random_state=42)
    reduced_embeddings = umap_model.fit_transform(embeddings)
    return reduced_embeddings


def group(reduced_embeddings):
    hdbscan_model = HDBSCAN(min_cluster_size=10, metric="euclidean",
                            cluster_selection_method="eom").fit(reduced_embeddings)
    clusters = hdbscan_model.labels_
    len(set(clusters))
    return clusters


def printing(clusters, title):
    cluster = 0
    for index in np.where(clusters == cluster)[0][:3]:
        print(title[index][:300] + "… \n")


def plot(reduced_embeddings, clusters, text):
    df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
    df["title"] = text[:1000]
    df["cluster"] = [str(c) for c in clusters]
    to_plot_df = df.loc[df.cluster != "-1", :]
    outliers_df = df.loc[df.cluster == "-1", :]
    plt.scatter(outliers_df.x, outliers_df.y, alpha=0.6, s=2, c="grey")
    plt.scatter(to_plot_df.x, to_plot_df.y, c=to_plot_df.cluster.astype(int),
                alpha=0.6, s=2, cmap="tab20b")
    plt.axis("off")
    return plt


def main():
    (text, title) = dataset()
    model = SentenceTransformer("thenlper/gte-small")
    embeddings = model.encode(title[:1000], show_progress_bar=True)

    topic_model = BERTopic(embedding_model=model, min_topic_size=10, verbose=True)
    topics, probs = topic_model.fit_transform(title[:1000], embeddings)

    # Get topic info dataframe
    topic_info = topic_model.get_topic_info()

    # Save topic info to CSV
    topic_info.to_csv("topic_info.csv", index=False)

    # Save topic overview as interactive HTML (no Kaleido needed)
    fig_overview = topic_model.visualize_topics()
    fig_overview.write_html("topics_overview.html")

    # Save bar chart of top 10 topics as interactive HTML
    fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
    fig_barchart.write_html("top_topics_barchart.html")

    # Also display interactively (optional)
    fig_overview.show()
    fig_barchart.show()


if __name__ == "__main__":
    main()
