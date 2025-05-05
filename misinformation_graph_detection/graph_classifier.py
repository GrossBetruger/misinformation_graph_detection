from pathlib import Path
from typing import Any

import pandas as pd
import sklearn
from analyze import (
    analyze_community_structure,
    build_social_graph,
    load_graphs_from_dir,
)
import kagglehub
import networkx as nx

path = kagglehub.dataset_download("arashnic/misinfo-graph")
PATH = Path(path)


def load_graphs(graphs_dir: Path) -> list[nx.Graph]:
    """
    Load all graphs from a directory of subfolders, each containing 'edges.txt' and 'nodes.csv'.
    """

    conspiracy_graphs = load_graphs_from_dir(graphs_dir / "Other_Graphs")
    fiveg_conspiracy_graphs = load_graphs_from_dir(graphs_dir / "5G_Conspiracy_Graphs")
    non_conspiracy_graphs = load_graphs_from_dir(graphs_dir / "Non_Conspiracy_Graphs")

    return conspiracy_graphs, fiveg_conspiracy_graphs, non_conspiracy_graphs


def create_dataset(
    graphs_dir: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create a dataset of graphs from a directory of subfolders, each containing 'edges.txt' and 'nodes.csv'.
    """

    conspiracy_graphs, fiveg_conspiracy_graphs, non_conspiracy_graphs = load_graphs(
        graphs_dir
    )

    conspiracy_features = [
        analyze_community_structure(graph) for graph in conspiracy_graphs
    ]
    conspiracy_vector = pd.DataFrame.from_records(conspiracy_features)

    fiveg_conspiracy_features = [
        analyze_community_structure(graph) for graph in fiveg_conspiracy_graphs
    ]
    fiveg_conspiracy_vector = pd.DataFrame.from_records(fiveg_conspiracy_features)

    non_conspiracy_features = [
        analyze_community_structure(graph) for graph in non_conspiracy_graphs
    ]
    non_conspiracy_vector = pd.DataFrame.from_records(non_conspiracy_features)

    # to single list
    features = pd.concat(
        [conspiracy_vector, fiveg_conspiracy_vector, non_conspiracy_vector]
    )
    labels = (
        [0] * len(conspiracy_features)
        + [1] * len(fiveg_conspiracy_features)
        + [2] * len(non_conspiracy_features)
    )
    labels = pd.Series(labels)
    return features, labels


def train_model(
    X_train: list[list[Any]],
    y_train: list[int],
) -> sklearn.ensemble.RandomForestClassifier:
    """
    Train a model to classify graphs as conspiracy or non-conspiracy.
    """
    model = sklearn.ensemble.RandomForestClassifier()
    model.fit(X, y)
    return model


if __name__ == "__main__":

    X, y = create_dataset(PATH)
    print(X.iloc[0], y.iloc[0])
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.1
    )

    model = train_model(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)
    print(sklearn.metrics.classification_report(y_test, y_pred))
