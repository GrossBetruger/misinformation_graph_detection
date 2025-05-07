from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost
from misinformation_graph_detection.analyze import (
    analyze_community_structure,
    load_graphs_from_dir,
)
from sklearn.inspection import permutation_importance
import kagglehub
import networkx as nx

"""Placeholder for dataset path; download only when run as script."""
PATH: Path | None = None


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

    model = sklearn.ensemble.RandomForestClassifier(
        n_estimators=100,
        # max_depth=5, # low depth (5) gets better precision for conspiracy with much smaller recall
        # random_state=42,
    )

    # model = xgboost.XGBClassifier()

    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    # Download dataset from Kaggle
    path = kagglehub.dataset_download("arashnic/misinfo-graph")
    PATH = Path(path)
    X, y = create_dataset(PATH)
    # undersample by min class 
    print("min class:", y.value_counts(), y.value_counts().idxmin())
    print("initial label distribution:", y.value_counts())
    print()
    min_class_idx = y.value_counts().idxmin()
    num_samples_min_class = y.value_counts()[min_class_idx]
    print("num samples min class:", num_samples_min_class)

    Xy = X.copy()
    Xy["y"] = y
    dfs = []
    for lbl in sorted(y.unique()):
        idx = Xy[Xy["y"] == lbl]
        dfs.append(idx.sample(n=num_samples_min_class, random_state=42))
    Xy_bal = pd.concat(dfs).reset_index(drop=True)
    assert len(Xy_bal) == num_samples_min_class * len(y.unique())
                                        
    # # merge labels 0 and 1 
    # Xy_bal["y"] = Xy_bal["y"].apply(lambda x: 0 if x in [0, 1] else 1)

    # print("Balanced Dataset: ", Xy_bal.head())
    # print()
    # print("Labels: ", Xy_bal["y"].value_counts())
    # y_bal = Xy_bal["y"]
    # X_bal = Xy_bal.drop(columns=["y"])


    y = y.apply(lambda x: 0 if x in [0, 1] else 1) # merge conspiracy and 5g conspiracy
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2
    )

    model = train_model(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)
    print()
    print("Model performance:")
    # replace labels with class names
    performance_str = sklearn.metrics.classification_report(
        y_test, y_pred, target_names=["conspiracy", "non-conspiracy"] #["conspiracy", "5g conspiracy", "non-conspiracy"]
    )
    print(performance_str)

    # Feature importance
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")   
    for f in range(X.shape[1]):
        print(
            f"{f + 1:2d}) {X.columns[indices[f]]:<30} {importances[indices[f]]:f}"
        )

    scoring = "f1"
    perm_imp = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=0, scoring=scoring
    )
    importances = perm_imp.importances_mean
    threshold = np.percentile(importances, 20)  # drop bottom 20%
    feat_imp = pd.Series(importances, index=X_train.columns)
    feat_imp_sorted = feat_imp.sort_values(ascending=False)

    print()
    print("permutation importances:\n", feat_imp_sorted)
    print()
    print("threshold:", threshold)

    selected = X_train.columns[importances > threshold]
    X_train_sel, X_test_sel = X_train[selected], X_test[selected]

    model_sel = train_model(X_train_sel, y_train)
    y_pred_sel = model_sel.predict(X_test_sel)
    print()
    print(f"Model performance after feature selection (by {scoring} score):")
    performance_str = sklearn.metrics.classification_report(
        y_test, y_pred_sel, target_names=["conspiracy", "non-conspiracy"] #["conspiracy", "5g conspiracy", "non-conspiracy"]
    )
    print(performance_str)


    # save performance metrics into performance_logs dir
    performance_logs_dir = Path("performance_logs")
    performance_logs_dir.mkdir(exist_ok=True)
    model_name = "random_forest"
    model_version = "v1.2"
    with open(performance_logs_dir / f"{model_name}_{model_version}.txt", "w") as f:
        f.write(sklearn.metrics.classification_report(y_test, y_pred))
