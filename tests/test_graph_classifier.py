import pandas as pd
import pytest
from pathlib import Path

from misinformation_graph_detection.graph_classifier import (
    load_graphs,
    create_dataset,
    train_model,
)


def write_graph(
    base_dir: Path, category: str, name: str, nodes: list[dict], edges: list[tuple]
):
    dir_path = base_dir / category / name
    dir_path.mkdir(parents=True)
    edges_path = dir_path / "edges.txt"
    if edges:
        edges_path.write_text("\n".join(f"{u} {v}" for u, v in edges) + "\n")
    else:
        # Empty graph
        edges_path.write_text("")
    nodes_df = pd.DataFrame(nodes)
    nodes_df.to_csv(dir_path / "nodes.csv", index=False)


@pytest.fixture
def sample_graphs_dir(tmp_path):
    base = tmp_path / "graphs"
    # Create category directories
    categories = ["Other_Graphs", "5G_Conspiracy_Graphs", "Non_Conspiracy_Graphs"]
    for cat in categories:
        (base / cat).mkdir(parents=True)
    # Other_Graphs: simple two-node graph
    nodes_a = [
        {"id": 1, "time": 0, "friends": 5, "followers": 2},  # root
        {"id": 2, "time": 20, "friends": 3, "followers": 4},
    ]
    edges_a = [(1, 2)]
    write_graph(base, "Other_Graphs", "graphA", nodes_a, edges_a)
    # 5G_Conspiracy_Graphs: three-node chain
    nodes_b = [
        {"id": 3, "time": 0, "friends": 7, "followers": 6},  # root
        {"id": 4, "time": 40, "friends": 8, "followers": 5},
        {"id": 5, "time": 50, "friends": 9, "followers": 4},
    ]
    edges_b = [(3, 4), (4, 5)]
    write_graph(base, "5G_Conspiracy_Graphs", "graphB", nodes_b, edges_b)
    # Non_Conspiracy_Graphs: single node, no edges
    nodes_c = [{"id": 6, "time": 0, "friends": 10, "followers": 8}]  # root
    edges_c = []
    write_graph(base, "Non_Conspiracy_Graphs", "graphC", nodes_c, edges_c)
    return base


def test_load_graphs(sample_graphs_dir):
    conspiracy_graphs, fiveg_graphs, non_graphs = load_graphs(sample_graphs_dir)
    # Lists of graphs for each category
    assert isinstance(conspiracy_graphs, list)
    assert isinstance(fiveg_graphs, list)
    assert isinstance(non_graphs, list)
    assert len(conspiracy_graphs) == 1
    assert len(fiveg_graphs) == 1
    assert len(non_graphs) == 1

    # Validate Other_Graphs graph
    G1 = conspiracy_graphs[0]
    assert set(G1.nodes()) == {1, 2}
    assert set(G1.edges()) == {(1, 2)}
    for nid, data in G1.nodes(data=True):
        assert isinstance(data.get("time"), int)
        assert isinstance(data.get("friends"), int)
        assert isinstance(data.get("followers"), int)

    # Validate 5G_Conspiracy_Graphs graph
    G2 = fiveg_graphs[0]
    assert set(G2.nodes()) == {3, 4, 5}
    assert set(G2.edges()) == {(3, 4), (4, 5)}

    # Validate Non_Conspiracy_Graphs graph
    G3 = non_graphs[0]
    assert set(G3.nodes()) == {6}
    assert set(G3.edges()) == set()


def test_create_dataset(sample_graphs_dir):
    features, labels = create_dataset(sample_graphs_dir)
    # DataFrame of features and Series of labels
    assert isinstance(features, pd.DataFrame)
    assert isinstance(labels, pd.Series)

    expected_cols = {
        "num_communities",
        "num_nodes",
        "num_edges",
        "community_avg_degree",
        "community_modularity",
        "avg_num_friends",
        "avg_num_followers",
    }
    assert expected_cols.issubset(set(features.columns))
    # Labels should be [0,1,2]
    assert labels.tolist() == [0, 1, 2]
    # Check feature values
    assert features["num_nodes"].tolist() == [2, 3, 1]
    assert features["num_edges"].tolist() == [1, 2, 0]
    assert features["num_communities"].tolist() == [1, 1, 1]
    # Average degree per community
    assert pytest.approx(features["community_avg_degree"].tolist()) == [2, 4, 0]


def test_train_model(sample_graphs_dir):
    features, labels = create_dataset(sample_graphs_dir)
    # Train with DataFrame and Series
    model = train_model(features, labels)
    from sklearn.ensemble import RandomForestClassifier

    assert isinstance(model, RandomForestClassifier)
    # classes_ should match unique labels
    assert list(model.classes_) == sorted(labels.unique().tolist())
    # Train with list inputs
    model2 = train_model(features.values.tolist(), labels.tolist())
    assert isinstance(model2, RandomForestClassifier)
    assert list(model2.classes_) == sorted(labels.unique().tolist())
