import pandas as pd
import pytest

from pathlib import Path
import networkx as nx

from misinformation_graph_detection.analyze import (
    build_social_graph,
    load_graphs_from_dir,
    PersonNode,
)


def test_build_social_graph_basic():
    # Create a small social graph with two nodes and one edge
    nodes = [
        PersonNode(id="1", time=10, friends=5, followers=15),
        PersonNode(id="2", time=20, friends=3, followers=8),
    ]
    edges = [("1", "2")]
    G = build_social_graph(nodes, edges)
    # Verify nodes and attributes
    assert set(G.nodes()) == {"1", "2"}
    for nid, expected in [("1", (10, 5, 15)), ("2", (20, 3, 8))]:
        data = G.nodes[nid]
        assert data["time"] == expected[0]
        assert data["friends"] == expected[1]
        assert data["followers"] == expected[2]
    # Verify edge
    assert G.has_edge("1", "2")
    assert not G.has_edge("2", "3")


def test_load_graphs_from_dir(tmp_path):
    # Prepare two graph directories under tmp_path
    # Graph A: nodes 1-2-3
    ga = tmp_path / "graphA"
    ga.mkdir()
    (ga / "edges.txt").write_text("1 2\n2 3\n")
    df_a = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "time": [0, 1, 2],
            "friends": [10, 20, 30],
            "followers": [5, 6, 7],
        }
    )
    df_a.to_csv(ga / "nodes.csv", index=False)

    # Graph B: nodes 4-5
    gb = tmp_path / "graphB"
    gb.mkdir()
    (gb / "edges.txt").write_text("4 5\n")
    df_b = pd.DataFrame(
        {
            "id": [4, 5],
            "time": [3, 4],
            "friends": [40, 50],
            "followers": [8, 9],
        }
    )
    df_b.to_csv(gb / "nodes.csv", index=False)

    graphs = load_graphs_from_dir(tmp_path)
    # Expect exactly two graphs loaded
    assert len(graphs) == 2
    # Check each graph has correct structure
    edge_sets = {frozenset(g.edges()) for g in graphs}
    expected = {frozenset({(1, 2), (2, 3)}), frozenset({(4, 5)})}
    assert edge_sets == expected
    # Ensure node data present and types are correct
    for g in graphs:
        for nid, data in g.nodes(data=True):
            assert isinstance(nid, (int, str))
            assert isinstance(data.get("time"), int)
            assert isinstance(data.get("friends"), int)
            assert isinstance(data.get("followers"), int)


def test_load_graphs_from_dir_missing_files(tmp_path):
    # Directory with edges.txt but missing nodes.csv should raise AssertionError
    bad = tmp_path / "bad_graph"
    bad.mkdir()
    (bad / "edges.txt").write_text("1 2\n")
    with pytest.raises(AssertionError):
        load_graphs_from_dir(tmp_path)
