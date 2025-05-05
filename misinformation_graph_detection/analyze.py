import random
import kagglehub
import networkx as nx
import pandas as pd
import os

from pathlib import Path
from typing import Iterable, Hashable, Tuple, List
from dataclasses import dataclass
from pyvis.network import Network


@dataclass(frozen=True)
class PersonNode:
    """A lightweight container for a user in the social-graph."""

    id: str  # unique, hashable key
    time: int  # when the account was created (or any timestamp you track)
    friends: int  # “following” count
    followers: int  # follower count


def build_social_graph(
    nodes: Iterable[PersonNode],
    edges: Iterable[Tuple[str, str]],
) -> nx.Graph:
    """
    Construct an undirected NetworkX graph whose nodes carry rich metadata.

    Parameters
    ----------
    nodes : Iterable[PersonNode]
        A collection of users.  Their ``id`` field becomes the node key and
        the remaining fields become node attributes.
    edges : Iterable[Tuple[str, str]]
        Pairs of node IDs defining undirected friendship edges.

    Returns
    -------
    nx.Graph
        A populated graph ready for analytics.
    """
    G: nx.Graph = nx.Graph()

    # unpack each dataclass into (node_key, attr_dict) 2-tuples
    G.add_nodes_from(
        (p.id, {"time": p.time, "friends": p.friends, "followers": p.followers})
        for p in nodes
    )

    G.add_edges_from(edges)
    return G


def load_graphs_from_dir(graphs_dir: Path) -> List[nx.Graph]:
    """
    Load all graphs from a directory of subfolders, each containing 'edges.txt' and 'nodes.csv'.
    Returns a list of populated NetworkX graphs.
    """
    graphs: List[nx.Graph] = []
    for graph_dir in graphs_dir.iterdir():
        edges_file = graph_dir / "edges.txt"
        nodes_file = graph_dir / "nodes.csv"
        assert edges_file.exists(), f"no edges: {graph_dir}"
        assert nodes_file.exists(), f"no nodes: {graph_dir}"

        edges: List[Tuple[int, int]] = []
        with edges_file.open("r") as ef:
            for line in ef:
                u, v = line.strip().split()
                edges.append((int(u), int(v)))

        df = pd.read_csv(nodes_file)
        nodes: List[PersonNode] = [
            PersonNode(
                id=int(row["id"]),
                time=int(row["time"]),
                friends=int(row["friends"]),
                followers=int(row["followers"]),
            )
            for _, row in df.iterrows()
        ]

        graphs.append(build_social_graph(nodes, edges))
    return graphs


if __name__ == "__main__":

    # Download latest version
    path = kagglehub.dataset_download("arashnic/misinfo-graph")

    f: Path
    for f in Path(path).iterdir():
        if f.is_dir() and str(f).endswith("Other_Graphs"):
            conspiracy_graphs_dir = f
        if f.is_dir() and str(f).endswith("Non_Conspiracy_Graphs"):
            non_conspiracy_graphs_dir = f

    conspiracy_graphs = load_graphs_from_dir(conspiracy_graphs_dir)
    non_conspiracy_graphs = load_graphs_from_dir(non_conspiracy_graphs_dir)

    random_index = random.randint(0, len(conspiracy_graphs) - 1)
    con_graph = conspiracy_graphs[random_index]
    net = Network(height="750px", bgcolor="#222", font_color="white")
    net.from_nx(con_graph)
    net.show("conspiracy_graph.html", notebook=False)
    os.system("open graph.html")

    random_index = random.randint(0, len(non_conspiracy_graphs) - 1)
    non_con_graph = non_conspiracy_graphs[random_index]
    net = Network(height="750px", bgcolor="#222", font_color="white")
    net.from_nx(non_con_graph)
    net.show("non_conspiracy_graph.html", notebook=False)
    os.system("open non_conspiracy_graph.html")
