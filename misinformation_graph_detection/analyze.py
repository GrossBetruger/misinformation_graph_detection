import kagglehub
import networkx as nx
import pandas as pd
import os

from pathlib import Path
from typing import Iterable, Hashable, Tuple
from dataclasses import dataclass
from pyvis.network import Network


@dataclass(frozen=True)
class PersonNode:
    """A lightweight container for a user in the social-graph."""
    id: str          # unique, hashable key
    time: int   # when the account was created (or any timestamp you track)
    friends: int     # “following” count
    followers: int   # follower count


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

# Download latest version
path = kagglehub.dataset_download("arashnic/misinfo-graph")

# print("Path to dataset files:", path)

f: Path
for f in Path(path).iterdir():
    if f.is_dir() and str(f).endswith("Other_Graphs"):
        conspiracy_graphs_dir = f
    if f.is_dir() and str(f).endswith("Non_Conspiracy_Graphs"):
        non_conspiracy_graphs_dir = f

conspiracy_graphs = []
for graph_dir in conspiracy_graphs_dir.iterdir():
    edges_file = graph_dir / "edges.txt"
    nodes_file = graph_dir / "nodes.csv"
    assert edges_file.exists(), f"no edges: {graph_dir}"
    assert nodes_file.exists(), f"no nodes: {graph_dir}"
    edges = []
    with open(edges_file, "r") as f:
        for line in f:
            raw_edge = line.strip().split(" ")
            edge = (int(raw_edge[0]), int(raw_edge[1]))
            edges.append(edge)

    nodes = pd.read_csv(nodes_file)
    nodes = [PersonNode(id=int(row["id"]), time=int(row["time"]), friends=int(row["friends"]), followers=int(row["followers"])) for _index, row in nodes.iterrows()]
    conspiracy_graph = build_social_graph(nodes, edges)
    conspiracy_graphs.append(conspiracy_graph)

con_graph = conspiracy_graphs[0]
print(con_graph.nodes(data=True))
net = Network(height="750px", bgcolor="#222", font_color="white")
net.from_nx(con_graph)
net.show("graph.html", notebook=False)
os.system("open graph.html")
