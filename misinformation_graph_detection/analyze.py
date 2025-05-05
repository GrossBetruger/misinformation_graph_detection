import random
import kagglehub
import networkx as nx
import numpy as np
import pandas as pd
import os

from pathlib import Path
from typing import Any, Dict, Iterable, Hashable, Tuple, List
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


def plot_social_graph(G: nx.Graph, title: str = "Social Graph") -> None:
    """
    Interactive social graph visualization using pyvis:
      • size    : proportional to #followers
      • colour  : Louvain community assignment (grouping)
    """
    # 1) Community detection for grouping
    communities = nx.algorithms.community.louvain_communities(G, seed=0)
    comm_map: Dict[Hashable, int] = {}
    for idx, community in enumerate(communities):
        for node in community:
            comm_map[node] = idx

    # 2) Initialize pyvis Network
    net = Network(height="750px", width="100%", bgcolor="#222", font_color="white")

    # 3) Add nodes with properties
    for node, data in G.nodes(data=True):
        followers = data.get("followers", 0)
        friends = data.get("friends", 0)
        time = data.get("time", 0)
        group = comm_map.get(node, 0)
        title_html = f"ID: {node}<br>Followers: {followers}<br>Friends: {friends}<br>Time: {time}"
        net.add_node(
            str(node),
            label=str(node),
            title=title_html,
            value=followers,
            group=group,
        )

    # 4) Add edges
    for src, dst in G.edges():
        net.add_edge(str(src), str(dst))

    # 5) Save to HTML and open
    filename = f"{title.replace(' ', '_').lower()}.html"
    # Write HTML file without notebook template
    net.write_html(filename, open_browser=False, notebook=False)
    os.system(f"open {filename}")


def analyze_community_structure(G: nx.Graph) -> Dict[str, Any]:
    """
    Analyze the community structure of a graph.
    """
    communities: List[List[Hashable]] = nx.algorithms.community.louvain_communities(
        G, seed=0
    )
    comm_map: Dict[Hashable, int] = {}
    for idx, community in enumerate(communities):
        for node in community:
            comm_map[node] = idx

    num_communities = len(communities)
    num_nodes = len(G.nodes())
    num_edges = len(G.edges())
    community_degrees = {
        c: sum(G.degree(n) for n in community)
        for c, community in enumerate(communities)
    }

    graph_features = {
        "num_communities": num_communities,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "community_avg_degree": np.mean(list(community_degrees.values())),
    }
    return graph_features


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
    print(path)

    f: Path
    for f in Path(path).iterdir():
        if f.is_dir() and str(f).endswith("Other_Graphs"):
            conspiracy_graphs_dir = f
        if f.is_dir() and str(f).endswith("Non_Conspiracy_Graphs"):
            non_conspiracy_graphs_dir = f
        if f.is_dir() and str(f).endswith("5G_Conspiracy_Graphs"):
            fiveg_conspiracy_graphs_dir = f

    fiveg_conspiracy_graphs = load_graphs_from_dir(fiveg_conspiracy_graphs_dir)
    conspiracy_graphs = load_graphs_from_dir(conspiracy_graphs_dir)
    non_conspiracy_graphs = load_graphs_from_dir(non_conspiracy_graphs_dir)

    random_index = random.randint(0, len(conspiracy_graphs) - 1)
    con_graph = conspiracy_graphs[random_index]

    random_index = random.randint(0, len(fiveg_conspiracy_graphs) - 1)
    fiveg_con_graph = fiveg_conspiracy_graphs[random_index]

    random_index = random.randint(0, len(non_conspiracy_graphs) - 1)
    non_con_graph = non_conspiracy_graphs[random_index]

    plot_social_graph(con_graph, "Conspiracy Graph")
    plot_social_graph(fiveg_con_graph, "5G Conspiracy Graph")
    plot_social_graph(non_con_graph, "Non-Conspiracy Graph")

    print("Conspiracy Graph: ", analyze_community_structure(con_graph))
    print("5G Conspiracy Graph: ", analyze_community_structure(fiveg_con_graph))
    print("Non-Conspiracy Graph: ", analyze_community_structure(non_con_graph))
