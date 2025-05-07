import random
import kagglehub
import networkx as nx
import numpy as np
import pandas as pd
import os
import plotly.express as px  # Fix: Import plotly.express

from pathlib import Path
from typing import Any, Counter, Dict, Iterable, Hashable, Tuple, List
from dataclasses import dataclass
from pyvis.network import Network
from tqdm import tqdm


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


def calc_time_entropy(times: List[int], n_bins: int = 20, base: int = 2) -> float:
    """
    Calculate the time entropy of a list of times.
    """
    # histogram
    counts, _ = np.histogram(times, bins=n_bins)
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log(p) / np.log(base))


def average_shortest_path_length_lcc(G: nx.Graph) -> float:
    """
    Compute the avg. shortest-path length on the Largest Connected Component of G.
    For directed graphs, uses weak connectivity.
    """
    # pick the right kind of components
    if G.is_directed():
        comps = nx.weakly_connected_components(G)
    else:
        comps = nx.connected_components(G)
    # find the largest one
    largest = max(comps, key=len)
    # induce the subgraph and compute
    H = G.subgraph(largest).copy()
    return nx.average_shortest_path_length(H)


def get_lcc(G: nx.Graph) -> nx.Graph:
    """
    Get the largest connected component of a graph.
    """
    return G.subgraph(max(nx.connected_components(G), key=len))


def analyze_community_structure(G: nx.Graph) -> Dict[str, Any]:
    """
    Analyze the community structure of a graph.
    """

    if G.number_of_edges() == 0:
        return {
            "num_communities": G.number_of_nodes(),
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "community_avg_degree": 0,
            "community_modularity": -1,
        }

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
    graph_avg_degree = np.mean([G.degree(n) for n in G.nodes()])
    graph_density = nx.density(G)
    graph_avg_clustering = nx.average_clustering(G)
    edge_betweenness_centrality = nx.edge_betweenness_centrality(G)
    avg_edge_betweenness_centrality = np.mean(
        list(edge_betweenness_centrality.values())
    )

    community_degrees = {
        c: sum(G.degree(n) for n in community)
        for c, community in enumerate(communities)
    }

    # LCC max depth, median depth: Conspiracy cascades often grow deeper even when small
    comps = nx.connected_components(G)
    # largest one
    largest = max(comps, key=len)
    # induce the subgraph and compute
    lcc = get_lcc(G)  # G.subgraph(largest).copy() # Largest Connected Component
    root = [node for node in G.nodes() if G.nodes[node]["time"] == 0][0]
    if root not in lcc:
        largest = max(
            [comp for comp in nx.connected_components(G) if root in comp], key=len
        )
        lcc = G.subgraph(largest).copy()
    depths = [nx.shortest_path_length(lcc, source=root, target=v) for v in lcc]
    max_depth = max(depths)
    median_depth = np.median(depths)
    # root node features
    root_time = G.nodes[root]["time"]
    root_friends = G.nodes[root]["friends"]
    root_followers = G.nodes[root]["followers"]
    root_degree = G.degree(root)
    if root in edge_betweenness_centrality:
        root_betweenness = edge_betweenness_centrality[root]
    else:
        root_betweenness = 0

    num_degree_0_nodes = sum(1 for n in G.nodes() if G.degree(n) == 0)
    num_degree_1_nodes = sum(1 for n in G.nodes() if G.degree(n) == 1)
    num_degree_2_nodes = sum(1 for n in G.nodes() if G.degree(n) == 2)
    ratio_degree_0_nodes = num_degree_0_nodes / G.number_of_nodes()
    ratio_degree_1_nodes = num_degree_1_nodes / G.number_of_nodes()
    ratio_degree_2_nodes = num_degree_2_nodes / G.number_of_nodes()

    largest_community_id, largest_community_size = Counter(
        comm_map.values()
    ).most_common(1)[0]
    largest_community_nodes = [
        node for node in G.nodes() if comm_map[node] == largest_community_id
    ]

    # Detect communities using greedy modularity maximization
    greedy_communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    wiener_index = average_shortest_path_length_lcc(
        G
    )  # Separates “star-shaped broadcast” from deep relay chains, a strong rumour signal (wiener_index)

    # Compute modularity score
    mod_value = nx.algorithms.community.modularity(G, greedy_communities)

    avg_num_friends = np.log2(
        np.mean([G.nodes[n]["friends"] for n in G.nodes()])
    )  # 4 → 2**4 = 16–31 real friends (dataset definition buckets of powers of 2)
    avg_num_followers = np.log2(
        np.mean([G.nodes[n]["followers"] for n in G.nodes()])
    )  # 4 → 2**4 = 16–31 real followers (dataset definition buckets of powers of 2)
    mean_time = np.mean([G.nodes[n]["time"] for n in G.nodes()])
    assert mean_time > 0, f"Mean time is not greater than 0, {mean_time}"
    log_mean_time = np.log(mean_time)
    sqrt_mean_time = np.sqrt(mean_time)
    median_time = np.median([G.nodes[n]["time"] for n in G.nodes()])
    max_time = np.max([G.nodes[n]["time"] for n in G.nodes()])
    min_time = np.min([G.nodes[n]["time"] for n in G.nodes()])
    std_time = np.std([G.nodes[n]["time"] for n in G.nodes()])
    times = [G.nodes[n]["time"] for n in G.nodes()]
    time_entropy = calc_time_entropy(times)
    t5 = np.percentile(times, 5)
    t10 = np.percentile(times, 10)
    t20 = np.percentile(times, 20)
    t90 = np.percentile(times, 90)
    t50 = np.percentile(times, 50)

    # largest community features
    largest_community_avg_num_friends = np.log2(
        np.mean([G.nodes[n]["friends"] for n in largest_community_nodes])
    )
    largest_community_avg_num_followers = np.log2(
        np.mean([G.nodes[n]["followers"] for n in largest_community_nodes])
    )
    largest_community_mean_time = np.mean(
        [G.nodes[n]["time"] for n in largest_community_nodes]
    )
    largest_community_median_time = np.median(
        [G.nodes[n]["time"] for n in largest_community_nodes]
    )
    largest_community_max_time = np.max(
        [G.nodes[n]["time"] for n in largest_community_nodes]
    )
    largest_community_min_time = np.min(
        [G.nodes[n]["time"] for n in largest_community_nodes]
    )
    largest_community_std_time = np.std(
        [G.nodes[n]["time"] for n in largest_community_nodes]
    )
    largest_community_avg_degree = np.mean(
        [G.degree(n) for n in largest_community_nodes]
    )
    largest_community_density = nx.density(G.subgraph(largest_community_nodes))
    largest_community_avg_clustering = nx.average_clustering(
        G.subgraph(largest_community_nodes)
    )
    largest_community_time_entropy = calc_time_entropy(
        [G.nodes[n]["time"] for n in largest_community_nodes]
    )
    largest_community_times = [G.nodes[n]["time"] for n in largest_community_nodes]
    largest_community_t5 = np.percentile(largest_community_times, 5)
    largest_community_t10 = np.percentile(largest_community_times, 10)
    largest_community_t20 = np.percentile(largest_community_times, 20)
    largest_community_t90 = np.percentile(largest_community_times, 90)
    largest_community_t50 = np.percentile(largest_community_times, 50)
    largest_community_wiener_index = average_shortest_path_length_lcc(
        G.subgraph(largest_community_nodes)
    )

    # second largest community features
    if len(Counter(comm_map.values()).most_common(2)) > 1:
        second_largest_community_id, second_largest_community_size = Counter(
            comm_map.values()
        ).most_common(2)[1]
    else:
        second_largest_community_id, second_largest_community_size = (
            largest_community_id,
            largest_community_size,
        )
    second_largest_community_nodes = [
        node for node in G.nodes() if comm_map[node] == second_largest_community_id
    ]
    second_largest_community_avg_num_friends = np.mean(
        [G.nodes[n]["friends"] for n in second_largest_community_nodes]
    )
    second_largest_community_avg_num_followers = np.mean(
        [G.nodes[n]["followers"] for n in second_largest_community_nodes]
    )
    second_largest_community_mean_time = np.mean(
        [G.nodes[n]["time"] for n in second_largest_community_nodes]
    )
    second_largest_community_median_time = np.median(
        [G.nodes[n]["time"] for n in second_largest_community_nodes]
    )
    second_largest_community_max_time = np.max(
        [G.nodes[n]["time"] for n in second_largest_community_nodes]
    )
    second_largest_community_min_time = np.min(
        [G.nodes[n]["time"] for n in second_largest_community_nodes]
    )
    second_largest_community_std_time = np.std(
        [G.nodes[n]["time"] for n in second_largest_community_nodes]
    )
    second_largest_community_avg_degree = np.mean(
        [G.degree(n) for n in second_largest_community_nodes]
    )
    second_largest_community_density = nx.density(
        G.subgraph(second_largest_community_nodes)
    )
    second_largest_community_avg_clustering = nx.average_clustering(
        G.subgraph(second_largest_community_nodes)
    )
    second_largest_community_time_entropy = calc_time_entropy(
        [G.nodes[n]["time"] for n in second_largest_community_nodes]
    )
    second_largest_community_times = [
        G.nodes[n]["time"] for n in second_largest_community_nodes
    ]
    second_largest_community_t5 = np.percentile(second_largest_community_times, 5)
    second_largest_community_t10 = np.percentile(second_largest_community_times, 10)
    second_largest_community_t20 = np.percentile(second_largest_community_times, 20)
    second_largest_community_t90 = np.percentile(second_largest_community_times, 90)
    second_largest_community_t50 = np.percentile(second_largest_community_times, 50)
    second_largest_community_wiener_index = average_shortest_path_length_lcc(
        G.subgraph(second_largest_community_nodes)
    )

    # for n in G.nodes():
    #     features = G.nodes[n]
    #     print(features)

    highest_betweenness_edge = max(
        edge_betweenness_centrality, key=edge_betweenness_centrality.get
    )
    highest_betweenness_node1, highest_betweenness_node2 = highest_betweenness_edge
    highest_betweenness = edge_betweenness_centrality[
        highest_betweenness_node1, highest_betweenness_node2
    ]
    # higest betweeness node features
    highest_betweenness_node1_time = G.nodes[highest_betweenness_node1]["time"]
    highest_betweenness_node1_friends = G.nodes[highest_betweenness_node1]["friends"]
    highest_betweenness_node1_followers = G.nodes[highest_betweenness_node1][
        "followers"
    ]
    highest_betweenness_node1_degree = G.degree(highest_betweenness_node1)
    # highest betweeness node 1 lcc features
    largest_comp_node1 = max(
        [
            comp
            for comp in nx.connected_components(G)
            if highest_betweenness_node1 in comp
        ],
        key=len,
    )
    lcc_node1 = G.subgraph(largest_comp_node1).copy()
    depths_node1 = [
        nx.shortest_path_length(lcc_node1, source=highest_betweenness_node1, target=v)
        for v in lcc_node1
    ]
    max_depth_node1 = max(depths_node1)
    median_depth_node1 = np.median(depths_node1)

    highest_betweenness_node2_time = G.nodes[highest_betweenness_node2]["time"]
    highest_betweenness_node2_friends = G.nodes[highest_betweenness_node2]["friends"]
    highest_betweenness_node2_followers = G.nodes[highest_betweenness_node2][
        "followers"
    ]
    highest_betweenness_node2_degree = G.degree(highest_betweenness_node2)
    follower_diff_node2_root = (
        G.nodes[highest_betweenness_node2]["followers"] - G.nodes[root]["followers"]
    )
    largest_comp_node2 = max(
        [
            comp
            for comp in nx.connected_components(G)
            if highest_betweenness_node2 in comp
        ],
        key=len,
    )
    lcc_node2 = G.subgraph(largest_comp_node2).copy()
    depths_node2 = [
        nx.shortest_path_length(lcc_node2, source=highest_betweenness_node2, target=v)
        for v in lcc_node2
    ]
    max_depth_node2 = max(depths_node2)
    median_depth_node2 = np.median(depths_node2)

    graph_features = {
        "num_communities": num_communities,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "graph_avg_degree": graph_avg_degree,
        "graph_density": graph_density,
        "graph_avg_clustering": graph_avg_clustering,
        "max_depth": max_depth,
        "median_depth": median_depth,
        "wiener_index": wiener_index,
        "avg_edge_betweenness_centrality": avg_edge_betweenness_centrality,
        "community_avg_degree": np.mean(list(community_degrees.values())),
        "community_modularity": mod_value,
        "avg_num_friends": avg_num_friends,
        "avg_num_followers": avg_num_followers,
        "num_degree_0_nodes": num_degree_0_nodes,
        "num_degree_1_nodes": num_degree_1_nodes,
        "num_degree_2_nodes": num_degree_2_nodes,
        "ratio_degree_0_nodes": ratio_degree_0_nodes,
        "ratio_degree_1_nodes": ratio_degree_1_nodes,
        "ratio_degree_2_nodes": ratio_degree_2_nodes,
        "mean_time": mean_time,
        "log_mean_time": log_mean_time,
        "sqrt_mean_time": sqrt_mean_time,
        "median_time": median_time,
        "max_time": max_time,
        "min_time": min_time,
        "std_time": std_time,
        "time_entropy": time_entropy,
        "t5": t5,
        "t10": t10,
        "t20": t20,
        "t90": t90,
        "t50": t50,
        "root_time": root_time,
        "root_friends": root_friends,
        "root_followers": root_followers,
        "root_degree": root_degree,
        "root_betweenness": root_betweenness,
        "largest_community_avg_num_friends": largest_community_avg_num_friends,
        "largest_community_avg_num_followers": largest_community_avg_num_followers,
        "largest_community_mean_time": largest_community_mean_time,
        "largest_community_max_time": largest_community_max_time,
        "largest_community_min_time": largest_community_min_time,
        "largest_community_size": largest_community_size,
        "largest_community_wiener_index": largest_community_wiener_index,
        "largest_community_std_time": largest_community_std_time,
        "largest_community_avg_degree": largest_community_avg_degree,
        "largest_community_density": largest_community_density,
        "largest_community_avg_clustering": largest_community_avg_clustering,
        "largest_community_median_time": largest_community_median_time,
        "largest_community_time_entropy": largest_community_time_entropy,
        "second_largest_community_size": second_largest_community_size,
        "second_largest_community_avg_num_friends": second_largest_community_avg_num_friends,
        "second_largest_community_avg_num_followers": second_largest_community_avg_num_followers,
        "second_largest_community_mean_time": second_largest_community_mean_time,
        "second_largest_community_median_time": second_largest_community_median_time,
        "second_largest_community_max_time": second_largest_community_max_time,
        "second_largest_community_min_time": second_largest_community_min_time,
        "second_largest_community_std_time": second_largest_community_std_time,
        "second_largest_community_avg_degree": second_largest_community_avg_degree,
        "second_largest_community_density": second_largest_community_density,
        "second_largest_community_avg_clustering": second_largest_community_avg_clustering,
        "second_largest_community_time_entropy": second_largest_community_time_entropy,
        "second_largest_community_wiener_index": second_largest_community_wiener_index,
        "second_largest_community_t5": second_largest_community_t5,
        "second_largest_community_t10": second_largest_community_t10,
        "second_largest_community_t20": second_largest_community_t20,
        "second_largest_community_t90": second_largest_community_t90,
        "second_largest_community_t50": second_largest_community_t50,
        "highest_betweenness": highest_betweenness,
        "highest_betweenness_node1_time": highest_betweenness_node1_time,
        "highest_betweenness_node1_friends": highest_betweenness_node1_friends,
        "highest_betweenness_node1_followers": highest_betweenness_node1_followers,
        "highest_betweenness_node1_degree": highest_betweenness_node1_degree,
        "highest_betweenness_node1_max_depth": max_depth_node1,
        "highest_betweenness_node1_median_depth": median_depth_node1,
        "highest_betweenness_node2_time": highest_betweenness_node2_time,
        "highest_betweenness_node2_friends": highest_betweenness_node2_friends,
        "highest_betweenness_node2_followers": highest_betweenness_node2_followers,
        "highest_betweenness_node2_degree": highest_betweenness_node2_degree,
        "highest_betweenness_node2_max_depth": max_depth_node2,
        "highest_betweenness_node2_median_depth": median_depth_node2,
        "follower_diff_node2_root": follower_diff_node2_root,
    }
    # find infinite values
    for key, value in graph_features.items():
        if isinstance(value, float) and np.isinf(value):
            raise ValueError(f"Infinite value found in {key}: {value}")
    return graph_features


def load_graphs_from_dir(graphs_dir: Path) -> List[nx.Graph]:
    """
    Load all graphs from a directory of subfolders, each containing 'edges.txt' and 'nodes.csv'.
    Returns a list of populated NetworkX graphs.
    """
    graphs: List[nx.Graph] = []
    print(f"Loading graphs from: {graphs_dir}")
    for graph_dir in tqdm(
        graphs_dir.iterdir()
    ):  # , desc=f"Loading graphs from: {graphs_dir}"):
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

    # for con_graph in conspiracy_graphs[:10]:
    #     con_times = [node[1]["time"] for node in con_graph.nodes(data=True)]
    #     px.histogram(con_times, nbins=100, title="Conspiracy Time Histogram").show()

    # for non_con_graph in non_conspiracy_graphs[:10]:
    #     non_con_times = [node[1]["time"] for node in non_con_graph.nodes(data=True)]
    #     px.histogram(non_con_times, nbins=100, title="Non-Conspiracy Time Histogram").show()

    # for fiveg_con_graph in fiveg_conspiracy_graphs[:10]:
    #     fiveg_con_times = [node[1]["time"] for node in fiveg_con_graph.nodes(data=True)]
    #     px.histogram(fiveg_con_times, nbins=100, title="5G Conspiracy Time Histogram").show()

    plot_social_graph(con_graph, "Conspiracy Graph")
    plot_social_graph(fiveg_con_graph, "5G Conspiracy Graph")
    plot_social_graph(non_con_graph, "Non-Conspiracy Graph")

    print("Conspiracy Graph: ", analyze_community_structure(con_graph))
    print("5G Conspiracy Graph: ", analyze_community_structure(fiveg_con_graph))
    print("Non-Conspiracy Graph: ", analyze_community_structure(non_con_graph))
