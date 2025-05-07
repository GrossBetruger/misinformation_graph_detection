# Misinformation Graph Detection

This project explores the detection of misinformation (e.g., conspiracy theories) by analyzing the structure and temporal dynamics of social diffusion graphs. We extract rich graph-level and community-level features from user interaction networks and train machine learning models to classify whether a given diffusion graph contains conspiratorial content or benign information.

## Project Overview
- Build social graphs from raw node and edge data.
- Compute structural, community, temporal, and centrality-based features.
- Train and evaluate classifiers (Random Forest, XGBoost) to distinguish between conspiracy and non-conspiracy cascades.
- Visualize diffusion graphs interactively with PyVis.

## Data Description
We leverage the Kaggle dataset `arashnic/misinfo-graph`, organized into three categories:
- **Other_Graphs**: general conspiracy diffusion graphs.
- **5G_Conspiracy_Graphs**: conspiracy graphs specifically about 5G topics.
- **Non_Conspiracy_Graphs**: diffusion graphs without conspiracy content.

Each graph folder contains:
- `nodes.csv`: node metadata with columns:
  - `id`: unique node identifier (int).
  - `time`: timestamp or relative time of the account/interaction (int).
  - `friends`: number of accounts the user follows (int).
  - `followers`: number of accounts following the user (int).
- `edges.txt`: undirected edge list (`u v` per line) representing social links or information propagation.

## Graph Construction
We represent each diffusion as a NetworkX graph, preserving node attributes:
```python
from misinformation_graph_detection.analyze import build_social_graph, PersonNode

# Prepare node and edge lists
nodes = [PersonNode(id, time, friends, followers), ...]
edges = [(u, v), ...]
# Build undirected graph with metadata
G = build_social_graph(nodes, edges)
```

## Feature Engineering
Features are extracted via `analyze_community_structure(G)`, returning a flat dictionary of graph‐level descriptors:

- **Structural Features**
  - `num_nodes`, `num_edges`
  - `graph_avg_degree`, `graph_density`, `graph_avg_clustering`
- **Community Features** (Louvain + greedy modularity)
  - `num_communities`, `community_modularity`
  - `community_avg_degree` (sum of degrees per community)
  - Sizes and connectivity of largest & second-largest communities
- **Temporal Features**
  - Statistics of `time` values: mean, median, min, max, std
  - Log and root transforms: `log_mean_time`, `sqrt_mean_time`
  - `time_entropy`: Shannon entropy of the time distribution
  - Percentiles (`t5`, `t10`, `t20`, `t50`, `t90`)
- **Cascade Shape Features**
  - `max_depth`, `median_depth`: path lengths from root (time == 0) in the largest connected component
  - `wiener_index`: average shortest-path length on the largest component
- **Centrality Features**
  - `avg_edge_betweenness_centrality`
  - Features for the highest-betweenness edge and its endpoints
- **Root Node Features**
  - `root_time`, `root_friends`, `root_followers`, `root_degree`, `root_betweenness`

These features capture how misinformation cascades differ from benign spreads in topology, community structure, and temporal dynamics.

## Modeling
Implemented in `misinformation_graph_detection/graph_classifier.py`:

- **Dataset Preparation**
  - `create_dataset(graphs_dir)` loads graphs, computes features, and returns `(X, y)` where `y` labels classes as:
    - `0`: conspiracy
    - `1`: 5G conspiracy
    - `2`: non-conspiracy
- **Training**
  - `train_model(X_train, y_train)` fits a `RandomForestClassifier` (configurable).
  - Optionally supports `XGBClassifier`.
- **Evaluation**
  - Train/test split with `sklearn.model_selection.train_test_split`
  - Classification reports (`precision`, `recall`, `f1-score`)
  - Feature importance ranking and permutation importance for selection
  - Saved reports in `performance_logs/`

## Visualization
Interactive graph visualizations via PyVis:
```python
from misinformation_graph_detection.analyze import plot_social_graph
plot_social_graph(G, title="Conspiracy Graph")  # saves HTML and opens browser
```
Node sizes reflect follower counts; colors denote Louvain communities.

## Installation & Usage
```bash
# Clone repository
git clone https://github.com/<username>/misinformation-graph-detection.git
cd misinformation-graph-detection

# Install dependencies (Poetry)
poetry install

# (Optional) Install system-wide
pip install -r requirements.txt
```

### Analyze & Visualize
```bash
python -m misinformation_graph_detection.analyze
```

### Train & Evaluate Classifier
```bash
python -m misinformation_graph_detection.graph_classifier
```

## Testing
Run the full test suite:
```bash
pytest
```

## Contributing
Contributions are welcome! Please open issues or PRs for new features, bug fixes, or improvements.

## License
This project does not include a license. Please contact the author for usage permissions.
