import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from misinformation_graph_detection.graph_classifier import load_graphs
import kagglehub
from pathlib import Path
import networkx as nx

path = kagglehub.dataset_download("arashnic/misinfo-graph")
PATH = Path(path)





conspiracy_graphs, fiveg_conspiracy_graphs, non_conspiracy_graphs = load_graphs(PATH)


import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import networkx as nx
import random

# ----------- Create synthetic graph dataset -----------

def create_graph(G: nx.Graph, label: int) -> Data:
    # Convert to torch_geometric Data
    data = from_networkx(G)
    # data.x = torch.stack([G.nodes[n]['x'] for n in G.nodes()])
    data.y = torch.tensor([label], dtype=torch.long)
    return data

def create_graph(G: nx.Graph, label: int) -> Data:
    # Extract node features as a matrix: [num_nodes, 3]
    x = []
    for node_id in G.nodes():
        attr = G.nodes[node_id]
        x.append([attr["time"], attr["friends"], attr["followers"]])
    
    x = torch.tensor(x, dtype=torch.float)
    
    data = from_networkx(G)
    data.x = x
    data.y = torch.tensor([label], dtype=torch.long)
    return data


# Create dataset: 100 graphs (half label 0, half label 1)
conspiracy_graphs = [create_graph(G, 0) for G in conspiracy_graphs]
fiveg_conspiracy_graphs = [create_graph(G, 1) for G in fiveg_conspiracy_graphs]
non_conspiracy_graphs = [create_graph(G, 2) for G in non_conspiracy_graphs]

dataset = conspiracy_graphs + fiveg_conspiracy_graphs + non_conspiracy_graphs
random.shuffle(dataset)

# Train/test split
train_dataset = dataset[:200]
test_dataset = dataset[200:]

train_loader = DataLoader(train_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# ----------- Define GNN Model for graph classification -----------

class GCNGraphClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)        # ← changed from 1 to 3
        self.conv2 = GCNConv(16, 32)
        self.lin = torch.nn.Linear(32, 3)  # ← 3-class output

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ----------- Training loop -----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCNGraphClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()

def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)

for epoch in range(1, 1200):
    train()
    acc = test(test_loader)
    print(f"Epoch {epoch:02d}, Test Accuracy: {acc:.2%}")

