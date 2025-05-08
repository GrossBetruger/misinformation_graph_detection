import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from misinformation_graph_detection.graph_classifier import load_graphs
import kagglehub
from pathlib import Path
import networkx as nx
from torch_geometric.nn import BatchNorm, GraphNorm, global_add_pool
from torch_geometric.utils import dropout_edge

from sklearn.preprocessing import StandardScaler
from torch.nn import Linear, Dropout

path = kagglehub.dataset_download("arashnic/misinfo-graph")
PATH = Path(path)





conspiracy_graphs, fiveg_conspiracy_graphs, non_conspiracy_graphs = load_graphs(PATH)


import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
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
    # collect raw features
    x = [[G.nodes[n][k] for k in ("time","friends","followers")] for n in G.nodes()]
    vals = torch.tensor(x, dtype=torch.float)       # shape [N,3]
    # per‑graph normalization to zero‑mean/unit‑std
    # vals = (vals - vals.mean(0)) / (vals.std(0) + 1e-6)
    data = from_networkx(G)
    data.x = vals
    scaler = StandardScaler()
    data.x = torch.tensor(scaler.fit_transform(data.x), dtype=torch.float)

    data.y = torch.tensor(label, dtype=torch.long)  # scalar, NOT [label]
    return data



mean_conspiracy_graph_size = sum([G.number_of_edges() for G in conspiracy_graphs]) / len(conspiracy_graphs)
mean_fiveg_conspiracy_graph_size = sum([G.number_of_edges() for G in fiveg_conspiracy_graphs]) / len(fiveg_conspiracy_graphs)
mean_non_conspiracy_graph_size = sum([G.number_of_edges() for G in non_conspiracy_graphs]) / len(non_conspiracy_graphs)

print(f"Mean conspiracy graph size: {mean_conspiracy_graph_size}")
print(f"Mean fiveg conspiracy graph size: {mean_fiveg_conspiracy_graph_size}")
print(f"Mean non conspiracy graph size: {mean_non_conspiracy_graph_size}")


# Create dataset: 100 graphs (half label 0, half label 1)
conspiracy_graphs = [create_graph(G, 0) for G in conspiracy_graphs]
fiveg_conspiracy_graphs = [create_graph(G, 1) for G in fiveg_conspiracy_graphs]
non_conspiracy_graphs = [create_graph(G, 2) for G in non_conspiracy_graphs]

dataset = conspiracy_graphs + fiveg_conspiracy_graphs + non_conspiracy_graphs[:((len(conspiracy_graphs) + len(fiveg_conspiracy_graphs))//2)] # TODO: remove balance
random.shuffle(dataset)

# Train/test split
train_dataset = dataset[:300]
test_dataset = dataset[300:]

train_loader = DataLoader(train_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# ----------- Define GNN Model for graph classification -----------

class GCNGraphClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 32); self.norm1 = GraphNorm(32)
        self.conv2 = GCNConv(32, 32); self.norm2 = GraphNorm(32)
        self.conv3 = GCNConv(32, 32); self.norm3 = GraphNorm(32)
        self.lin = Linear(32, 3)
        self.dropout = Dropout(0.5)

    def forward(self, x, edge_index, batch):
        # During training randomly drop 10 % of edges → model stops memorising exact cascades
        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=0.1) 
        # 1) 1st convolution + GraphNorm + activation + dropout
        x1 = F.relu(self.norm1(self.conv1(x, edge_index)))
        x1 = self.dropout(x1)

        # 2) 2nd convolution + GraphNorm + activation + dropout + residual from x1
        h2 = self.norm2(self.conv2(x1, edge_index))
        x2 = F.relu(h2 + x1)
        x2 = self.dropout(x2)

        # 3) 3rd convolution + GraphNorm + activation + dropout + residual from x2
        h3 = self.norm3(self.conv3(x2, edge_index))
        x3 = F.relu(h3 + x2)
        x3 = self.dropout(x3)

        # 4) global add‐pool (sum of node embeddings per graph)
        pooled = global_add_pool(x3, batch)    # shape: [batch_size, hidden_dim]

        # 5) size‐normalization (divide by number of nodes per graph)
        counts = torch.bincount(batch)         # shape: [batch_size]
        pooled = pooled / counts.unsqueeze(1).float()

        # 6) final linear classification
        out = self.lin(pooled)                 # shape: [batch_size, num_classes]
        return out

# ----------- Training loop -----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCNGraphClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
loss_fn = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs

    train_loss = total_loss / total
    train_acc  = correct    / total
    return train_loss, train_acc

def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in range(1,501):
    train_loss, train_acc = train()
    test_acc = test(test_loader)
    print(f"Epoch {epoch:03d}  train_loss={train_loss:.4f}  train_acc={train_acc:.2%}  test_acc={test_acc:.2%}")

