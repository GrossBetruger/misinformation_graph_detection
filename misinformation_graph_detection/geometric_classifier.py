import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from misinformation_graph_detection.graph_classifier import load_graphs
import kagglehub
from pathlib import Path
import networkx as nx
from torch_geometric.nn import GraphNorm, global_add_pool
from torch_geometric.utils import dropout_edge
from torch.optim.lr_scheduler import OneCycleLR

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import random

from sklearn.preprocessing import StandardScaler
from torch.nn import Linear, Dropout
from sklearn.metrics import f1_score, classification_report
import plotly.express as px

path = kagglehub.dataset_download("arashnic/misinfo-graph")
PATH = Path(path)

BASE_MAX_LR = 1e-3

conspiracy_graphs, fiveg_conspiracy_graphs, non_conspiracy_graphs = load_graphs(PATH)



# ----------- Create synthetic graph dataset -----------


def create_graph(G: nx.Graph, label: int) -> Data:
    # collect raw features
    x = [[G.nodes[n][k] for k in ("time", "friends", "followers")] for n in G.nodes()]
    vals = torch.tensor(x, dtype=torch.float)  # shape [N,3]
    # per‑graph normalization to zero‑mean/unit‑std
    # vals = (vals - vals.mean(0)) / (vals.std(0) + 1e-6)
    data = from_networkx(G)
    data.x = vals
    scaler = StandardScaler()
    data.x = torch.tensor(scaler.fit_transform(data.x), dtype=torch.float)

    data.y = torch.tensor(label, dtype=torch.long)  # scalar, NOT [label]
    return data


mean_conspiracy_graph_size = sum(
    [G.number_of_edges() for G in conspiracy_graphs]
) / len(conspiracy_graphs)
mean_fiveg_conspiracy_graph_size = sum(
    [G.number_of_edges() for G in fiveg_conspiracy_graphs]
) / len(fiveg_conspiracy_graphs)
mean_non_conspiracy_graph_size = sum(
    [G.number_of_edges() for G in non_conspiracy_graphs]
) / len(non_conspiracy_graphs)

print(f"Mean conspiracy graph size: {mean_conspiracy_graph_size}")
print(f"Mean fiveg conspiracy graph size: {mean_fiveg_conspiracy_graph_size}")
print(f"Mean non conspiracy graph size: {mean_non_conspiracy_graph_size}")


# Create dataset: 100 graphs (half label 0, half label 1)
conspiracy_graphs = [create_graph(G, 0) for G in conspiracy_graphs]
fiveg_conspiracy_graphs = [create_graph(G, 1) for G in fiveg_conspiracy_graphs]
non_conspiracy_graphs = [create_graph(G, 2) for G in non_conspiracy_graphs]

dataset = (
    conspiracy_graphs
    + fiveg_conspiracy_graphs
    + non_conspiracy_graphs[
        : ((len(conspiracy_graphs) + len(fiveg_conspiracy_graphs)) // 2)
    ]
)  # TODO: remove balance
random.shuffle(dataset)

# Train/test split
train_dataset = dataset[:300]
test_dataset = dataset[300:]

train_loader = DataLoader(train_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# ----------- Define GNN Model for graph classification -----------
BASE_HIDDEN = 32


class GCNGraphClassifier(torch.nn.Module):
    def __init__(self, feat_mask_p=0.1, edge_dropout_p=0.1):
        super().__init__()
        self.feat_mask_p = feat_mask_p
        self.edge_dropout_p = edge_dropout_p
        self.feat_mask = torch.nn.Dropout(p=feat_mask_p)
        self.conv1 = GCNConv(3, BASE_HIDDEN)
        self.norm1 = GraphNorm(BASE_HIDDEN)
        self.conv2 = GCNConv(BASE_HIDDEN, BASE_HIDDEN)
        self.norm2 = GraphNorm(BASE_HIDDEN)
        self.conv3 = GCNConv(BASE_HIDDEN, BASE_HIDDEN)
        self.norm3 = GraphNorm(BASE_HIDDEN)
        self.lin = Linear(BASE_HIDDEN, 3)
        self.dropout = Dropout(0.5)

    def forward(self, x, edge_index, batch):
        # During training randomly drop 10 % of edges → model stops memorising exact cascades
        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout_p)
            x = self.feat_mask(x)
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
        pooled = global_add_pool(x3, batch)  # shape: [batch_size, hidden_dim]

        # 5) size‐normalization (divide by number of nodes per graph)
        counts = torch.bincount(batch)  # shape: [batch_size]
        pooled = pooled / counts.unsqueeze(1).float()

        # 6) final linear classification
        out = self.lin(pooled)  # shape: [batch_size, num_classes]
        return out


# ----------- Training loop -----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCNGraphClassifier().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_MAX_LR, weight_decay=5e-4)

# 5) Hook up OneCycleLR with the scaled max_lr

hidden_dim = model.conv1.out_channels

# 3) Compute a scaling factor ∝ sqrt(curr / base)
scale = (hidden_dim / BASE_HIDDEN) ** 0.5
max_lr = BASE_MAX_LR * scale


epochs = 100
steps_per_epoch = len(train_loader)


def train(scheduler: OneCycleLR):
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
        scheduler.step()

        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs

    train_loss = total_loss / total
    train_acc = correct / total
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


def evaluate(loader: DataLoader) -> tuple[float, np.ndarray, float, str]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            ys.append(data.y.cpu())
            ps.append(pred.cpu())
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    
    loss = loss_fn(out, data.y)
    acc = (y_true == y_pred).mean()
    # per‑class F1: array of shape [n_classes]
    f1_per_class = f1_score(y_true, y_pred, average=None)
    # macro F1: unweighted mean of per‑class F1
    f1_macro = f1_score(y_true, y_pred, average="macro")
    # optional nice text report
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["conspiracy", "fiveg_conspiracy", "non_conspiracy"],
        zero_division=0,
    )
    return acc, f1_per_class, f1_macro, report, loss


num_epochs = 200
scheduler = OneCycleLR(
    optimizer,
    max_lr=max_lr,
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.3,       # 30% of cycle ramp‐up, 70% anneal
    anneal_strategy="cos"  # cosine down‐ramp
)

test_acc_history = []
f1_mac_history = []
loss_history = []

for epoch in range(1, num_epochs + 1):
    if epoch > 100:
        model.feat_mask_p = 0.0
        model.edge_dropout_p = 0.0
    train_loss, train_acc = train(scheduler)
    if epoch % 100 == 0:
        epoch_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:03d} completed. Learning rate is now {epoch_lr:.6f}")
    test_acc, f1_pc, f1_mac, rpt, loss = evaluate(test_loader)
    test_acc_history.append(test_acc)
    f1_mac_history.append(f1_mac)
    loss_history.append(loss)

    print(
        f"Epoch {epoch:03d}  "
        f"train_loss={train_loss:.4f}  "
        f"train_acc={train_acc:.2%}  "
        f"test_acc={test_acc:.2%}  "
        f"macro_F1={f1_mac:.3f}  "
        f"F1_per_class={[f'{x:.3f}' for x in f1_pc]}"
    )

# plot test accuracy and f1 macro
fig = px.line(loss_history, title="Loss")
fig.show()

fig = px.line(test_acc_history, title="Test Accuracy")
fig.show()

fig = px.line(f1_mac_history, title="F1 Macro")
fig.show()

human_readable_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
# save model
models_path = Path("models")
models_path.mkdir(exist_ok=True)
number_of_parameters = sum(p.numel() for p in model.parameters())
model_path = (
    models_path
    / f"model_{human_readable_time}_params_{number_of_parameters}_epochs_{num_epochs}.pth"
)
torch.save(model.state_dict(), model_path)

print(f"Model saved to {model_path}")

# load model
model = GCNGraphClassifier().to(device)
model.load_state_dict(torch.load(model_path))

# test model
test_acc = test(test_loader)
print(f"Test accuracy: {test_acc:.2%}")
