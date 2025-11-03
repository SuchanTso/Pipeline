# conditional_consistency_gnn_v2.py
"""
Conditional Consistency Model for GNNs (V2)
 - Task: Given node degree (a), generate clustering coefficient (b).
 - NEW: Saves the model with the best training loss.
 - NEW: Adds graph-based visualization to compare condition, ground truth, and prediction.
"""
import math, random, copy, os ### NEW ###: import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

# ---------------- model components (Unchanged) ----------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
    def forward(self, t):
        t = t.flatten()
        device = t.device
        half = self.d_model // 2
        emb = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        out = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if out.size(1) < self.d_model:
            out = F.pad(out, (0, self.d_model - out.size(1)))
        return out

class AdaLN(nn.Module):
    def __init__(self, d_model, d_cond):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(d_cond, 2*d_model))
    def forward(self, x, cond):
        x_norm = self.layer_norm(x)
        style, shift = self.cond_proj(cond).chunk(2, dim=1)
        return x_norm * (1 + style) + shift

class SimpleGNNLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = GCNConv(d_model, d_model)
        self.adaln = AdaLN(d_model, d_model * 2)
    def forward(self, x, edge_index, t_emb, a_emb):
        t_emb_expanded = t_emb.expand(x.size(0), -1)
        cond_emb = torch.cat([t_emb_expanded, a_emb], dim=1)
        x_mod = self.adaln(x, cond_emb)
        out = F.relu(self.conv(x_mod, edge_index))
        return x + out

class PureConsistencyGNN(nn.Module):
    def __init__(self, F_in=1, F_a=1, d_model=64, num_layers=3):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.input_proj = nn.Linear(F_in, d_model)
        self.cond_a_proj = nn.Linear(F_a, d_model)
        self.layers = nn.ModuleList([SimpleGNNLayer(d_model) for _ in range(num_layers)])
        self.output_proj = nn.Linear(d_model, F_in)
    def forward(self, x_t, edge_index, t, cond_a):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=x_t.device)
        t = t.view(-1, 1)
        t_emb = self.time_emb(t)
        a_emb = self.cond_a_proj(cond_a)
        h = self.input_proj(x_t)
        for lyr in self.layers:
            h = lyr(h, edge_index, t_emb, a_emb)
        return self.output_proj(h)

# ---------------- sampling utilities (Unchanged) ----------------
def sample_log_uniform(batch, t_min, t_max, device):
    u = torch.rand(batch, device=device)
    log_min, log_max = math.log(t_min), math.log(t_max)
    return torch.exp(log_min + u * (log_max - log_min))

def make_xt_from_x0(x0, t, eps):
    return x0 + t.view(-1, 1) * eps

def edm_coeffs(t, sigma_data):
    c_skip = sigma_data**2 / (t**2 + sigma_data**2)
    c_out = (t * sigma_data) / (t**2 + sigma_data**2).sqrt()
    c_in = 1.0 / (t**2 + sigma_data**2).sqrt()
    return c_in, c_out, c_skip

def model_x0_pred(model, x_t, edge_index, t, sigma_data, cond_a):
    if not isinstance(t, torch.Tensor):
        t = torch.tensor([t], device=x_t.device)
    c_in, c_out, c_skip = edm_coeffs(t, sigma_data)
    x_in = c_in.view(-1, 1) * x_t
    F_pred = model(x_in, edge_index, t, cond_a)
    x0_pred = c_skip.view(-1, 1) * x_t + c_out.view(-1, 1) * F_pred
    return x0_pred

# ---------------- prepare data (Unchanged) ----------------
G = nx.karate_club_graph()
num_nodes = G.number_of_nodes()
# Feature a (condition): Node Degree
degs = torch.tensor([d for _, d in G.degree()], dtype=torch.float32).view(-1,1)
a0 = degs.clone().to(device)
a0_norm = (a0 - a0.mean()) / (a0.std() + 1e-12)
# Feature b (target): Clustering Coefficient
clustering_coeffs = torch.tensor(list(nx.clustering(G).values()), dtype=torch.float32).view(-1,1)
b0 = clustering_coeffs.clone().to(device)
b0_norm = (b0 - b0.mean()) / (b0.std() + 1e-12)
sigma_data = float(b0_norm.std().cpu().item())
edge_index = torch.tensor(list(G.edges)).t().contiguous().to(device)
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

# ---------------- hyperparams & setup ----------------
t_min = 0.002
t_max = 10.0
d_model = 64
num_layers = 3
model = PureConsistencyGNN(F_in=1, F_a=1, d_model=d_model, num_layers=num_layers).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-6)
steps = 5000
batch_pairs = 4
print_interval = 200
losses = []

### NEW ###: Variables for saving the best model
best_loss = float('inf')
save_path = "model/cond_cm.pt"
if os.path.dirname(save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

# ---------------- training loop (MODIFIED) ----------------
for step in tqdm(range(steps), desc="Conditional Consistency Training"):
    optimizer.zero_grad()
    total_loss = 0.0
    for _ in range(batch_pairs):
        t1 = sample_log_uniform(1, t_min, t_max, device)
        t2 = sample_log_uniform(1, t_min, t_max, device)
        eps = torch.randn_like(b0_norm)
        x_t1 = make_xt_from_x0(b0_norm, t1, eps)
        x_t2 = make_xt_from_x0(b0_norm, t2, eps)
        x0_pred1 = model_x0_pred(model, x_t1, edge_index, t1, sigma_data, cond_a=a0_norm)
        x0_pred2 = model_x0_pred(model, x_t2, edge_index, t2, sigma_data, cond_a=a0_norm)
        loss_pair = F.mse_loss(x0_pred1, x0_pred2)
        total_loss = total_loss + loss_pair
    
    total_loss = total_loss / float(batch_pairs)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optimizer.step()
    
    current_loss = total_loss.item()
    losses.append(current_loss)

    ### NEW ###: Check for best model and save
    if current_loss < best_loss:
        best_loss = current_loss
        torch.save(model.state_dict(), save_path)
        # Optional: print a message when a new best model is saved
        # print(f"Step {step+1}: New best model saved with loss {best_loss:.6e}")

    if (step+1) % print_interval == 0:
        print(f"step {step+1}/{steps} loss={current_loss:.6e} (best: {best_loss:.6e})")

print(f"\nTraining finished. Best model saved to {save_path}")

# ----------------- evaluation (MODIFIED) -----------------
# Load the best model for evaluation
model.load_state_dict(torch.load(save_path))
model.eval()

with torch.no_grad():
    x_single = torch.randn_like(b0_norm) * t_max
    t_gen = torch.tensor([t_max], device=device)
    pred_single_norm = model_x0_pred(model, x_single, edge_index, t_gen, sigma_data, cond_a=a0_norm)
    r2_single = r2_score(b0_norm.cpu().numpy(), pred_single_norm.cpu().numpy())
print(f"R2 single-step generation (on normalized data): {r2_single:.4f}")


# ----------------- 1. Scatter plot visualization (Unchanged) -----------------
plt.figure(figsize=(8, 6))
plt.scatter(b0_norm.cpu().numpy(), pred_single_norm.cpu().numpy(), c='green', alpha=0.7)
plt.plot([-2.5, 2.5], [-2.5, 2.5], 'r--')
plt.xlabel("True Feature b (Normalized Clustering Coeff)")
plt.ylabel("Predicted Feature b")
plt.title(f"Conditional Generation Scatter Plot (R²={r2_single:.4f})")
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.savefig("conditional_consistency_gnn_scatter.png")
print("Saved scatter plot: conditional_consistency_gnn_scatter.png")


### NEW ###: 2. Graph-based visualization
plt.figure(figsize=(20, 5))
pos = nx.spring_layout(G, seed=seed) # Fix node positions for comparison

# Get feature values for coloring
a0_vals = a0_norm.cpu().numpy().flatten()
b0_vals = b0_norm.cpu().numpy().flatten()
pred_vals = pred_single_norm.cpu().numpy().flatten()

# Determine a shared color range for b0 and prediction for fair comparison
vmin = min(b0_vals.min(), pred_vals.min())
vmax = max(b0_vals.max(), pred_vals.max())

# Plot 1: Condition (Feature a)
plt.subplot(1, 3, 1)
nx.draw(G, pos, with_labels=True, node_color=a0_vals, cmap=plt.cm.viridis, font_color='white')
plt.title("Condition: Node Degree (a)")
# Add a colorbar
sm_a = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=a0_vals.min(), vmax=a0_vals.max()))
sm_a.set_array([])
plt.colorbar(sm_a, ax=plt.gca(), orientation='horizontal', pad=0.05)

# Plot 2: Ground Truth (Feature b)
plt.subplot(1, 3, 2)
nx.draw(G, pos, with_labels=True, node_color=b0_vals, cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax, font_color='white')
plt.title("Ground Truth: Clustering Coeff (b)")
# Add a colorbar with shared scale
sm_b = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm_b.set_array([])
plt.colorbar(sm_b, ax=plt.gca(), orientation='horizontal', pad=0.05)

# Plot 3: Generated Result
plt.subplot(1, 3, 3)
nx.draw(G, pos, with_labels=True, node_color=pred_vals, cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax, font_color='white')
plt.title(f"Generated Clustering Coeff (R²={r2_single:.4f})")
# Add a colorbar with shared scale
plt.colorbar(sm_b, ax=plt.gca(), orientation='horizontal', pad=0.05)

plt.tight_layout()
plt.savefig("conditional_consistency_gnn_graph_vis.png")
print("Saved graph visualization: conditional_consistency_gnn_graph_vis.png")