# train_phys_consistency.py
import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# torch-geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv


# ----------------------------
# 0. Repro / Device
# ----------------------------
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

# ----------------------------
# 1. Utilities (from your original script)
# ----------------------------
def select_random_indices(total_num, num_to_select, excluded_indices, seed=None):
    if seed is not None: rng = np.random.default_rng(seed)
    else: rng = np.random.default_rng()
    all_indices = np.arange(total_num)
    try:
        excluded = excluded_indices.cpu().numpy()
    except:
        excluded = np.array([], dtype=int)
    eligible = np.setdiff1d(all_indices, excluded, assume_unique=True)
    if num_to_select > len(eligible): num_to_select = len(eligible)
    if num_to_select <= 0:
        return torch.empty(0, dtype=torch.long)
    selected = rng.choice(eligible, num_to_select, replace=False)
    return torch.from_numpy(selected).long()

def sample_log_uniform(batch, t_min, t_max, device):
    u = torch.rand(batch, device=device)
    log_min, log_max = math.log(t_min), math.log(t_max)
    return torch.exp(log_min + u * (log_max - log_min))

def make_xt_from_x0(x0, t, eps):
    # x0: (N,1) or (batch_nodes,1) ; t: (batch_graphs,) expanded to nodes outside
    # here we assume t already expanded to node-level before passing
    return x0 + t.view(-1,1) * eps

def edm_coeffs(t, sigma_data):
    t = t.view(-1, 1)
    denom = (t**2 + sigma_data**2)
    c_skip = sigma_data**2 / denom
    c_out = (t * sigma_data) / torch.sqrt(denom)
    c_in = 1.0 / torch.sqrt(denom)
    return c_in, c_out, c_skip

# ----------------------------
# 2. Mock data loader (fallback when real loader not provided)
# ----------------------------
def load_data_fallback(dataset_name=None, path=None, timesteps=72, batch_size=1, mask_ratio=0.2):
    print("--- Using MOCK data loader ---")
    class DummyDataset(Dataset):
        def __init__(self, num_graphs=200, num_nodes=50):
            super().__init__()
            self.num_graphs = num_graphs
            self.num_nodes = num_nodes
            # simple random graph
            G = nx.gnm_random_graph(num_nodes, int(num_nodes*2), seed=seed)
            edges = list(G.edges())
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            self.edge_index = edge_index
            # static node features: head, demand, masked_pressure, is_reservoir
            self.x_static = torch.randn(num_nodes, 4)
            # node_type: mark reservoir node(s)
            self.node_type = torch.zeros(num_nodes, 1)
            self.node_type[0] = 1
            # edge_attr: diameter, length, roughness
            E = edge_index.size(1)
            self.edge_attr = torch.randn(E, 3).abs() + 0.1
            # y_node: random true pressures
            self.y_node = torch.randn(num_graphs, num_nodes, 1)
        def len(self): return self.num_graphs
        def get(self, idx):
            data = Data(x_static=self.x_static, y_node=self.y_node[idx:idx+1].squeeze(0),
                        edge_index=self.edge_index, node_type=self.node_type, edge_attr=self.edge_attr)
            # adapt fields: put x (node input) and y_node as previous script expects
            # We'll set data.x as x_static here for simplicity; training code expects data.x_static separately
            data.x = torch.zeros(self.num_nodes, 4)  # placeholder (not used)
            return data
    ds = DummyDataset()
    train_size = int(0.7 * len(ds)); val_size = int(0.15 * len(ds))
    test_size = len(ds) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    # normalizers placeholders
    class Z:
        def __init__(self): self.mean=0; self.std=1
        def fit(self,x): pass
        def transform(self,x): return x
    return train_loader, val_loader, test_loader, Z(), Z()

# wrapper load_data: try to import from utils else fallback
def load_data_wrapper(dataset_name, path, timesteps, batch_size, mask_ratio):
    try:
        from utils import load_data as real_load_data
        print("Using real load_data from utils.")
        return real_load_data(dataset_name, path, timesteps, batch_size, mask_ratio)
    except Exception as e:
        print("Real load_data not found or failed; using fallback mock loader. (Error: {})".format(e))
        return load_data_fallback(dataset_name, path, timesteps, batch_size, mask_ratio)

# ----------------------------
# 3. Model: PureConsistencyGNN (modified to output mu + logvar)
# ----------------------------

class ConsistencyGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # 均值预测
        self.mean_head = nn.Linear(hidden_dim, out_dim)
        # 方差预测（用softplus保证非负）
        self.var_head = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.Softplus()
        )

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mean_head(h)
        sigma = self.var_head(h)
        return mu, sigma

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
    def __init__(self,d_model,d_cond):
        super().__init__()
        self.layer_norm=nn.LayerNorm(d_model,elementwise_affine=False)
        self.cond_proj=nn.Sequential(nn.SiLU(),nn.Linear(d_cond,2*d_model))
    def forward(self,x,cond):
        x_norm=self.layer_norm(x)
        style,shift=self.cond_proj(cond).chunk(2,dim=1)
        return x_norm*(1+style)+shift

class SimpleGNNLayer(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.conv=GCNConv(d_model,d_model)
        self.adaln=AdaLN(d_model,d_model*2)
    def forward(self,x,edge_index,t_emb,a_emb):
        t_emb_expanded = t_emb.expand(x.size(0), -1)
        cond_emb = torch.cat([t_emb_expanded, a_emb], dim=1)
        x_mod = self.adaln(x, cond_emb)
        out = F.relu(self.conv(x_mod, edge_index))
        return x + out

class PureConsistencyGNN(nn.Module):
    def __init__(self, F_in=2, F_a=3, d_model=64, num_layers=3):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.input_proj = nn.Linear(F_in, d_model)
        self.cond_a_proj = nn.Linear(F_a, d_model)
        self.layers = nn.ModuleList([SimpleGNNLayer(d_model) for _ in range(num_layers)])
        # output mu and logvar per node
        self.output_proj = nn.Linear(d_model, 2)
    def forward(self, x_t, edge_index, t, cond_a):
        if not isinstance(t, torch.Tensor): t = torch.tensor([t], device=x_t.device)
        t = t.view(-1, 1)
        t_emb = self.time_emb(t)
        a_emb = self.cond_a_proj(cond_a)
        h = self.input_proj(x_t)
        for lyr in self.layers: h = lyr(h, edge_index, t_emb, a_emb)
        out = self.output_proj(h)
        mu = out[:, 0:1]
        logvar = out[:, 1:2]
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar

# ----------------------------
# 4. Physical helpers (implicit flow)
# ----------------------------
def build_k_from_edge_attr(edge_attr):
    if edge_attr is None:
        return None
    diam = edge_attr[:, 0].float().clamp(min=1e-6)
    length = edge_attr[:, 1].float().clamp(min=1e-6)
    k = (diam ** 4.0) / length
    # normalize
    if torch.isfinite(k).all():
        k = (k - k.mean()) / (k.std() + 1e-9)
    else:
        k = torch.ones_like(k)
    return k.view(-1, 1)

def compute_node_residuals_from_mu(mu, edge_index, edge_attr, node_demand=None, non_linearity=0.54):
    """
    Computes node residuals with a non-linear pressure-flow relationship.
    Flow Q is proportional to (delta_p)^non_linearity. For Hazen-Williams, this is ~0.54 (1/1.852).
    """
    src, dst = edge_index[0], edge_index[1]
    device = mu.device
    
    if edge_attr is None:
        k = torch.ones((edge_index.size(1), 1), device=device)
    else:
        # k now represents the pipe conductance coefficient
        # For Hazen-Williams: k ∝ D^2.63 / L^0.54
        # We can approximate this or use a simple form as before.
        diam = edge_attr[:, 0].float().clamp(min=1e-6)
        length = edge_attr[:, 1].float().clamp(min=1e-6)
        # A simplified conductance term, the exact physics can be complex.
        # Let's stick with a learnable effective k, but apply non-linearity on pressure.
        k = (diam.pow(2.63) / length.pow(0.54)).clamp(min=1e-6, max=1e6)
        # Normalizing k is very important to keep gradients stable
        k = (k - k.mean()) / (k.std() + 1e-9)
        k = k.view(-1, 1)

    p_src, p_dst = mu[src], mu[dst]
    delta_p = p_src - p_dst
    
    # --- THIS IS THE KEY CHANGE ---
    # Apply non-linearity. The sign preserves flow direction.
    # Add a small epsilon for numerical stability of the gradient at delta_p=0.
    q_edge = k * torch.sign(delta_p) * (torch.abs(delta_p) + 1e-8).pow(non_linearity)
    
    N = mu.shape[0]
    node_res = torch.zeros((N, 1), device=device)
    node_res.index_add_(0, dst, q_edge)
    node_res.index_add_(0, src, -q_edge)
    
    # In water networks, demand is usually positive for consumption, so it's flow *out* of the system.
    # The convention is sum(flows_in) - sum(flows_out) - demand = 0
    # Our `net_flow = flow_in - flow_out`, so `net_flow - demand` is the residual.
    if node_demand is not None:
        # Assuming node_demand is positive for consumption (flow leaving the node)
        node_res = node_res - node_demand
        
    return node_res, q_edge


def physical_constraint_loss(pred_pressure, edge_index):
    src, dst = edge_index
    # 假设简单守恒约束：节点压差在相邻节点之间平滑
    diff = pred_pressure[src] - pred_pressure[dst]
    return torch.mean(diff ** 2)

# ----------------------------
# 5. model_x0_pred_inpainting: prediction returning mu and logvar
# ----------------------------
def model_x0_pred_inpainting(model, x_t_combined, edge_index, t, sigma_data, cond_a):
    x_t_pressure, mask = x_t_combined[:, 0:1], x_t_combined[:, 1:2]
    if not isinstance(t, torch.Tensor): t = torch.tensor([t], device=x_t_combined.device)
    c_in, c_out, c_skip = edm_coeffs(t, sigma_data)
    x_in_pressure = c_in * x_t_pressure
    model_input = torch.cat([x_in_pressure, mask], dim=1)
    mu_pred, logvar_pred = model(model_input, edge_index, t, cond_a)
    x0_mu_pred = c_skip * x_t_pressure + c_out * mu_pred
    x0_logvar_pred = logvar_pred
    return x0_mu_pred, x0_logvar_pred

# ----------------------------
# 6. Training & Evaluation & Visualization
# ----------------------------
def train_one_epoch(model, dataloader, optimizer, epoch, epochs, t_min, t_max, batch_pairs, device,
                    lambda_phys=1.0, lambda_nll=1.0, beta=1.0, sigma0=1e-3):
    model.train()
    total_epoch_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in pbar:
        batch = batch.to(device)
        b0_norm, a0_norm, edge_index = batch.y_node, batch.x_static, batch.edge_index
        optimizer.zero_grad(); total_batch_loss = 0.0
        for _ in range(batch_pairs):
            # mask sampling
            mask = torch.zeros_like(b0_norm)
            reservoir_indices = torch.where(batch.node_type.flatten() == 1)[0]
            num_masked_nodes = int(batch.num_nodes * random.uniform(0.1, 0.5))
            known_indices = select_random_indices(batch.num_nodes, num_masked_nodes, reservoir_indices)
            if known_indices.numel() > 0:
                mask[known_indices] = 1.0
            # times
            num_graphs_in_batch = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
            t1 = sample_log_uniform(num_graphs_in_batch, t_min, t_max, device)
            t2 = sample_log_uniform(num_graphs_in_batch, t_min, t_max, device)
            eps = torch.randn_like(b0_norm)
            t1_expanded, t2_expanded = t1[batch.batch], t2[batch.batch]
            xt1_noisy = make_xt_from_x0(b0_norm, t1_expanded, eps)
            xt2_noisy = make_xt_from_x0(b0_norm, t2_expanded, eps)
            xt1_pressure = torch.where(mask.bool(), b0_norm, xt1_noisy)
            xt2_pressure = torch.where(mask.bool(), b0_norm, xt2_noisy)
            xt1_combined = torch.cat([xt1_pressure, mask], dim=1)
            xt2_combined = torch.cat([xt2_pressure, mask], dim=1)

            # model preds
            x0_mu_pred1, x0_logvar_pred1 = model_x0_pred_inpainting(model, xt1_combined, edge_index, t1_expanded, 1.0, a0_norm)
            x0_mu_pred2, x0_logvar_pred2 = model_x0_pred_inpainting(model, xt2_combined, edge_index, t2_expanded, 1.0, a0_norm)

            # consistency loss on mu
            L_cons = F.mse_loss(x0_mu_pred1, x0_mu_pred2)

            # physical residuals
            node_demand = a0_norm[:, 1:2] if a0_norm is not None else None
            node_res1, _ = compute_node_residuals_from_mu(x0_mu_pred1, edge_index, getattr(batch, 'edge_attr', None), node_demand)
            L_phys = (node_res1 ** 2).mean()

            # NLL with modulated variance
            var_theta = torch.exp(x0_logvar_pred1)
            eta = beta * (node_res1 ** 2)
            var_tilde = var_theta + eta + sigma0
            nll_per_node = ((b0_norm - x0_mu_pred1) ** 2) / var_tilde + torch.log(var_tilde)
            L_nll = nll_per_node.mean()

            loss_pair = L_cons# + lambda_phys * L_phys + lambda_nll * L_nll
            total_batch_loss += loss_pair

        avg_batch_loss = total_batch_loss / batch_pairs
        avg_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_epoch_loss += avg_batch_loss.item()
        pbar.set_postfix(loss=avg_batch_loss.item())
    avg_epoch_loss = total_epoch_loss / max(1, len(dataloader))
    print(f"Epoch {epoch+1} Avg Train Loss: {avg_epoch_loss:.6e}")
    return avg_epoch_loss

@torch.no_grad()
def evaluate(model, test_loader, t_max, save_path, device, alpha_beta=1.0, sigma0=1e-3):
    print("\nStarting evaluation on the test set for inpainting...")
    model.load_state_dict(torch.load(save_path)); model.eval()
    all_true_unknown, all_preds_unknown = [], []
    preds_list, trues_list, var_list, res_list = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Inpainting"):
            batch = batch.to(device)
            b0_norm, a0_norm, edge_index = batch.y_node, batch.x_static, batch.edge_index
            mask = torch.zeros_like(b0_norm)
            reservoir_indices = torch.where(batch.node_type.flatten() == 1)[0]
            num_known_nodes = int(batch.num_nodes * 0.2)
            known_indices = select_random_indices(batch.num_nodes, num_known_nodes, reservoir_indices, seed=seed)
            if known_indices.numel() > 0: mask[known_indices] = 1.0
            unknown_mask = (1.0 - mask).bool()
            if not unknown_mask.any(): continue
            initial_noise = torch.randn_like(b0_norm) * t_max
            x_T_pressure = torch.where(mask.bool(), b0_norm, initial_noise)
            x_T_combined = torch.cat([x_T_pressure, mask], dim=1)
            t_gen = torch.full((batch.num_graphs,), t_max, device=device)
            t_gen_expanded = t_gen[batch.batch]
            mu_pred, logvar_pred = model_x0_pred_inpainting(model, x_T_combined, edge_index, t_gen_expanded, 1.0, a0_norm)
            node_res, _ = compute_node_residuals_from_mu(mu_pred, edge_index, getattr(batch, 'edge_attr', None), a0_norm[:,1:2])
            var_theta = torch.exp(logvar_pred)
            var_tilde = var_theta + alpha_beta * (node_res ** 2) + sigma0

            preds_list.append(mu_pred.cpu())
            trues_list.append(b0_norm.cpu())
            var_list.append(var_tilde.cpu())
            res_list.append(node_res.cpu())

            all_true_unknown.append(b0_norm[unknown_mask].cpu())
            all_preds_unknown.append(mu_pred[unknown_mask].cpu())

    if len(all_true_unknown) == 0:
        print("No unknown nodes found in test set for evaluation.")
        return None, None, None, None, None
    all_true_tensor = torch.cat(all_true_unknown); all_preds_tensor = torch.cat(all_preds_unknown)
    final_r2 = r2_score(all_true_tensor.numpy(), all_preds_tensor.numpy())
    print(f"\nFinal Inpainting R² Score (on unknown nodes): {final_r2:.4f}")

    preds = torch.cat(preds_list, dim=0); trues = torch.cat(trues_list, dim=0)
    vars_ = torch.cat(var_list, dim=0); res = torch.cat(res_list, dim=0)
    return final_r2, preds, trues, vars_, res

def visualize_progress(model, save_path_fig, dataloader, r2_score_val, t_max, device, alpha_beta=1.0, sigma0=1e-3):
    test_sample = next(iter(dataloader)).to(device)
    b0_norm, a0_norm, edge_index = test_sample.y_node, test_sample.x_static, test_sample.edge_index
    mask = torch.zeros_like(b0_norm)
    reservoir_indices = torch.where(test_sample.node_type.flatten() == 1)[0]
    num_known_nodes = int(test_sample.num_nodes * 0.2)
    known_indices = select_random_indices(test_sample.num_nodes, num_known_nodes, reservoir_indices, seed=seed)
    if known_indices.numel() > 0: mask[known_indices] = 1.0
    with torch.no_grad():
        initial_noise = torch.randn_like(b0_norm) * t_max
        x_T_pressure = torch.where(mask.bool(), b0_norm, initial_noise)
        x_T_combined = torch.cat([x_T_pressure, mask], dim=1)
        t_gen = torch.full((test_sample.num_graphs,), t_max, device=device)
        t_gen_expanded = t_gen[test_sample.batch]
        mu_pred, logvar_pred = model_x0_pred_inpainting(model, x_T_combined, edge_index, t_gen_expanded, 1.0, a0_norm)

    node_res, _ = compute_node_residuals_from_mu(mu_pred, edge_index, getattr(test_sample, 'edge_attr', None), a0_norm[:,1:2])
    var_theta = torch.exp(logvar_pred)
    var_tilde = var_theta + alpha_beta * (node_res ** 2) + sigma0

    G_vis = nx.from_edgelist(edge_index.t().cpu().numpy())
    pos = nx.spring_layout(G_vis, seed=seed)
    true_vals = b0_norm.cpu().numpy().flatten(); pred_vals = mu_pred.cpu().numpy().flatten()
    vmin, vmax = min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())
    plt.figure(figsize=(20, 8))
    plt.subplot(2, 3, 1)
    nx.draw(G_vis, pos, with_labels=False, node_size=80, node_color=true_vals, cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax)
    nx.draw_networkx_nodes(G_vis, pos, nodelist=known_indices.cpu().tolist(), node_color='lime', edgecolors='black', node_size=100)
    plt.title("Ground Truth (Known nodes in green)")

    plt.subplot(2, 3, 2)
    init_vals = x_T_pressure.cpu().numpy().flatten()
    nx.draw(G_vis, pos, with_labels=False, node_size=80, node_color=init_vals, cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax)
    nx.draw_networkx_nodes(G_vis, pos, nodelist=known_indices.cpu().tolist(), node_color='lime', edgecolors='black', node_size=100)
    plt.title("Initial State (Known + Noise)")

    plt.subplot(2, 3, 3)
    nx.draw(G_vis, pos, with_labels=False, node_size=80, node_color=pred_vals, cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax)
    nx.draw_networkx_nodes(G_vis, pos, nodelist=known_indices.cpu().tolist(), node_color='lime', edgecolors='black', node_size=100)
    plt.title(f"Inpainting Prediction (R² unk={r2_score_val:.4f})")

    plt.subplot(2, 3, 4)
    plt.scatter(np.abs(node_res.cpu().numpy()), np.sqrt(var_tilde.cpu().numpy()), alpha=0.7)
    plt.xlabel("|Physical Residual|"); plt.ylabel("Predicted std")
    plt.title("Residual vs Predicted Std")

    plt.subplot(2, 3, 5)
    plt.scatter(b0_norm.cpu().numpy().flatten(), pred_vals, alpha=0.6)
    plt.plot([b0_norm.cpu().numpy().min(), b0_norm.cpu().numpy().max()],
             [b0_norm.cpu().numpy().min(), b0_norm.cpu().numpy().max()], 'r--')
    plt.xlabel("True"); plt.ylabel("Pred")
    plt.title("Pred vs True")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path_fig), exist_ok=True)
    plt.savefig(save_path_fig)
    print(f"Saved graph visualization: {save_path_fig}")

def visualize_uncertainty_band(mu_pred, var_tilde, true_vals, save_path_uncertainty):
    """
    绘制预测均值 μ 与不确定性区间 (μ ± σ) 的可视化。
    纵轴: 节点压力（或其他量）
    横轴: 节点索引 (可换成物理位置序号)
    """
    mu_pred = mu_pred.cpu().numpy().flatten()
    std_pred = np.sqrt(var_tilde.cpu().numpy().flatten())
    true_vals = true_vals.cpu().numpy().flatten()

    idx = np.arange(len(mu_pred))

    plt.figure(figsize=(10, 5))
    plt.plot(idx, true_vals, 'k-', lw=2, label='Ground Truth')
    plt.plot(idx, mu_pred, 'b--', lw=2, label='Predicted Mean (μ)')
    plt.fill_between(idx,
                     mu_pred - std_pred,
                     mu_pred + std_pred,
                     color='blue', alpha=0.2, label='±1σ Uncertainty Band')
    plt.xlabel("Node index")
    plt.ylabel("Normalized Pressure")
    plt.legend()
    plt.title("Predicted Mean and Uncertainty Interval")
    plt.tight_layout()
    plt.savefig(save_path_uncertainty)
    plt.show()
    print(f"Saved uncertainty visualization: {save_path_uncertainty}")


# ----------------------------
# 7. Main script: training loop & calls
# ----------------------------
def main():
    # data loading (tries real loader first)
    train_loader, val_loader, test_loader, _, _ = load_data_wrapper("PretrainDataset", "/data/zsc/Pipeline/data/epaNet/Anytown.inp", 72, 1, mask_ratio=0.2)

    model = PureConsistencyGNN(F_in=2, F_a=3, d_model=64, num_layers=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)

    epochs = 300
    t_min = 0.002; t_max = 80.0; batch_pairs = 2
    best_loss = float('inf')
    save_path = "model/inpainting_cm_phys_uncert.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # optionally load
    if os.path.exists(save_path):
        try:
            model.load_state_dict(torch.load(save_path))
            print("Loaded existing model.")
        except Exception as e:
            print("Could not load model:", e)

    for epoch in range(epochs):
        avg_epoch_loss = train_one_epoch(model, train_loader, optimizer, epoch, epochs, t_min, t_max, batch_pairs, device,
                                        lambda_phys=0, lambda_nll=0, beta=1e-3, sigma0=1e-6)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved with loss {best_loss:.6e}")

        if (epoch + 1) % 10 == 0:
            # quick eval & viz
            try:
                final_r2, preds, trues, vars_, res = evaluate(model, test_loader, t_max, save_path, device, alpha_beta=1.0, sigma0=1e-6)
                # visualize_progress(model, "figures/cm_inpainting_phys_uncert_epoch{:03d}.png".format(epoch+1), test_loader, final_r2, t_max, device, alpha_beta=1.0, sigma0=1e-6)
                visualize_uncertainty_band(preds, vars_, trues, save_path_uncertainty="uncertainty_band.png")

            except Exception as e:
                print("Evaluation/visualization error:", e)

if __name__ == "__main__":
    main()
