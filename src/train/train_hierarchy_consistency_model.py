# full_script_structural_mask.py
"""
Conditional Consistency Model for Inpainting with Structural Masking.

This script implements the Hierarchical Consistency Model (HCM) and a Baseline
model for a graph inpainting task. The key feature is the use of structural
(regional) masking instead of random node masking. This is designed to
specifically test the model's ability to handle global information and avoid
"structural drift" when large, contiguous areas of the graph are unknown.

To perform the ablation study:
1. Set `run_mode = 'HCM'` to train and evaluate the Hierarchical Consistency Model.
2. Set `run_mode = 'Baseline'` to train and evaluate the Baseline model (hcl_lambda=0).
Compare the evaluation metrics, especially the 'Mean Error (ME)', from both runs.
"""
import math, random, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph
import networkx as nx
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import load_data

# ==============================================================================
# 0. 配置与MOCK数据
# ==============================================================================

# --- Reproducibility & Device Setup ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# --- MOCK UTILS FOR STANDALONE EXECUTION ---
def load_data_mock(dataset_name, path, timesteps, batch_size, mask_ratio):
    print("--- Using MOCK data loader ---")
    class ZScoreNormalizer:
        def __init__(self): self.mean = 0; self.std = 1
        def fit(self, x): self.mean = x.mean(0); self.std = x.std(0); self.std[self.std == 0] = 1
        def transform(self, x): return (x - self.mean) / self.std
    class DummyDataset(Dataset):
        def __init__(self, num_graphs=100, num_nodes=80):
            super().__init__()
            self.num_graphs = num_graphs
            G = nx.watts_strogatz_graph(n=num_nodes, k=4, p=0.1, seed=seed)
            self.edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
            self.edge_index = torch.cat([self.edge_index, self.edge_index.flip(0)], dim=1)
            self.x_static = torch.randn(num_nodes, 3)
            self.node_type = torch.zeros(num_nodes, 1)
            self.node_type[0] = 1
            self.y_node = torch.randn(num_graphs, num_nodes, 1)
        def len(self): return self.num_graphs
        def get(self, idx):
            return Data(x_static=self.x_static, y_node=self.y_node[idx], edge_index=self.edge_index, node_type=self.node_type, num_nodes=self.y_node.shape[1])
    
    full_dataset = DummyDataset()
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader, ZScoreNormalizer(), ZScoreNormalizer()

# ==============================================================================
# 1. NEW: 结构化掩码工具函数
# ==============================================================================
def create_structural_mask(data, num_blocks=1, block_radius=2, known_ratio_range=(0.1, 0.5), excluded_indices=None):
    num_nodes = data.num_nodes
    mask = torch.ones(num_nodes, 1, device=data.edge_index.device)
    target_known_ratio = random.uniform(*known_ratio_range)
    num_target_unknown = int(num_nodes * (1 - target_known_ratio))
    
    candidate_centers = torch.arange(num_nodes, device=data.edge_index.device)
    if excluded_indices is not None:
        exclude_mask = torch.ones_like(candidate_centers, dtype=torch.bool)
        if excluded_indices.numel() > 0:
            exclude_mask[excluded_indices] = False
        candidate_centers = candidate_centers[exclude_mask]

    if len(candidate_centers) == 0: return mask

    current_unknown_nodes = set()
    attempts = 0
    while len(current_unknown_nodes) < num_target_unknown and attempts < num_nodes * 2:
        center_node = candidate_centers[torch.randint(len(candidate_centers), (1,))]
        block_nodes, _, _, _ = k_hop_subgraph(
            node_idx=center_node.item(), num_hops=block_radius,
            edge_index=data.edge_index, relabel_nodes=False, num_nodes=num_nodes)
        current_unknown_nodes.update(block_nodes.tolist())
        attempts += 1

    if current_unknown_nodes:
        unknown_indices = torch.tensor(list(current_unknown_nodes), dtype=torch.long, device=mask.device)
        mask[unknown_indices] = 0.0
    return mask

# ==============================================================================
# 2. 模型定义
# ==============================================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model): super().__init__(); self.d_model = d_model
    def forward(self, t):
        t, device, half = t.flatten(), t.device, self.d_model // 2
        emb = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        out = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if out.size(1) < self.d_model: out = F.pad(out, (0, self.d_model - out.size(1)))
        return out

class AdaLN(nn.Module):
    def __init__(self,d_model,d_cond): super().__init__(); self.layer_norm=nn.LayerNorm(d_model,elementwise_affine=False); self.cond_proj=nn.Sequential(nn.SiLU(),nn.Linear(d_cond,2*d_model))
    def forward(self,x,cond): x_norm=self.layer_norm(x); style,shift=self.cond_proj(cond).chunk(2,dim=1); return x_norm*(1+style)+shift

class SimpleGNNLayer(nn.Module):
    def __init__(self,d_model): super().__init__(); self.conv=GCNConv(d_model,d_model); self.adaln=AdaLN(d_model,d_model*2)
    def forward(self,x,edge_index,t_emb,a_emb): t_emb_expanded=t_emb.expand(x.size(0),-1); cond_emb=torch.cat([t_emb_expanded,a_emb],dim=1); x_mod=self.adaln(x,cond_emb); out=F.relu(self.conv(x_mod,edge_index)); return x+out

class PureConsistencyGNN(nn.Module):
    def __init__(self, F_in=2, F_a=3, d_model=64, num_layers=4, F_out=1, pool_ratio=0.5):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.input_proj = nn.Linear(F_in, d_model)
        self.cond_a_proj = nn.Linear(F_a, d_model)
        self.layers = nn.ModuleList([SimpleGNNLayer(d_model) for _ in range(num_layers)])
        self.output_proj = nn.Linear(d_model, F_out)
        self.pool = TopKPooling(in_channels=F_out, ratio=pool_ratio)

    def forward(self, x_t, edge_index, t, cond_a):
        if not isinstance(t, torch.Tensor): t = torch.tensor([t], device=x_t.device)
        t = t.view(-1, 1); t_emb = self.time_emb(t); a_emb = self.cond_a_proj(cond_a)
        h = self.input_proj(x_t)
        for lyr in self.layers: h = lyr(h, edge_index, t_emb, a_emb)
        return self.output_proj(h)

    def apply_pooling(self, x_pred, edge_index, batch):
        pooled_x, _, _, pooled_batch, _, _ = self.pool(x_pred, edge_index, batch=batch)
        return pooled_x, pooled_batch

# ==============================================================================
# 3. 采样与辅助函数
# ==============================================================================
def sample_log_uniform(batch, t_min, t_max, device):
    u = torch.rand(batch, device=device); log_min, log_max = math.log(t_min), math.log(t_max)
    return torch.exp(log_min + u * (log_max - log_min))
def make_xt_from_x0(x0, t, eps): return x0 + t.view(-1, 1) * eps
def edm_coeffs(t, sigma_data):
    t = t.view(-1, 1)
    denom = (t**2 + sigma_data**2)
    c_skip = sigma_data**2 / denom; c_out = (t * sigma_data) / torch.sqrt(denom); c_in = 1.0 / torch.sqrt(denom)
    return c_in, c_out, c_skip
def model_x0_pred_inpainting(model, x_t_combined, edge_index, t, sigma_data, cond_a):
    x_t_pressure, mask = x_t_combined[:, 0:1], x_t_combined[:, 1:2]
    if not isinstance(t, torch.Tensor): t = torch.tensor([t], device=x_t_combined.device)
    c_in, c_out, c_skip = edm_coeffs(t, sigma_data)
    x_in_pressure = c_in * x_t_pressure
    model_input = torch.cat([x_in_pressure, mask], dim=1)
    F_pred = model(model_input, edge_index, t, cond_a)
    x0_pred_pressure = c_skip * x_t_pressure + c_out * F_pred
    return x0_pred_pressure

# ==============================================================================
# 4. MODIFIED: 训练与评估函数
# ==============================================================================
def train_one_epoch(model, dataloader, optimizer, epoch, epochs, t_min, t_max, batch_pairs, hcl_lambda, device):
    model.train(); total_epoch_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in pbar:
        batch = batch.to(device)
        b0_norm, a_norm, edge_index = batch.y_node, batch.x_static, batch.edge_index
        optimizer.zero_grad(); total_batch_loss = 0
        for _ in range(batch_pairs):
            # --- NEW: 使用结构化掩码 ---
            mask_list = []
            for single_graph_data in batch.to_data_list():
                # 对于训练，使用相对温和的掩码设置
                single_mask = create_structural_mask(
                    single_graph_data, num_blocks=2, block_radius=2, known_ratio_range=(0.4, 0.8))
                mask_list.append(single_mask)
            mask = torch.cat(mask_list, dim=0)
            # --- END NEW ---

            eps = torch.randn_like(b0_norm)
            t1 = sample_log_uniform(batch.num_graphs, t_min, t_max, device)[batch.batch]
            t2 = sample_log_uniform(batch.num_graphs, t_min, t_max, device)[batch.batch]
            xt1_noisy, xt2_noisy = make_xt_from_x0(b0_norm, t1, eps), make_xt_from_x0(b0_norm, t2, eps)
            xt1_pressure, xt2_pressure = torch.where(mask.bool(), b0_norm, xt1_noisy), torch.where(mask.bool(), b0_norm, xt2_noisy)
            xt1_combined, xt2_combined = torch.cat([xt1_pressure, mask], dim=1), torch.cat([xt2_pressure, mask], dim=1)
            x0_pred1 = model_x0_pred_inpainting(model, xt1_combined, edge_index, t1, 1.0, a_norm)
            x0_pred2 = model_x0_pred_inpainting(model, xt2_combined, edge_index, t2, 1.0, a_norm)
            
            loss_fine = F.mse_loss(x0_pred1, x0_pred2)
            loss_coarse = 0.0
            if hcl_lambda > 0:
                x0_pred1_pooled, _ = model.apply_pooling(x0_pred1, edge_index, batch.batch)
                x0_pred2_pooled, _ = model.apply_pooling(x0_pred2, edge_index, batch.batch)
                if x0_pred1_pooled.numel() > 0:
                    loss_coarse = F.mse_loss(x0_pred1_pooled, x0_pred2_pooled)
            
            loss_pair = loss_fine + hcl_lambda * loss_coarse
            total_batch_loss += loss_pair

        avg_batch_loss = total_batch_loss / batch_pairs
        avg_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_epoch_loss += avg_batch_loss.item()
        pbar.set_postfix(loss=avg_batch_loss.item())
    avg_epoch_loss = total_epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} Avg Train Loss: {avg_epoch_loss:.6e}")
    return avg_epoch_loss

def evaluate(model, test_loader, t_max, save_path, device):
    print("\nStarting evaluation on the test set with STRUCTURAL MASKING...")
    model.load_state_dict(torch.load(save_path)); model.eval()
    all_true_unknown, all_preds_unknown, all_unknown_errors = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Inpainting (Structural Mask)"):
            batch = batch.to(device)
            b0_norm, a_norm, edge_index = batch.y_node, batch.x_static, batch.edge_index
            
            # --- NEW: 使用固定的、更具挑战性的结构化掩码进行评估 ---
            mask_list = []
            for single_graph_data in batch.to_data_list():
                # 固定已知比例为30%，区块更大，更有挑战性
                single_mask = create_structural_mask(
                    single_graph_data, num_blocks=3, block_radius=3, known_ratio_range=(0.1, 0.3))
                mask_list.append(single_mask)
            mask = torch.cat(mask_list, dim=0)
            # --- END NEW ---

            unknown_mask_bool = (mask < 0.5).squeeze()
            if not unknown_mask_bool.any(): continue
            
            initial_noise = torch.randn_like(b0_norm) * t_max
            x_T_pressure = torch.where(mask.bool(), b0_norm, initial_noise)
            x_T_combined = torch.cat([x_T_pressure, mask], dim=1)
            t_gen_expanded = torch.full((b0_norm.size(0),), t_max, device=device)
            preds_norm = model_x0_pred_inpainting(model, x_T_combined, edge_index, t_gen_expanded, 1.0, a_norm)

            true_unknown, preds_unknown = b0_norm[unknown_mask_bool], preds_norm[unknown_mask_bool]
            all_true_unknown.append(true_unknown.cpu())
            all_preds_unknown.append(preds_unknown.cpu())
            all_unknown_errors.append((preds_unknown - true_unknown).cpu())

    all_true_tensor = torch.cat(all_true_unknown)
    all_preds_tensor = torch.cat(all_preds_unknown)
    all_errors_tensor = torch.cat(all_unknown_errors)

    final_r2 = r2_score(all_true_tensor.numpy(), all_preds_tensor.numpy())
    rmse = torch.sqrt(torch.mean(all_errors_tensor**2)).item()
    mae = torch.mean(torch.abs(all_errors_tensor)).item()
    mean_error = torch.mean(all_errors_tensor).item()

    print(f"\n--- Evaluation Results on Unknown Nodes (Structural Mask) ---")
    print(f"R² Score: {final_r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Mean Error (ME): {mean_error:.4f}  <-- Key metric for structural drift!")
    return final_r2, all_true_tensor, all_preds_tensor

# ==============================================================================
# 5. 主程序
# ==============================================================================
if __name__ == '__main__':
    # --- CHOOSE RUN MODE FOR ABLATION STUDY ---
    # 'HCM': Train the full Hierarchical Consistency Model
    # 'Baseline': Train the model without the hierarchical loss (hcl_lambda=0)
    run_mode = 'HCM' 
    
    print(f"--- Running in {run_mode} mode ---")

    # --- a) 数据加载 ---
    # 使用 MOCK 加载器进行演示。请用你的真实加载器替换。
    # train_loader, val_loader, test_loader, _, _ = (
    #     "Dummy", "path", 72, 16, mask_ratio=0.2)
    train_loader, val_loader, test_loader, _, _ = load_data("PretrainDataset" , "/data/zsc/Pipeline/data/epaNet/L-TOWN.inp" , 72 , 16 , mask_ratio=0.2)


    # --- b) 模型、优化器和超参数设置 ---
    model = PureConsistencyGNN(
        F_in=2, F_a=3, d_model=128, num_layers=5, F_out=1, pool_ratio=0.75
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    
    epochs = 50 # 在新任务上，可以先从较少的epoch开始
    t_min, t_max = 0.002, 80.0
    batch_pairs = 2
    save_path = "model/L-town_baseline.pt"
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    
    # Configure based on run_mode
    if run_mode == 'HCM':
        hcl_lambda = 0.2  # 宏观损失权重
    elif run_mode == 'Baseline':
        hcl_lambda = 0.0  # 关闭宏观损失
    else:
        raise ValueError("Invalid run_mode. Choose 'HCM' or 'Baseline'.")
        
    best_loss = float('inf')
    
    
    # Optional: Load pre-trained model to continue training
    # if os.path.exists(save_path):
    #     print(f"Loading pre-trained model from {save_path}")
    #     try: model.load_state_dict(torch.load(save_path))
    #     except Exception as e: print(f"Could not load model weights: {e}")

    # --- c) 训练循环 ---
    for epoch in range(epochs):
        avg_epoch_loss = train_one_epoch(
            model, train_loader, optimizer, epoch, epochs, 
            t_min, t_max, batch_pairs, hcl_lambda, device)
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved with loss {best_loss:.6e}")
            
    # --- d) 评估与可视化 ---
    print(f"\n--- FINAL EVALUATION FOR {run_mode} ---")
    final_r2, _, _ = evaluate(model, test_loader, t_max, save_path, device)
    
    print(f"\n{run_mode} run completed. Final R² on test set (structural mask): {final_r2:.4f}")
    print("\nNext step: Run again with the other mode and compare the evaluation metrics.")