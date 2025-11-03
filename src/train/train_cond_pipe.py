# real_task_cm_gnn.py
"""
Conditional Consistency Model adapted for a real-world water network task.
- Task: Given static node features (elevation, demand, type), generate node pressure.
- The training loop is adapted to handle epochs and batches from a DataLoader.
- Evaluation and visualization are updated for the new dataset.
"""
import math, random, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import networkx as nx
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *


# --- Reproducibility & Device Setup ---
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


# ==============================================================================
# 1. 模型定义 (与之前版本基本一致)
# ==============================================================================
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

# ==============================================================================
# 2. 采样工具函数 (与之前版本一致)
# ==============================================================================
def sample_log_uniform(batch, t_min, t_max, device):
    u = torch.rand(batch, device=device)
    log_min, log_max = math.log(t_min), math.log(t_max)
    return torch.exp(log_min + u * (log_max - log_min))

def make_xt_from_x0(x0, t, eps):
    # Ensure t is broadcastable to x0
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


# ==============================================================================
# 4. 主程序: 训练、评估、可视化
# ==============================================================================
def train_one_epoch(model, dataloader, optimizer, t_min, t_max, device):
    model.train()
    total_epoch_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        batch = batch.to(device)
        # 从batch中提取条件和目标
        b0_norm = batch.y_node
        a0_norm = batch.x_static
        edge_index = batch.edge_index
        
        optimizer.zero_grad()
        total_batch_loss = 0
        
        for _ in range(batch_pairs):
            # t的形状需要匹配batch中图的数量
            num_graphs_in_batch = batch.num_graphs
            t1 = sample_log_uniform(num_graphs_in_batch, t_min, t_max, device)
            t2 = sample_log_uniform(num_graphs_in_batch, t_min, t_max, device)
            
            # 为batch中的每个节点生成噪声
            eps = torch.randn_like(b0_norm)
            
            # t需要扩展以匹配每个节点的形状
            # t_expanded: [num_graphs] -> [num_nodes_in_batch]
            t1_expanded = t1[batch.batch]
            t2_expanded = t2[batch.batch]
            
            x_t1 = make_xt_from_x0(b0_norm, t1_expanded, eps)
            x_t2 = make_xt_from_x0(b0_norm, t2_expanded, eps)

            x0_pred1 = model_x0_pred(model, x_t1, edge_index, t1_expanded, sigma_data=1.0, cond_a=a0_norm)
            x0_pred2 = model_x0_pred(model, x_t2, edge_index, t2_expanded, sigma_data=1.0, cond_a=a0_norm)
            
            loss_pair = F.mse_loss(x0_pred1, x0_pred2)
            total_batch_loss += loss_pair

        avg_batch_loss = total_batch_loss / batch_pairs
        avg_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_epoch_loss += avg_batch_loss.item()
        
    avg_epoch_loss = total_epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} Avg Loss: {avg_epoch_loss:.6e}")
    return avg_epoch_loss

def evaluate(model, test_loader, t_max, device):
    print("\nStarting evaluation on the test set...")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    all_true = []
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)
            b0_norm = batch.y_node
            a0_norm = batch.x_static
            edge_index = batch.edge_index
            print(f"static: {a0_norm[:10]}\n true: {b0_norm[:10]}")
            
            # 从纯噪声开始生成
            initial_noise = torch.randn_like(b0_norm) * t_max
            t_gen = torch.full((batch.num_graphs,), t_max, device=device)
            t_gen_expanded = t_gen[batch.batch]
            
            preds_norm = model_x0_pred(model, initial_noise, edge_index, t_gen_expanded, sigma_data=1.0, cond_a=a0_norm)
            
            all_true.append(b0_norm.cpu())
            all_preds.append(preds_norm.cpu())
            
    all_true_tensor = torch.cat(all_true, dim=0)
    all_preds_tensor = torch.cat(all_preds, dim=0)
    
    final_r2 = r2_score(all_true_tensor.numpy(), all_preds_tensor.numpy())
    print(f"\nFinal Test R² Score: {final_r2:.4f}")
    return final_r2 , all_true_tensor , all_preds_tensor

def visualize_progress(model,save_path, dataloader,r2_score, all_true_tensor, all_preds_tensor, device):
    plt.figure(figsize=(8, 6))
    plt.scatter(all_true_tensor.numpy(), all_preds_tensor.numpy(), alpha=0.1)
    min_val = min(all_true_tensor.min(), all_preds_tensor.min())
    max_val = max(all_true_tensor.max(), all_preds_tensor.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel("True Normalized Pressure")
    plt.ylabel("Predicted Normalized Pressure")
    plt.title(f"Overall Test Set Performance (R²={r2_score:.4f})")
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("real_task_cm_scatter.png")
    print("Saved scatter plot: real_task_cm_scatter.png")

    # 2. 单个样本的图可视化
    test_sample = dataloader.dataset[0].to(device)
    G_vis = nx.from_edgelist(test_sample.edge_index.t().cpu().numpy())
    pos = nx.spring_layout(G_vis, seed=seed)

    with torch.no_grad():
        b0_sample = test_sample.y_node
        a0_sample = test_sample.x_static
        noise_sample = torch.randn_like(b0_sample) * t_max
        t_sample = torch.tensor([t_max], device=device)
        pred_sample = model_x0_pred(model, noise_sample, test_sample.edge_index, t_sample, sigma_data=1.0, cond_a=a0_sample)

    # 可视化条件中的"高程"
    cond_vals = a0_sample[:, 0].cpu().numpy()
    true_vals = b0_sample.cpu().numpy().flatten()
    pred_vals = pred_sample.cpu().numpy().flatten()
    
    plt.figure(figsize=(20, 5))
    vmin = min(true_vals.min(), pred_vals.min())
    vmax = max(true_vals.max(), pred_vals.max())
    
    pred_vmin = pred_vals.min()
    pred_vmax = pred_vals.max()

    plt.subplot(1, 3, 1)
    nx.draw(G_vis, pos, with_labels=False, node_size=50, node_color=cond_vals, cmap=plt.cm.viridis)
    plt.title("Condition: Node Elevation")
    
    plt.subplot(1, 3, 2)
    nx.draw(G_vis, pos, with_labels=False, node_size=50, node_color=true_vals, cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax)
    plt.title("Ground Truth: Node Pressure")
    sm_a = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=pred_vmin, vmax=pred_vmax))
    sm_a.set_array([])
    plt.colorbar(sm_a, ax=plt.gca(), orientation='horizontal', pad=0.05)

    plt.subplot(1, 3, 3)
    nx.draw(G_vis, pos, with_labels=False, node_size=50, node_color=pred_vals, cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax)
    plt.title(f"Generated Node Pressure")
    sm_b = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm_b.set_array([])
    plt.colorbar(sm_b, ax=plt.gca(), orientation='horizontal', pad=0.05)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved graph visualization: {save_path}")

if __name__ == '__main__':
    # --- a) 数据加载 ---
    train_loader, val_loader, test_loader , pressure_norm , flow_norm = load_data("PretrainDataset" , "/data/zsc/Pipeline/data/epaNet/Anytown.inp" , 72 , 16 , mask_ratio=0.2)

    # --- b) 模型、优化器和超参数设置 ---
    # ** 核心适配 **: F_a=3, 因为 x_static 是 [高程, 需水量, 类型]
    model = PureConsistencyGNN(F_in=1, F_a=3, d_model=64, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)
    
    epochs = 20
    t_min = 0.002
    t_max = 80.0
    batch_pairs = 2 # 每个batch内采样多少对 (t1, t2)
    
    best_loss = float('inf')
    save_path = "model/tt_cm.pt"
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    
    # --- c) 训练循环 ---
    for epoch in range(epochs):
        avg_epoch_loss = train_one_epoch(model, train_loader, optimizer, t_min, t_max, device)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with loss {best_loss:.6e}")
            
    final_r2 , all_true_tensor , all_preds_tensor = evaluate(model, test_loader, t_max, device)
    visualize_progress(model,"figures/cm_cond.png", test_loader, final_r2, all_true_tensor, all_preds_tensor, device)