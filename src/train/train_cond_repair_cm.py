# real_task_cm_inpainting_strict_FIXED.py
"""
Conditional Consistency Model for Inpainting, strictly following the self-consistency loss.
- Task: Given static node features and some known pressure values, generate the full pressure field.
- Model Input: [pressure, mask]
- Loss Function: Pure self-consistency loss: F.mse_loss(x0_pred1, x0_pred2)
- FIX: Corrected a bug in SinusoidalTimeEmbedding that caused it to return None.
"""
import math, random, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch.utils.data import Subset
import networkx as nx
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import * # Assuming this exists in your project

# --- MOCK UTILS FOR STANDALONE EXECUTION ---
def select_random_indices(total_num, num_to_select, excluded_indices, seed=None):
    if seed is not None: rng = np.random.default_rng(seed)
    else: rng = np.random.default_rng()
    all_indices = np.arange(total_num)
    eligible_indices = np.setdiff1d(all_indices, excluded_indices.cpu().numpy(), assume_unique=True)
    if num_to_select > len(eligible_indices): num_to_select = len(eligible_indices)
    selected = rng.choice(eligible_indices, num_to_select, replace=False)
    return torch.from_numpy(selected).long()

# Mock your data loading function
def load_data_mock(dataset_name, path, timesteps, batch_size, mask_ratio):
    print("--- Using MOCK data loader ---")
    class ZScoreNormalizer:
        def __init__(self): self.mean=0; self.std=1
        def fit(self,x): self.mean=x.mean(0); self.std=x.std(0); self.std[self.std==0]=1
        def transform(self,x): return (x-self.mean)/self.std
    class DummyDataset(Dataset):
        def __init__(self, num_graphs=200):
            super().__init__()
            self.num_graphs = num_graphs
            G = nx.gnm_random_graph(50, 100, seed=42); self.edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
            self.edge_index = torch.cat([self.edge_index, self.edge_index.flip(0)], dim=1)
            self.x_static = torch.randn(50, 3)
            self.node_type = torch.zeros(50, 1); self.node_type[0]=1
            self.y_node = torch.randn(num_graphs, 50, 1)
        def len(self): return self.num_graphs
        def get(self, idx):
            return Data(x_static=self.x_static, y_node=self.y_node[idx], edge_index=self.edge_index, node_type=self.node_type, num_nodes=50)
    
    full_dataset = DummyDataset()
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader, ZScoreNormalizer(), ZScoreNormalizer()
# --- END MOCK ---


# --- Reproducibility & Device Setup ---
seed = 42; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


# ==============================================================================
# 1. 模型定义 (FIXED SinusoidalTimeEmbedding)
# ==============================================================================
class SinusoidalTimeEmbedding(nn.Module):
    # *****************************
    # ******** FIXED BLOCK ********
    # *****************************
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
        
        return out # This return statement is now outside the if block
    # *****************************
    # ******* END FIXED BLOCK *****
    # *****************************

class AdaLN(nn.Module):
    def __init__(self,d_model,d_cond): super().__init__(); self.layer_norm=nn.LayerNorm(d_model,elementwise_affine=False); self.cond_proj=nn.Sequential(nn.SiLU(),nn.Linear(d_cond,2*d_model))
    def forward(self,x,cond): x_norm=self.layer_norm(x); style,shift=self.cond_proj(cond).chunk(2,dim=1); return x_norm*(1+style)+shift

class SimpleGNNLayer(nn.Module):
    def __init__(self,d_model): super().__init__(); self.conv=GCNConv(d_model,d_model); self.adaln=AdaLN(d_model,d_model*2)
    def forward(self,x,edge_index,t_emb,a_emb): t_emb_expanded=t_emb.expand(x.size(0),-1); cond_emb=torch.cat([t_emb_expanded,a_emb],dim=1); x_mod=self.adaln(x,cond_emb); out=F.relu(self.conv(x_mod,edge_index)); return x+out

class PureConsistencyGNN(nn.Module):
    def __init__(self, F_in=2, F_a=3, d_model=64, num_layers=3, F_out=1):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.input_proj = nn.Linear(F_in, d_model)
        self.cond_a_proj = nn.Linear(F_a, d_model)
        self.layers = nn.ModuleList([SimpleGNNLayer(d_model) for _ in range(num_layers)])
        self.output_proj = nn.Linear(d_model, F_out)
    def forward(self, x_t, edge_index, t, cond_a):
        if not isinstance(t, torch.Tensor): t = torch.tensor([t], device=x_t.device)
        t = t.view(-1, 1); t_emb = self.time_emb(t); a_emb = self.cond_a_proj(cond_a)
        h = self.input_proj(x_t)
        for lyr in self.layers: h = lyr(h, edge_index, t_emb, a_emb)
        return self.output_proj(h)

# ... The rest of your code (sampling, training, eval, main) remains the same ...
# I am re-pasting it for completeness.

# ==============================================================================
# 2. 采样工具函数
# ==============================================================================
def sample_log_uniform(batch, t_min, t_max, device):
    u = torch.rand(batch, device=device); log_min, log_max = math.log(t_min), math.log(t_max)
    return torch.exp(log_min + u * (log_max - log_min))
def make_xt_from_x0(x0, t, eps):
    return x0 + t.view(-1, 1) * eps
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
# 3. 训练与评估函数
# ==============================================================================
def train_one_epoch(model, dataloader, optimizer, epoch, epochs, t_min, t_max, batch_pairs, device):
    model.train(); total_epoch_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in pbar:
        batch = batch.to(device)
        b0_norm, a0_norm, edge_index = batch.y_node, batch.x_static, batch.edge_index
        optimizer.zero_grad(); total_batch_loss = 0
        for _ in range(batch_pairs):
            mask = torch.zeros_like(b0_norm)
            reservoir_indices = torch.where(batch.node_type.flatten() == 1)[0]
            num_masked_nodes = int(batch.num_nodes * random.uniform(0.1, 0.5))
            known_indices = select_random_indices(batch.num_nodes, num_masked_nodes, reservoir_indices)
            if known_indices.numel() > 0: mask[known_indices] = 1.0
            num_graphs_in_batch = batch.num_graphs
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
            x0_pred1 = model_x0_pred_inpainting(model, xt1_combined, edge_index, t1_expanded, 1.0, a0_norm)
            x0_pred2 = model_x0_pred_inpainting(model, xt2_combined, edge_index, t2_expanded, 1.0, a0_norm)
            loss_pair = F.mse_loss(x0_pred1, x0_pred2)
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
    print("\nStarting evaluation on the test set for inpainting...")
    model.load_state_dict(torch.load(save_path)); model.eval()
    all_true_unknown, all_preds_unknown = [], []
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
            preds_norm = model_x0_pred_inpainting(model, x_T_combined, edge_index, t_gen_expanded, 1.0, a0_norm)
            all_true_unknown.append(b0_norm[unknown_mask].cpu())
            all_preds_unknown.append(preds_norm[unknown_mask].cpu())
    all_true_tensor = torch.cat(all_true_unknown); all_preds_tensor = torch.cat(all_preds_unknown)
    final_r2 = r2_score(all_true_tensor.numpy(), all_preds_tensor.numpy())
    print(f"\nFinal Inpainting R² Score (on unknown nodes): {final_r2:.4f}")
    return final_r2, all_true_tensor, all_preds_tensor

def visualize_progress(model, save_path_fig, dataloader, r2_score, t_max, device):
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
        preds_norm = model_x0_pred_inpainting(model, x_T_combined, edge_index, t_gen_expanded, 1.0, a0_norm)
    G_vis = nx.from_edgelist(edge_index.t().cpu().numpy())
    pos = nx.spring_layout(G_vis, seed=seed)
    true_vals = b0_norm.cpu().numpy().flatten(); pred_vals = preds_norm.cpu().numpy().flatten()
    vmin, vmax = min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    nx.draw(G_vis, pos, with_labels=False, node_size=80, node_color=true_vals, cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax)
    nx.draw_networkx_nodes(G_vis, pos, nodelist=known_indices.cpu().tolist(), node_color='lime', edgecolors='black', node_size=100)
    plt.title("Ground Truth (Known nodes in green)")
    plt.subplot(1, 3, 2)
    init_vals = x_T_pressure.cpu().numpy().flatten()
    nx.draw(G_vis, pos, with_labels=False, node_size=80, node_color=init_vals, cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax)
    nx.draw_networkx_nodes(G_vis, pos, nodelist=known_indices.cpu().tolist(), node_color='lime', edgecolors='black', node_size=100)
    plt.title("Initial State (Known + Noise)")
    plt.subplot(1, 3, 3)
    nx.draw(G_vis, pos, with_labels=False, node_size=80, node_color=pred_vals, cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax)
    nx.draw_networkx_nodes(G_vis, pos, nodelist=known_indices.cpu().tolist(), node_color='lime', edgecolors='black', node_size=100)
    plt.title(f"Inpainting Result (R² on unknown={r2_score:.4f})")
    plt.tight_layout(); plt.savefig(save_path_fig); print(f"Saved graph visualization: {save_path_fig}")
    
    
def visualize_uncertainty_band(mu_pred, true_vals, save_path_uncertainty):
    """
    绘制预测均值 μ 与不确定性区间 (μ ± σ) 的可视化。
    纵轴: 节点压力（或其他量）
    横轴: 节点索引 (可换成物理位置序号)
    """
    mu_pred = mu_pred.cpu().numpy().flatten()
    # std_pred = np.sqrt(var_tilde.cpu().numpy().flatten())
    true_vals = true_vals.cpu().numpy().flatten()

    idx = np.arange(len(mu_pred))

    plt.figure(figsize=(10, 5))
    plt.plot(idx, true_vals, 'k-', lw=2, label='Ground Truth')
    plt.plot(idx, mu_pred, 'b--', lw=2, label='Predicted Mean (μ)')
    # plt.fill_between(idx,
    #                  mu_pred - std_pred,
    #                  mu_pred + std_pred,
    #                  color='blue', alpha=0.2, label='±1σ Uncertainty Band')
    plt.xlabel("Node index")
    plt.ylabel("Normalized Pressure")
    plt.legend()
    plt.title("Predicted Mean and Uncertainty Interval")
    plt.tight_layout()
    plt.savefig(save_path_uncertainty)
    print(f"Saved uncertainty visualization: {save_path_uncertainty}")

# ==============================================================================
# 5. 主程序
# ==============================================================================
if __name__ == '__main__':
    # --- a) 数据加载 ---
    # 使用 MOCK 加载器进行演示。请用你的真实加载器替换。
    # from utils import load_data
    # wds_list = ["/data/zsc/Pipeline/data/epaNet/Anytown.inp",
    #             "/data/zsc/Pipeline/data/epaNet/CTOWN.INP",
    #             "/data/zsc/Pipeline/data/epaNet/L-TOWN.inp",
    #             "/data/zsc/Pipeline/data/epaNet/EXN.inp",
    #             "/data/zsc/Pipeline/data/epaNet/EPANET/example-networks/Net3.inp"]
    # raw_data_list = []
    # for wd in wds_list:
    #     epanet = EpytHelper(wd , hrs=72)
    #     raw_data = epanet.get_raw_data()
    #     raw_data_list.cat(raw_data)
    #     epanet.destroy()
    # pressure_norm = ZScoreNormalizer(); flow_norm = ZScoreNormalizer()
    # dataset = PretrainDataset(raw_data_list, fit_ratio=0.8, fit_node_mask_ratio=0, pressure_norm=pressure_norm, flow_norm=flow_norm, fit_pipe_mask_ratio=0)
    # train_loader, val_loader, test_loader = dataset.gen_train_loader(train_ratio=0.8, val_ratio=0.1, batch_size=1, shuffle=True)
    train_loader, val_loader, test_loader, _, _ = load_data("PretrainDataset" , "/data/zsc/Pipeline/data/epaNet/shiqi.inp" , 72 , 16 , mask_ratio=0.2)

    # --- b) 模型、优化器和超参数设置 ---
    model = PureConsistencyGNN(F_in=2, F_a=3, d_model=64, num_layers=4, F_out=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)
    epochs = 800; t_min = 0.002; t_max = 80.0; batch_pairs = 2
    best_loss = float('inf')
    save_path = "model/real_inpainting_cm_strict.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        print(f"Loading pre-trained model from {save_path}")
        try: model.load_state_dict(torch.load(save_path))
        except Exception as e: print(f"Could not load model weights: {e}")

    # --- c) 训练循环 ---
    for epoch in range(epochs):
        avg_epoch_loss = train_one_epoch(model, train_loader, optimizer, epoch, epochs, t_min, t_max, batch_pairs, device)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved with loss {best_loss:.6e}")
            
    # --- d) 评估与可视化 ---
    final_r2, all_true_tensor, all_pred_tensor = evaluate(model, test_loader, t_max, save_path, device)
    # visualize_uncertainty_band(all_pred_tensor, all_true_tensor, "figures/cm_inpainting_strict_uncertainty.png")
    print(f"\nEvaluation completed. Final R² on test set: {final_r2:.4f}")
    # visualize_progress(model, "figures/cm_inpainting_strict.png", test_loader, final_r2, t_max, device)