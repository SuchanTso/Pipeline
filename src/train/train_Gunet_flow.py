import math
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, TopKPooling, global_mean_pool
from torch.utils.data import Subset
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm
from utils import load_data , load_multigraph_data


# ==============================================================================
# 0. 数据加载与预处理 (Mock & Real)
# ==============================================================================
class ZScoreNormalizer:
    def __init__(self): self.mean=0; self.std=1
    def fit(self,x): self.mean=x.mean(0, keepdim=True); self.std=x.std(0, keepdim=True); self.std[self.std==0]=1
    def transform(self,x): return (x-self.mean)/self.std
    def inverse_transform(self,x): return x*self.std+self.mean

def select_random_indices(total_num, num_to_select, excluded_indices, seed=None):
    if seed is not None: rng = np.random.default_rng(seed)
    else: rng = np.random.default_rng()
    # Ensure excluded_indices is a 1D array for setdiff1d
    excluded_indices = np.array(excluded_indices).flatten()
    all_indices = np.arange(total_num)
    eligible_indices = np.setdiff1d(all_indices, excluded_indices, assume_unique=True)
    if num_to_select > len(eligible_indices): num_to_select = len(eligible_indices)
    if len(eligible_indices) == 0: return torch.tensor([], dtype=torch.long)
    selected = rng.choice(eligible_indices, num_to_select, replace=False)
    return torch.from_numpy(selected).long()

# ==============================================================================
# 1. 模型定义
# ==============================================================================
# --- 1.1 辅助模块 (保持不变) ---
class SinusoidalTimeEmbedding(nn.Module):
    # ... (你的 SinusoidalTimeEmbedding 代码) ...
    def __init__(self, d_model):
        super().__init__(); self.d_model = d_model
    def forward(self, t):
        t = t.flatten(); device = t.device; half = self.d_model // 2
        emb = math.log(10000) / max(half - 1, 1); emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0); out = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if out.size(1) < self.d_model: out = F.pad(out, (0, self.d_model - out.size(1)))
        return out

class AdaGN(nn.Module):
    # ... (你的 AdaGN 代码) ...
    def __init__(self, d_model, d_cond, num_groups=4):
        super().__init__(); self.group_norm = nn.GroupNorm(num_groups, d_model, affine=False)
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(d_cond, 2 * d_model))
    def forward(self, x, cond):
        x_norm = self.group_norm(x); style, shift = self.cond_proj(cond).chunk(2, dim=1); return x_norm * (1 + style) + shift

class GINEBlock(nn.Module):
    # ... (你的 GINEBlock 代码) ...
    def __init__(self, d_in, d_model, d_cond, d_out=None):
        super().__init__()
        d_out = d_out or d_in
        self.proj_in = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()
        self.conv1 = GINEConv(nn.Sequential(nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)))
        self.conv2 = GINEConv(nn.Sequential(nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)))
        self.norm1 = AdaGN(d_model, d_cond)
        self.norm2 = AdaGN(d_model, d_cond)
        self.res_proj = nn.Linear(d_model, d_out)
        self.input_res_proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x, edge_index, edge_attr, cond):
        x_res = self.input_res_proj(x)
        h = self.proj_in(x)
        h_norm1 = self.norm1(h, cond)
        h = self.conv1(h_norm1, edge_index, edge_attr)
        h = F.silu(h)
        h_norm2 = self.norm2(h, cond)
        h = self.conv2(h_norm2, edge_index, edge_attr)
        h = F.silu(h)
        h_out = self.res_proj(h)
        return x_res + h_out

# --- 1.2 流量预测模型 [核心修改] ---
class WaterGUNet_Flow(nn.Module):
    def __init__(self, d_node_in, d_edge_in, d_model=64, d_time_emb=64, pool_ratios=[0.8, 0.8]):
        super().__init__()
        self.d_model = d_model
        self.time_emb = SinusoidalTimeEmbedding(d_time_emb)
        self.node_encoder = nn.Linear(d_node_in, d_model)
        self.edge_encoder = nn.Linear(d_edge_in, d_model)
        d_cond = d_time_emb + d_model
        
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        for ratio in pool_ratios:
            self.encoder_blocks.append(GINEBlock(d_model, d_model, d_cond))
            self.pools.append(TopKPooling(d_model, ratio=ratio))
        self.bottleneck = GINEBlock(d_model, d_model, d_cond)
        self.decoder_blocks = nn.ModuleList()
        for _ in pool_ratios:
            self.decoder_blocks.append(GINEBlock(2 * d_model, d_model, d_cond, d_out=d_model))
            
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * 2 + d_model, d_model), # h_u || h_v || edge_attr_init
            nn.SiLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x_node_in, edge_attr_in, edge_index, batch_indices_in, t_per_graph):
        h_node_init = self.node_encoder(x_node_in) 
        edge_attr_init = self.edge_encoder(edge_attr_in)
        t_emb = self.time_emb(t_per_graph)
        
        # U-Net Body
        h = h_node_init
        edge_index_current, edge_attr_current, batch_indices_current = edge_index, edge_attr_init, batch_indices_in
        pre_pool_states = []
        for block, pool in zip(self.encoder_blocks, self.pools):
            global_h = global_mean_pool(h, batch_indices_current)
            cond = torch.cat([t_emb, global_h], dim=1)[batch_indices_current]
            h = block(h, edge_index_current, edge_attr_current, cond)
            pre_pool_states.append({'skip_h': h, 'edge_index': edge_index_current, 'edge_attr': edge_attr_current, 'batch': batch_indices_current})
            h, edge_index_current, edge_attr_current, batch_indices_current, perm, _ = pool(h, edge_index_current, edge_attr_current, batch_indices_current)
            pre_pool_states[-1]['perm'] = perm
        
        global_h_bottle = global_mean_pool(h, batch_indices_current)
        cond_bottle = torch.cat([t_emb, global_h_bottle], dim=1)[batch_indices_current]
        h = self.bottleneck(h, edge_index_current, edge_attr_current, cond_bottle)
        
        for i, block in enumerate(self.decoder_blocks):
            idx = len(self.decoder_blocks) - 1 - i
            state = pre_pool_states[idx]
            upsampled_h = torch.zeros_like(state['skip_h']); upsampled_h[state['perm']] = h
            edge_index_current, edge_attr_current, batch_indices_current = state['edge_index'], state['edge_attr'], state['batch']
            global_h = global_mean_pool(state['skip_h'], batch_indices_current)
            cond = torch.cat([t_emb, global_h], dim=1)[batch_indices_current]
            h = torch.cat([upsampled_h, state['skip_h']], dim=1)
            h = block(h, edge_index_current, edge_attr_current, cond)

        # Output Reconstruction
        src, dst = edge_index[0], edge_index[1]
        output_features = torch.cat([h[src], h[dst], edge_attr_init], dim=1)
        out_flow = self.output_proj(output_features)
        return out_flow

# ==============================================================================
# 2. CM 包裹函数 [核心修改]
# ==============================================================================
def sample_log_uniform(batch_size, t_min, t_max, device):
    u = torch.rand(batch_size, device=device)
    log_min, log_max = math.log(t_min), math.log(t_max)
    return torch.exp(log_min + u * (log_max - log_min))

def edm_coeffs(t, sigma_data=1.0):
    t = t.view(-1, 1, 1)
    c_skip = sigma_data**2 / (t**2 + sigma_data**2)
    c_out = (t * sigma_data) / torch.sqrt(t**2 + sigma_data**2)
    c_in = 1.0 / torch.sqrt(t**2 + sigma_data**2)
    return c_in.squeeze(-1), c_out.squeeze(-1), c_skip.squeeze(-1)

def model_x0_pred_flow(model_q, xt_q, mask_q, batch_data, t_per_graph, sigma_data=1.0):
    edge_batch_indices = batch_data.batch[batch_data.edge_index[0]]
    t_per_edge = t_per_graph[edge_batch_indices]
    c_in, c_out, c_skip = edm_coeffs(t_per_edge.to(xt_q.device), sigma_data)
    
    xt_q_scaled = c_in * xt_q
    edge_attr_in = torch.cat([xt_q_scaled, mask_q, batch_data.edge_attr_static], dim=1)
    
    # 模型需要节点静态特征作为输入
    x_node_in = batch_data.x_static
    
    F_pred_q = model_q(x_node_in, edge_attr_in, batch_data.edge_index, batch_data.batch, t_per_graph)
    q0_pred = c_skip * xt_q + c_out * F_pred_q
    return q0_pred

# ==============================================================================
# 3. 训练与评估函数 [核心修改]
# ==============================================================================
def train_one_epoch_flow(model_q, dataloader, optimizer, t_min, t_max, batch_pairs, device):
    model_q.train()
    total_epoch_loss = 0
    pbar = tqdm(dataloader, desc=f"Training Flow Model")
    for batch_data in pbar:
        batch_data = batch_data.to(device)
        # 目标是边的标签 y_edge
        q0_true = batch_data.y_edge
        optimizer.zero_grad()
        total_batch_loss = 0

        for _ in range(batch_pairs):
            mask_q = torch.zeros_like(q0_true)
            # 这里的masking逻辑可能需要调整为针对边
            num_known_edges = int(batch_data.num_edges * random.uniform(0.1, 0.5))
            known_indices = select_random_indices(batch_data.num_edges, num_known_edges, [])
            if known_indices.numel() > 0: mask_q[known_indices] = 1.0
            
            num_graphs_in_batch = batch_data.num_graphs
            t1 = sample_log_uniform(num_graphs_in_batch, t_min, t_max, device)
            t2 = sample_log_uniform(num_graphs_in_batch, t_min, t_max, device)
            
            eps_q = torch.randn_like(q0_true)
            edge_batch = batch_data.batch[batch_data.edge_index[0]]
            
            xt1_q_noisy = q0_true + t1[edge_batch].view(-1, 1) * eps_q
            xt2_q_noisy = q0_true + t2[edge_batch].view(-1, 1) * eps_q
            
            xt1_q = torch.where(mask_q.bool(), q0_true, xt1_q_noisy)
            xt2_q = torch.where(mask_q.bool(), q0_true, xt2_q_noisy)
            
            q0_pred1 = model_x0_pred_flow(model_q, xt1_q, mask_q, batch_data, t1)
            q0_pred2 = model_x0_pred_flow(model_q, xt2_q, mask_q, batch_data, t2)
            
            loss_pair = F.mse_loss(q0_pred1.squeeze(), q0_pred2.squeeze())
            total_batch_loss += loss_pair
        
        avg_batch_loss = total_batch_loss / batch_pairs
        avg_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_q.parameters(), 1.0)
        optimizer.step()
        
        total_epoch_loss += avg_batch_loss.item()
        pbar.set_postfix(loss=avg_batch_loss.item())
        
    return total_epoch_loss / len(dataloader)

@torch.no_grad()
def evaluate_flow(model_q, dataloader, t_max, device, known_ratio=0.2):
    model_q.eval()
    all_true_unknown, all_preds_unknown = [], []
    for batch_data in tqdm(dataloader, desc="Evaluating Flow Model"):
        batch_data = batch_data.to(device)
        q0_true = batch_data.y_edge
        
        mask_q = torch.zeros_like(q0_true)
        num_known_edges = int(batch_data.num_edges * known_ratio)
        known_indices = select_random_indices(batch_data.num_edges, num_known_edges, [], seed=42)
        if known_indices.numel() > 0: mask_q[known_indices] = 1.0
        unknown_mask = (1.0 - mask_q).bool()
        if not unknown_mask.any(): continue
        
        initial_noise = torch.randn_like(q0_true) * t_max
        x_T_q = torch.where(mask_q.bool(), q0_true, initial_noise)
        t_gen_per_graph = torch.full((batch_data.num_graphs,), t_max, device=device)
        
        preds_norm = model_x0_pred_flow(model_q, x_T_q, mask_q, batch_data, t_gen_per_graph)
        
        all_true_unknown.append(q0_true[unknown_mask].cpu())
        all_preds_unknown.append(preds_norm[unknown_mask].cpu())
        
    if not all_true_unknown: return -1.0 # No unknown edges to evaluate
    all_true_tensor = torch.cat(all_true_unknown)
    all_preds_tensor = torch.cat(all_preds_unknown)
    return r2_score(all_true_tensor.numpy(), all_preds_tensor.numpy())

# ==============================================================================
# 5. 主程序 (Main Function)
# ==============================================================================
if __name__ == '__main__':
    # --- 1. 设置 ---

    seed = 42; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading data...")
    data_list = ["/data/zsc/Pipeline/data/epaNet/Anytown.inp",
                 "/data/zsc/Pipeline/data/epaNet/CTOWN.INP",
                 "/data/zsc/Pipeline/data/epaNet/L-TOWN.inp",
                 "/data/zsc/Pipeline/data/epaNet/tt.inp",
                 "/data/zsc/Pipeline/data/epaNet/Richmond_standard.inp",
                 "/data/zsc/Pipeline/data/epaNet/d-town.inp"]
    single_data = "/data/zsc/Pipeline/data/epaNet/Anytown.inp"
    use_multigraph = True
    if use_multigraph: 
        train_loader, val_loader, test_loader, pressure_norm, flow_norm,node_static_norm, flow_static_norm = load_multigraph_data("MultiGraphPretrainDataset" , data_list , 72 , 1 , mask_ratio=0.5)
        torch.save(pressure_norm, "model/pressure_norm.pt")
        torch.save(flow_norm, "model/flow_norm.pt")
        torch.save(node_static_norm, "model/node_static_norm.pt")
        torch.save(flow_static_norm, "model/flow_static_norm.pt")
    else:
        pressure_norm = torch.load("model/pressure_norm.pt", weights_only = False) if os.path.exists("model/pressure_norm.pt") else None
        flow_norm = torch.load("model/flow_norm.pt", weights_only = False) if os.path.exists("model/pressure_norm.pt") else None
        node_static_norm = torch.load("model/node_static_norm.pt", weights_only = False) if os.path.exists("model/pressure_norm.pt") else None
        edge_static_norm = torch.load("model/flow_static_norm.pt", weights_only = False) if os.path.exists("model/pressure_norm.pt") else None
        train_loader, val_loader, test_loader, _, _ = load_data("PretrainDataset" , single_data , 72 , 1 ,
                                                                pressure_normalizer=pressure_norm,
                                                                flow_normalizer=flow_norm,
                                                                node_static_norm=node_static_norm,
                                                                edge_static_norm=edge_static_norm,
                                                                mask_ratio=0.5)


    # --- 3. 初始化流量模型和优化器 ---
    # 确认你的输入维度
    # d_node_in = 你的 x_static 维度
    # d_edge_in = 1 (xt_q) + 1 (mask_q) + 你的 edge_attr_static 维度
    d_node_in_dim = 3
    d_edge_in_dim = 5
    
    model_q = WaterGUNet_Flow(
        d_node_in=d_node_in_dim,
        d_edge_in=d_edge_in_dim,
        d_model=64,
        d_time_emb=64,
        pool_ratios=[0.8, 0.8]
    ).to(device)
    
    optimizer_q = torch.optim.AdamW(model_q.parameters(), lr=1e-4, weight_decay=1e-6)
    
    # --- 4. 训练 ---
    epochs = 0; t_min = 0.002; t_max = 80.0; batch_pairs = 2
    save_path = "model/water_gunet_cm_flow_model_best.pt"; best_val_r2 = -float('inf')
    if os.path.exists(save_path):
        print(f"Loading pre-trained model from {save_path}")
        try: model_q.load_state_dict(torch.load(save_path))
        except Exception as e: print(f"Could not load model weights: {e}")
    
    print("Starting training for Flow Consistency Model...")
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        avg_train_loss = train_one_epoch_flow(model_q, train_loader, optimizer_q, t_min, t_max, batch_pairs, device)
        print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.6f}")
        
        val_r2 = evaluate_flow(model_q, val_loader, t_max, device)
        print(f"Epoch {epoch+1} Validation R² (Flow): {val_r2:.4f}")
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model_q.state_dict(), save_path)
            print(f"  -> New best flow model saved with Val R²: {best_val_r2:.4f}")

    print("\n--- Final Evaluation on Test Set ---")
    model_q.load_state_dict(torch.load(save_path))
    test_r2 = evaluate_flow(model_q, test_loader, t_max, device)
    print(f"\n==========================================")
    print(f"Final Test R² Score (Flow): {test_r2:.4f}")
    print(f"==========================================")