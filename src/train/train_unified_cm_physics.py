# ==========================================================
# [开始] 导入和辅助函数部分 (基本不变)
# ==========================================================
from physics_constraint import calculate_physics_loss_undirected, project_gradients, continuity_loss_undirected, hazen_williams_loss_undirected
from model import WaterGUNet, WaterGUNet_Flow, UnifiedWaterGUNet
import math
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch_geometric.utils import scatter
from train_cm_physics_inject import select_random_indices, load_custom_data, load_custom_model,sample_log_uniform, edm_coeffs, model_x0_pred_pressure, model_x0_pred_flow,evaluate_flow,evaluate_pressure
# ... (其他导入)
# ==========================================================
# [结束] 导入和辅助函数部分
# ==========================================================


# ==========================================================
# [开始] 物理损失函数的最终实现 (鲁棒归一化版本)
# 你需要将这部分代码放到你的 physics_constraint.py 文件中，并在这里导入
# 为方便起见，我暂时将它们直接放在这里
# ==========================================================
def continuity_loss_undirected_normalized(p_real, q_real, batch_data, flow_normalizer):
    edge_index = batch_data.edge_index
    num_nodes = batch_data.num_nodes
    
    total_head = p_real.flatten() + batch_data.elevations_real.flatten()
    row, col = edge_index
    head_diff = total_head[row] - total_head[col]
    flow_direction = torch.sign(head_diff)
    
    q_directed = torch.abs(q_real.flatten()) * flow_direction
    
    flows = torch.cat([q_directed, -q_directed], dim=0)
    indices = torch.cat([row, col], dim=0)
    net_flow_out = scatter(flows, indices, dim=0, dim_size=num_nodes, reduce='sum')
    demands_real = batch_data.demands_real.flatten()
    
    error_real = net_flow_out - demands_real
    
    scale = flow_normalizer.std.to(error_real.device) + 1e-8
    error_normalized = error_real / scale.flatten()

    loss = F.mse_loss(error_normalized, torch.zeros_like(error_normalized))
    return loss

def hazen_williams_loss_undirected_normalized(p_real, q_real, batch_data, pressure_normalizer, unit_conversion_factor=10.67):
    epsilon = 1e-8
    edge_index = batch_data.edge_index
    
    diameters_real = batch_data.diameters_real
    lengths_real = batch_data.lengths_real
    roughnesses_real = batch_data.roughnesses_real

    is_pipe_mask = diameters_real > 0
    if not torch.any(is_pipe_mask):
        return torch.tensor(0.0, device=p_real.device, requires_grad=True)

    K = (unit_conversion_factor * lengths_real[is_pipe_mask]) / \
        (torch.pow(roughnesses_real[is_pipe_mask], 1.852) * 
         torch.pow(diameters_real[is_pipe_mask], 4.87) + epsilon)

    q_abs_pipes = torch.abs(q_real.flatten()[is_pipe_mask])
    head_loss_from_flow_magnitude = K * torch.pow(q_abs_pipes, 1.852)

    total_head_pred = p_real.flatten() + batch_data.elevations_real.flatten()
    
    edge_index_pipes = edge_index[:, is_pipe_mask]
    row, col = edge_index_pipes
    
    head_src = total_head_pred[row]
    head_dst = total_head_pred[col]
    head_loss_from_pressure_magnitude = torch.abs(head_src - head_dst)
    
    error_real = head_loss_from_pressure_magnitude - head_loss_from_flow_magnitude
    
    scale = pressure_normalizer.std.to(error_real.device) + 1e-8
    error_normalized = error_real / scale.flatten()

    loss = F.mse_loss(error_normalized, torch.zeros_like(error_normalized))
    return loss
# ==========================================================
# [结束] 物理损失函数的最终实现
# ==========================================================


# ==========================================================
# [开始] 最终的、解耦的训练函数
# ==========================================================
def train_one_epoch_unified(
    unified_model,
    dataloader,
    optimizer,
    normalizers,
    scalers, # 传入 scaler 字典
    t_min, t_max, device,
    lambda_phys=0.1,
    phys_loss_lambdas={'cont': 1.0, 'eng': 0.1},
    batch_pairs=1,
    known_ratio_range=(0.1, 0.5)
    ):
    unified_model.train()
    total_loss_data_epoch, total_loss_phys_epoch = 0.0, 0.0

    pbar = tqdm(dataloader, desc=f"Unified PINN Training")
    for batch_data in pbar:
        batch_data = batch_data.to(device)
        p0_true, q0_true = batch_data.y_node, batch_data.y_edge
        
        for _ in range(batch_pairs):
            optimizer.zero_grad()

            # --- 1. 准备数据和噪声 (与之前相同) ---
            # ... (masking and noising logic) ...

            # --- 2. 一次前向传播，得到两个预测 ---
            p0_pred, q0_pred = unified_model_x0_pred(
                unified_model, xt_p, xt_q, mask_p, mask_q, batch_data, t
            )
            
            # --- 3. 计算统一的损失 ---
            # 直接使用MSE与真值比较，在微调和PINN中更稳定
            L_data = F.mse_loss(p0_pred, p0_true) + F.mse_loss(q0_pred, q0_true)

            L_physics = calculate_physics_loss_robust(
                p0_pred, q0_pred, batch_data, normalizers, scalers,
                lambda_cont=phys_loss_lambdas['cont'], 
                lambda_eng=phys_loss_lambdas['eng']
            )
            
            L_total = L_data + lambda_phys * L_physics
            
            # --- 4. 一次反向传播和更新 ---
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(unified_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss_data_epoch += L_data.item()
            total_loss_phys_epoch += L_physics.item()

        # ... (更新 pbar 的逻辑) ...
        
    avg_loss_data_epoch = total_loss_data_epoch / len(dataloader) / batch_pairs
    avg_loss_phys_epoch = total_loss_phys_epoch / len(dataloader) / batch_pairs
    return avg_loss_data_epoch, avg_loss_phys_epoch

# ==========================================================
# 主程序
# ==========================================================
if __name__ == "__main__":
    # --- 1. 设置 ---
    seed = 42; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 2. 加载数据和 Normalizers ---
    # train_loader, val_loader, test_loader, normalizers = load_multigraph_data(...)
    # (假设你已经加载了这些)

    # --- 3. 模型实例化 ---
    print("Instantiating UnifiedWaterGUNet model...")
    
    # 统一模型的输入维度
    unified_model_config = {
        "d_node_in": 5,        # 假设: scaled_xt_p(1) + mask_p(1) + x_static(3)
        "d_edge_in": 5,        # 假设: scaled_xt_q(1) + mask_q(1) + edge_attr_static(3)
        "d_model": 64,
        "d_time_emb": 64,
        "pool_ratios": [0.8, 0.8]
    }

    unified_model = UnifiedWaterGUNet(**unified_model_config).to(device)

    # 可选：加载预训练的骨干网络权重
    # 如果你想从之前训练好的模型迁移知识，可以这样做：
    # pretrained_p_dict = torch.load('path/to/pressure_model.pt')
    # model_dict = unified_model.state_dict()
    # # 1. Filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_p_dict.items() if k in model_dict and not k.startswith('output_proj')}
    # # 2. Overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict) 
    # # 3. Load the new state dict
    # unified_model.load_state_dict(model_dict)
    # print("Loaded pre-trained backbone weights.")


    # --- 4. 初始化优化器和损失尺度计算器 ---
    optimizer = torch.optim.AdamW(unified_model.parameters(), lr=1e-4) # 可以从一个稍大的学习率开始
    
    # 初始化鲁棒的损失归一化工具
    continuity_scaler = RunningLossScaler()
    energy_scaler = RunningLossScaler()
    scalers = {'cont': continuity_scaler, 'eng': energy_scaler}

    # --- 5. 训练循环 (带课程学习) ---
    finetune_epochs = 100
    warmup_epochs = 10
    lambda_phys_target = 0.1 # 物理约束最终的目标权重
    
    print("Starting unified PINN training...")
    best_val_metric = -float('inf')

    for epoch in range(finetune_epochs):
        # --- Lambda 调度 (课程学习) ---
        if epoch < warmup_epochs:
            # 在预热阶段，从0线性增加到目标权重
            current_lambda_phys = lambda_phys_target * (epoch / warmup_epochs)
        else:
            current_lambda_phys = lambda_phys_target
            
        print(f"\n--- Epoch {epoch+1}/{finetune_epochs} with lambda_phys = {current_lambda_phys:.4f} ---")

        avg_loss_d, avg_loss_p = train_one_epoch_unified(
            unified_model, train_loader,
            optimizer, normalizers, scalers,
            t_min=0.002, t_max=80.0, device=device,
            lambda_phys=current_lambda_phys,
            phys_loss_lambdas={'cont': 1.0, 'eng': 0.1},
            batch_pairs=1,
            known_ratio_range=(0.1, 0.5)
        )
        print(f"Avg Losses: Data={avg_loss_d:.6f}, Physics={avg_loss_p:.6f}")

        # --- 6. 评估和保存 ---
        if (epoch + 1) % 5 == 0:
            # 你需要一个新的评估函数来处理统一的模型
            # p_r2, f_r2 = evaluate_unified(unified_model, val_loader, ...)
            # print(f"  Validation R2 scores: Pressure={p_r2:.4f}, Flow={f_r2:.4f}")
            
            # current_metric = 0.7 * p_r2 + 0.3 * f_r2
            # if current_metric > best_val_metric:
            #     best_val_metric = current_metric
            #     torch.save(unified_model.state_dict(), 'model/unified_pinn_model_best.pt')
            #     print(f"  Saved new best model with validation metric: {best_val_metric:.4f}")
    
    print("Training complete.")