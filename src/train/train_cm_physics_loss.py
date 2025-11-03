from physics_constraint import calculate_physics_loss_undirected , project_gradients , continuity_loss_undirected,hazen_williams_loss_undirected
from model import WaterGUNet, WaterGUNet_Flow
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
from datetime import datetime, UTC


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

def load_custom_data(use_multigraph):
    print("Loading data...")
    data_list = [
                "/data/zsc/Pipeline/data/epaNet/Anytown.inp",
                 "/data/zsc/Pipeline/data/epaNet/CTOWN.INP",
                 "/data/zsc/Pipeline/data/epaNet/L-TOWN.inp",
                 "/data/zsc/Pipeline/data/epaNet/tt.inp",
                 "/data/zsc/Pipeline/data/epaNet/Richmond_standard.inp",
                 "/data/zsc/Pipeline/data/epaNet/d-town.inp"
    ]
    single_data = "/data/zsc/Pipeline/data/epaNet/EPANET/example-networks/Net1.inp"
    if use_multigraph: 
        data_path_train = "data/multigraph_data_train.pt"
        data_path_val = "data/multigraph_data_val.pt"
        data_path_test = "data/multigraph_data_test.pt"
        
        pressure_norm_path = "model/pressure_norm.pt"
        flow_norm_path = "model/flow_norm.pt"
        node_static_norm_path = "model/node_static_norm.pt"
        flow_static_norm_path = "model/flow_static_norm.pt"
        if os.path.exists(data_path_train) and os.path.exists(data_path_val) and os.path.exists(data_path_test):
            train_loader = torch.load(data_path_train,weights_only = False)
            val_loader = torch.load(data_path_val,weights_only = False)
            test_loader = torch.load(data_path_test,weights_only = False)
        else:
            train_loader, val_loader, test_loader, pressure_norm, flow_norm, node_static_norm, flow_static_norm = load_multigraph_data("MultiGraphPretrainDataset" , data_list , 72 , 1 , mask_ratio=0.5)
            torch.save(train_loader, data_path_train)
            torch.save(val_loader, data_path_val)
            torch.save(test_loader, data_path_test)
        # train_loader, val_loader, test_loader, pressure_norm, flow_norm,node_static_norm, flow_static_norm = load_multigraph_data("MultiGraphPretrainDataset" , data_list , 72 , 1 , mask_ratio=0.5)
        if os.path.exists(pressure_norm_path) and os.path.exists(flow_norm_path) and os.path.exists(node_static_norm_path) and os.path.exists(flow_static_norm_path):
            pressure_norm = torch.load(pressure_norm_path, weights_only = False)
            flow_norm = torch.load(flow_norm_path, weights_only = False)
            node_static_norm = torch.load(node_static_norm_path, weights_only = False)
            flow_static_norm = torch.load(flow_static_norm_path, weights_only = False)
        else:
            torch.save(pressure_norm, pressure_norm_path)
            torch.save(flow_norm, flow_norm_path)
            torch.save(node_static_norm, node_static_norm_path)
            torch.save(flow_static_norm, flow_static_norm_path)
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
        
    # torch.save(train_loader, "model/train_loader.pt")
    # torch.save(val_loader, "model/val_loader.pt")
    # torch.save(test_loader, "model/test_loader.pt")
        
    return train_loader, val_loader, test_loader, pressure_norm, flow_norm, node_static_norm, flow_static_norm


def load_custom_model(pressure_model_config , flow_model_config , cm_pressure_path , cm_flow_path , device):
    """
    创建压力和流量模型，加载预训练权重，并将它们移动到指定设备。
    """
    # 使用字典解包来实例化模型
    model_p = WaterGUNet(**pressure_model_config)
    model_q = WaterGUNet_Flow(**flow_model_config) # 确保这里的类名是流量模型的

    try:
        # 使用 map_location 来确保权重被加载到正确的设备上
        model_p.load_state_dict(torch.load(cm_pressure_path, map_location=device))
        print(f"Loaded pre-trained pressure model from {cm_pressure_path}")
    except Exception as e:
        print(f"Failed to load pre-trained pressure model from {cm_pressure_path}: {e}")
        
    try:
        model_q.load_state_dict(torch.load(cm_flow_path, map_location=device))
        print(f"Loaded pre-trained flow model from {cm_flow_path}")
    except Exception as e:
        print(f"Failed to load pre-trained flow model from {cm_flow_path}: {e}")

    # [关键修改]：在函数内部将模型移动到设备
    model_p.to(device)
    model_q.to(device)
    
    # 返回已经准备好的模型
    return model_p, model_q

#* consistency model
def sample_log_uniform(batch_size, t_min, t_max, device):
    u = torch.rand(batch_size, device=device)
    log_min, log_max = math.log(t_min), math.log(t_max)
    return torch.exp(log_min + u * (log_max - log_min))

def make_xt_from_x0(x0, t_per_graph, batch_indices, eps):
    t_per_node = t_per_graph[batch_indices]
    return x0 + t_per_node.view(-1, 1) * eps

def edm_coeffs(t, sigma_data=1.0):
    t = t.view(-1, 1, 1)
    c_skip = sigma_data**2 / (t**2 + sigma_data**2)
    c_out = (t * sigma_data) / torch.sqrt(t**2 + sigma_data**2)
    c_in = 1.0 / torch.sqrt(t**2 + sigma_data**2)
    return c_in.squeeze(-1), c_out.squeeze(-1), c_skip.squeeze(-1)

def model_x0_pred_pressure(model, xt_pressure, mask, static_node_features, edge_index, static_edge_features, batch_indices, t_per_graph):
    t_per_node = t_per_graph[batch_indices]
    c_in, c_out, c_skip = edm_coeffs(t_per_node.to(xt_pressure.device))
    xt_pressure_scaled = c_in * xt_pressure
    x_in = torch.cat([xt_pressure_scaled, mask, static_node_features], dim=1)
    F_pred = model(x_in, edge_index, static_edge_features, batch_indices, t_per_graph)
    x0_pred = c_skip * xt_pressure + c_out * F_pred
    return x0_pred

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

#*


def train_one_epoch_physics_injection(
    model_p, model_q,
    dataloader,
    optimizer_p, optimizer_q,
    normalizers,
    t_min, t_max, device,
    lambda_phys=0.1,
    phys_loss_lambdas={'cont': 1.0, 'eng': 0.1},
    batch_pairs=1,
    known_ratio_range=(0.1, 0.5)
    ):
    """
    使用物理梯度注入的范式，对压力和流量模型进行一个epoch的训练。

    Args:
        model_p (nn.Module): 压力预测模型。
        model_q (nn.Module): 流量预测模型。
        dataloader (DataLoader): 训练数据加载器。
        optimizer_p (Optimizer): 压力模型的优化器。
        optimizer_q (Optimizer): 流量模型的优化器。
        normalizers (dict): 包含 'pressure' 和 'flow' normalizer 的字典。
        t_min, t_max (float): 采样时间的范围。
        device (str): 'cuda' 或 'cpu'。
        lambda_phys (float): 物理梯度的全局注入强度。
        phys_loss_lambdas (dict): 物理损失内部各项的权重。
        batch_pairs (int): 每个batch内进行多少次加噪和预测。
        known_ratio_range (tuple): 已知节点/边的比例范围。
    
    Returns:
        tuple: (平均数据损失, 平均物理损失)。
    """
    model_p.train()
    model_q.train()
    
    # 用于累积整个epoch的损失，以便最后计算平均值
    total_loss_data_epoch = 0.0
    total_loss_phys_epoch = 0.0

    # 初始化 tqdm 进度条
    pbar = tqdm(dataloader, desc=f"Physics Injection Training")
    
    for batch_data in pbar:
        batch_data = batch_data.to(device)
        p0_true, q0_true = batch_data.y_node, batch_data.y_edge
        
        # 用于累积一个batch内的损失（如果batch_pairs > 1）
        total_loss_data_batch = 0.0
        total_loss_phys_batch = 0.0

        for _ in range(batch_pairs):
            optimizer_p.zero_grad()
            optimizer_q.zero_grad()

            # --- 1. 准备 Mask (Inpainting) ---
            mask_p = torch.zeros_like(p0_true)
            known_ratio_p = random.uniform(known_ratio_range[0], known_ratio_range[1])
            num_known_nodes = int(batch_data.num_nodes * known_ratio_p)
            excluded_indices_p = batch_data.reservoir_index.cpu().numpy()
            
            known_indices_p = select_random_indices(
                total_num=batch_data.num_nodes, 
                num_to_select=num_known_nodes, 
                excluded_indices=excluded_indices_p
            )
            if known_indices_p.numel() > 0:
                mask_p[known_indices_p] = 1.0

            mask_q = torch.zeros_like(q0_true)
            known_ratio_q = random.uniform(known_ratio_range[0], known_ratio_range[1])
            num_known_edges = int(batch_data.num_edges * known_ratio_q)
            known_indices_q = select_random_indices(
                total_num=batch_data.num_edges, 
                num_to_select=num_known_edges, 
                excluded_indices=[]
            )
            if known_indices_q.numel() > 0:
                mask_q[known_indices_q] = 1.0
            
            # --- 2. 准备带噪数据 ---
            num_graphs_in_batch = batch_data.num_graphs
            t1 = sample_log_uniform(num_graphs_in_batch, t_min, t_max, device)
            t2 = sample_log_uniform(num_graphs_in_batch, t_min, t_max, device)
            
            eps_p = torch.randn_like(p0_true)
            t1_p = t1[batch_data.batch].view(-1, 1); t2_p = t2[batch_data.batch].view(-1, 1)
            xt1_p_noisy = p0_true + t1_p * eps_p; xt2_p_noisy = p0_true + t2_p * eps_p
            xt1_p = torch.where(mask_p.bool(), p0_true, xt1_p_noisy)
            xt2_p = torch.where(mask_p.bool(), p0_true, xt2_p_noisy)
            
            eps_q = torch.randn_like(q0_true)
            edge_batch = batch_data.batch[batch_data.edge_index[0]]
            t1_q = t1[edge_batch].view(-1, 1); t2_q = t2[edge_batch].view(-1, 1)
            xt1_q_noisy = q0_true + t1_q * eps_q; xt2_q_noisy = q0_true + t2_q * eps_q
            xt1_q = torch.where(mask_q.bool(), q0_true, xt1_q_noisy)
            xt2_q = torch.where(mask_q.bool(), q0_true, xt2_q_noisy)
            
            

            # --- 3. 前向传播与损失计算 ---
            p0_pred1 = model_x0_pred_pressure(model_p, xt1_p, mask_p, batch_data.x_static, batch_data.edge_index, batch_data.edge_attr_static, batch_data.batch, t1)
            p0_pred2 = model_x0_pred_pressure(model_p, xt2_p, mask_p, batch_data.x_static, batch_data.edge_index, batch_data.edge_attr_static, batch_data.batch, t2)
            loss_consistency_p = F.mse_loss(p0_pred1, p0_pred2)

            q0_pred1 = model_x0_pred_flow(model_q, xt1_q, mask_q, batch_data, t1)
            q0_pred2 = model_x0_pred_flow(model_q, xt2_q, mask_q, batch_data, t2)
            loss_consistency_q = F.mse_loss(q0_pred1, q0_pred2)
            
            L_data = loss_consistency_p + loss_consistency_q

            p_real_pred1 = normalizers['pressure'].inverse_transform(p0_pred1)
            p_real_true = normalizers['pressure'].inverse_transform(p0_true)
            q_real_pred1 = normalizers['flow'].inverse_transform(q0_pred1)
            q_real_true = normalizers['flow'].inverse_transform(q0_true)
            
            
            p_static = normalizers['node_static'].inverse_transform(batch_data.x_static)
            q_static = normalizers['edge_static'].inverse_transform(batch_data.edge_attr_static)
            # print(f"q_static: {q_static.shape} , {q_static}")
            
            
            batch_data.lengths_real = q_static[:, 1]
            batch_data.diameters_real = q_static[:, 0]
            batch_data.roughnesses_real = q_static[:,2]
            batch_data.elevations_real = p_static[:, 0]
            
            L_physics = calculate_physics_loss_undirected(
                p_real_pred1, q_real_pred1, batch_data, 
                lambda_cont=phys_loss_lambdas['cont'], 
                lambda_eng=phys_loss_lambdas['eng']
            )
            
            loss_energy_p = hazen_williams_loss_undirected(p_real_pred1, q_real_true, batch_data)
            
            loss_continuity_q = continuity_loss_undirected(p_real_pred1, q_real_pred1, batch_data) 
            
            loss_energy_q = hazen_williams_loss_undirected(p_real_true, q_real_pred1, batch_data)
            
            # total_loss = L_data + lambda_phys * L_physics
            
            loss_phyic_p = phys_loss_lambdas['eng'] * loss_energy_p
            loss_phyic_q = phys_loss_lambdas['cont'] * loss_continuity_q + phys_loss_lambdas['eng'] * loss_energy_q
            L_physics = loss_phyic_p + loss_phyic_q

            total_loss_p = loss_consistency_p + lambda_phys * loss_phyic_p
            total_loss_q = loss_consistency_q + lambda_phys * loss_phyic_q

            # --- 4. 标准的反向传播与更新 ---
            # total_loss.backward()
            total_loss_p.backward(retain_graph=True)
            total_loss_q.backward()
            
            torch.nn.utils.clip_grad_norm_(model_p.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_q.parameters(), 1.0)
            
            # # --- 4. 梯度计算与注入 ---
            # params_p = list(model_p.parameters())
            # params_q = list(model_q.parameters())

            # all_params = params_p + params_q

            # # 4. 计算数据梯度
            # # 现在 inputs 是一个扁平的张量列表
            # # retain_graph=True 仍然是必需的
            # g_data_all = torch.autograd.grad(L_data, all_params, retain_graph=True)

            # # 5. 计算物理梯度
            # g_phys_all = torch.autograd.grad(L_physics, all_params)

            # # --- 现在需要将合并后的梯度重新拆分回 model_p 和 model_q ---
            # num_params_p = len(params_p)
            # g_data_p = g_data_all[:num_params_p]
            # g_data_q = g_data_all[num_params_p:]

            # g_phys_p = g_phys_all[:num_params_p]
            # g_phys_q = g_phys_all[num_params_p:]

            # g_phys_p_proj = project_gradients(g_phys_p, g_data_p)
            # g_phys_q_proj = project_gradients(g_phys_q, g_data_q)

            # with torch.no_grad():
            #     for param, grad_d, grad_p_proj in zip(params_p, g_data_p, g_phys_p_proj):
            #         param.grad = grad_d + lambda_phys * grad_p_proj
                
            #     for param, grad_d, grad_p_proj in zip(params_q, g_data_q, g_phys_q_proj):
            #         param.grad = grad_d + lambda_phys * grad_p_proj
            
            # torch.nn.utils.clip_grad_norm_(params_p, 1.0)
            # torch.nn.utils.clip_grad_norm_(params_q, 1.0)
            
            optimizer_p.step()
            optimizer_q.step()
            
            # 累加当前 pair 的损失
            total_loss_data_batch += L_data.item()
            total_loss_phys_batch += L_physics.item()

        # 在一个 batch 的所有 pair结束后，累加到 epoch 总损失中
        total_loss_data_epoch += total_loss_data_batch
        total_loss_phys_epoch += total_loss_phys_batch

        # --- 更新 tqdm 进度条 ---
        # 计算到当前 batch 为止的 epoch 平均损失
        # pbar.n 是 tqdm 内部的迭代计数器 (从0开始)
        # len(dataloader) 是总的 batch 数量
        current_batches = pbar.n + 1
        avg_loss_data_running = total_loss_data_epoch / current_batches / batch_pairs
        avg_loss_phys_running = total_loss_phys_epoch / current_batches / batch_pairs
        
        # 设置进度条的后缀，显示实时平均损失
        pbar.set_postfix(
            loss_data=f"{avg_loss_data_running:.5f}", 
            loss_phys=f"{avg_loss_phys_running:.5f}"
        )
        
    # 计算整个 epoch 的最终平均损失
    avg_loss_data_epoch = total_loss_data_epoch / len(dataloader) / batch_pairs
    avg_loss_phys_epoch = total_loss_phys_epoch / len(dataloader) / batch_pairs
    
    return avg_loss_data_epoch, avg_loss_phys_epoch

@torch.no_grad()
def evaluate_pressure(model, dataloader, t_max, device, known_ratio=0.2):
    model.eval(); all_true_unknown, all_preds_unknown = [], []
    for batch_data in tqdm(dataloader, desc="Evaluating"):
        batch_data = batch_data.to(device)
        b0_norm = batch_data.y_node
        mask = torch.zeros_like(b0_norm)
        num_known_nodes = int(batch_data.num_nodes * known_ratio)
        known_indices = select_random_indices(batch_data.num_nodes, num_known_nodes, batch_data.reservoir_index.cpu().numpy(), seed=42)
        if known_indices.numel() > 0: mask[known_indices] = 1.0
        unknown_mask = (1.0 - mask).bool()
        if not unknown_mask.any(): continue
        
        initial_noise = torch.randn_like(b0_norm) * t_max
        x_T_pressure = torch.where(mask.bool(), b0_norm, initial_noise)
        t_gen_per_graph = torch.full((batch_data.num_graphs,), t_max, device=device)
        preds_norm = model_x0_pred_pressure(model, x_T_pressure, mask, batch_data.x_static, batch_data.edge_index, batch_data.edge_attr_static, batch_data.batch, t_gen_per_graph)
        
        all_true_unknown.append(b0_norm[unknown_mask].cpu())
        all_preds_unknown.append(preds_norm[unknown_mask].cpu())
        
    all_true_tensor = torch.cat(all_true_unknown); all_preds_tensor = torch.cat(all_preds_unknown)
    return r2_score(all_true_tensor.numpy(), all_preds_tensor.numpy())

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


if __name__ == "__main__":
    train_loader, val_loader, test_loader, pressure_norm, flow_norm, node_static_norm, flow_static_norm = load_custom_data(use_multigraph=True)
    basic_pressure_model = 'model/water_gunet_cm_model_best.pt'
    basic_flow_model = 'model/water_gunet_cm_flow_model_best.pt'
    finetune_pressure_model = 'model/loss_water_gunet_finetuned_cm_pressure_model_best.pt'
    finetune_flow_model = 'model/loss_water_gunet_finetuned_cm_flow_model_best.pt'
    # basic_pressure_model = finetune_pressure_model
    # basic_flow_model = finetune_flow_model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_p , model_q = load_custom_model(
        pressure_model_config={
            "d_node_in":5,
            "d_edge_in":3,
            "d_model":64,
            "d_time_emb":64,
            "pool_ratios":[0.8, 0.8]
        },
        flow_model_config={
            "d_node_in":3,
            "d_edge_in":5,
            "d_model":64,
            "d_time_emb":64,
            "pool_ratios":[0.8, 0.8]
        },
        device=device,
        cm_pressure_path=basic_pressure_model,
        cm_flow_path=basic_flow_model
    )
    
    model_p.to(device)
    model_q.to(device)

    # 使用较小的学习率
    optimizer_p = torch.optim.AdamW(model_p.parameters(), lr=1e-4)
    optimizer_q = torch.optim.AdamW(model_q.parameters(), lr=1e-4)
    
    finetune_epochs = 100
    print("Starting physics-injection fine-tuning...")
    best_val_loss = float('inf')
    for epoch in range(finetune_epochs):
        avg_loss_d, avg_loss_p = train_one_epoch_physics_injection(
            model_p, model_q, train_loader,
            optimizer_p, optimizer_q,
            normalizers={'pressure':pressure_norm, 'flow':flow_norm , 'node_static':node_static_norm, 'edge_static':flow_static_norm},
            t_min=0.002, t_max=80.0, device=device,
            lambda_phys=1e-2, # 初始物理引导强度
            phys_loss_lambdas={'cont': 1, 'eng': 0.1},
            batch_pairs=1
        )
        print(f"Epoch {epoch+1}/{finetune_epochs} -> Avg Data Loss: {avg_loss_d:.6f}, Avg Physics Loss: {avg_loss_p:.6f}")
        if epoch % 5 == 0:
            p_r2 = evaluate_pressure(model_p, val_loader, t_max=80.0, device=device)
            f_r2 = evaluate_flow(model_q, val_loader, t_max=80.0, device=device)
            print(f"  Validation R2 scores at epoch {epoch+1}: Pressure={p_r2:.4f}, Flow={f_r2:.4f}")
        if avg_loss_d < best_val_loss:
            best_val_loss = avg_loss_d
            torch.save(model_p.state_dict(), finetune_pressure_model)
            torch.save(model_q.state_dict(), finetune_flow_model)
            print(f"  Saved best models at epoch {epoch+1} with validation loss {best_val_loss:.6f}")
            
    print("Fine-tuning complete.")
    p_r2 = evaluate_pressure(model_p, val_loader, t_max=80.0, device=device)
    f_r2 = evaluate_flow(model_q, val_loader, t_max=80.0, device=device)
    print(f"Validation R2 scores: Pressure={p_r2:.4f}, Flow={f_r2:.4f}")
        
