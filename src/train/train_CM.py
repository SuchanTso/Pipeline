import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Subset
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from utils import *
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import select_random_indices

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class UltimateSimpleModel(nn.Module):
    """
    一个用于最终诊断的、被简化到极致的模型。
    任务：给定一个带掩码的图，重建掩码部分。
    """
    def __init__(self, d_model, num_layers, F_dyn_n=1):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Linear(F_dyn_n, d_model)
        
        # GCN层
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(d_model, d_model))
            
        # 输出投影
        self.output_proj = nn.Linear(d_model, F_dyn_n)

    def forward(self, x, edge_index):
        """
        Args:
            x (Tensor): 输入的画布 [N, F_dyn_n]. 
                        观测节点是真实值，掩码节点是0或随机值。
            edge_index (Tensor): 边索引.
        """
        # 1. 投影到高维
        h = self.input_proj(x)
        
        # 2. 通过GCN层
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            
        # 3. 投影回原始维度
        output = self.output_proj(h)
        
        return output

def evaluate(model, dataloader, time_steps, mask_ratio, device, dynamic_normalizer):
    """
    对模型在给定数据集上的性能进行评估，包含 R2 指标。
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_masked_nodes = 0
    
    # --- 新增: 用于计算R2 ---
    all_true_values = []
    all_pred_values = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for data in progress_bar:
            data = data.to(device)
            
            num_nodes = data.num_nodes
            perm = torch.randperm(num_nodes, device=device)
            masked_node_count = int(num_nodes * mask_ratio)
            
            if masked_node_count == 0:
                continue

            node_masked_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            node_masked_mask[perm[:masked_node_count]] = True
            node_obs_mask = ~node_masked_mask

            t_max = time_steps[0]
            
            x_masked_true_norm = data.x_dynamic_window[node_masked_mask, -1, :]
            x_obs_true_norm = data.x_dynamic_window[node_obs_mask, -1, :]
            
            initial_noise = torch.randn_like(x_masked_true_norm) * t_max

            canvas_noise = torch.zeros(num_nodes, data.x_dynamic_window.size(-1), device=device)
            canvas_noise[node_obs_mask] = x_obs_true_norm
            canvas_noise[node_masked_mask] = initial_noise

            prediction = model(canvas_noise, data.edge_index)
            # prediction_norm = model(data, node_obs_mask, canvas_noise, t_max, 
                                    # use_teacher=True, train_encoder=False)
            
            pred_masked_norm = prediction[node_masked_mask]

            # --- 反归一化 ---
            pred_masked_orig = dynamic_normalizer.inverse_transform(pred_masked_norm)
            x_masked_true_orig = dynamic_normalizer.inverse_transform(x_masked_true_norm)
            
            # --- 累加误差 ---
            total_mse += F.mse_loss(pred_masked_orig, x_masked_true_orig, reduction='sum').item()
            total_mae += F.l1_loss(pred_masked_orig, x_masked_true_orig, reduction='sum').item()
            total_masked_nodes += masked_node_count

            # --- 新增: 收集数据用于计算R2 ---
            all_true_values.append(x_masked_true_orig.cpu())
            all_pred_values.append(pred_masked_orig.cpu())

    # --- 计算最终平均指标 ---
    avg_mse = total_mse / total_masked_nodes
    avg_mae = total_mae / total_masked_nodes
    avg_rmse = np.sqrt(avg_mse)
    
    # --- 新增: 计算R2 ---
    # 将所有批次的张量拼接起来
    all_true_tensor = torch.cat(all_true_values, dim=0)
    all_pred_tensor = torch.cat(all_pred_values, dim=0)

    # 计算 SS_tot (总平方和)
    mean_true_value = torch.mean(all_true_tensor)
    ss_tot = torch.sum((all_true_tensor - mean_true_value) ** 2)

    # 计算 SS_res (残差平方和)
    # 我们可以直接用 all_true_tensor 和 all_pred_tensor 计算，这比用 total_mse 更直接
    ss_res = torch.sum((all_true_tensor - all_pred_tensor) ** 2)

    # 计算 R2
    # 添加一个小的 epsilon 防止 ss_tot 为0（如果所有真实值都一样）
    r2_score = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'mse': avg_mse,
        'mae': avg_mae,
        'rmse': avg_rmse,
        'r2': r2_score.item() # .item() 将单元素张量转为Python数字
    }


def train_one_epoch(model, dataloader, optimizer, time_steps, mask_ratio, device):
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for data in progress_bar:
        data = data.to(device)
        
        # --- 1. 准备掩码 ---
        num_nodes = data.num_nodes
        perm = torch.randperm(num_nodes, device=device)
        masked_node_count = int(num_nodes * mask_ratio)
        node_masked_indices = perm[:masked_node_count]
        
        node_masked_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        node_masked_mask[node_masked_indices] = True
        node_obs_mask = ~node_masked_mask

        # --- 2. 采样时间并从真实值加噪 ---
        N_steps = len(time_steps)
        n = torch.randint(0, N_steps - 1, (1,), device=device)
        t_curr = time_steps[n]
        t_next = time_steps[n+1]
        
        x_masked_true = data.x_dynamic_window[node_masked_mask, -1, :]
        noise = torch.randn_like(x_masked_true)
        x_masked_t_curr = x_masked_true + t_curr * noise
        x_masked_t_next = x_masked_true + t_next * noise

        # --- 3. 构建解码器输入的画布 ---
        F_dyn = data.x_dynamic_window.size(-1)
        x_obs_true = data.x_dynamic_window[node_obs_mask, -1, :]

        canvas_curr = torch.zeros(num_nodes, F_dyn, device=device)
        canvas_curr[node_obs_mask] = x_obs_true
        canvas_curr[node_masked_mask] = x_masked_t_curr

        canvas_next = torch.zeros(num_nodes, F_dyn, device=device)
        canvas_next[node_obs_mask] = x_obs_true
        canvas_next[node_masked_mask] = x_masked_t_next

        # --- 4. 前向传播和损失计算 ---
        optimizer.zero_grad()
        prediction = model(canvas_curr, data.edge_index)
        # with torch.no_grad():
        #     target = model(data, node_obs_mask, canvas_next, t_next, use_teacher=True)

        # prediction = model(data, node_obs_mask, canvas_curr, t_curr, use_teacher=False, train_encoder=True)
        
        # loss = F.mse_loss(prediction[node_masked_mask], target[node_masked_mask])
        loss = F.mse_loss(prediction[node_masked_mask], x_masked_true) # 只用这行

        
        # --- 5. 反向传播和更新 ---
        loss.backward()
        
        # # --- 梯度检查 ---
        # print("\n--- Gradient Check ---")
        # # 检查Encoder
        # encoder_grad = model.encoder.static_encoder[0].lin.weight.grad
        # print(f"Encoder GCN grad norm: {torch.linalg.norm(encoder_grad).item() if encoder_grad is not None else 'None'}")

        # # 检查Decoder的Attention
        # q_proj_grad = model.decoder.layers[0].q_proj.weight.grad
        # print(f"Decoder Q_proj grad norm: {torch.linalg.norm(q_proj_grad).item() if q_proj_grad is not None else 'None'}")

        # # 检查AdaLN
        # adaln_grad = model.decoder.layers[0].adaln_1.cond_projector[-1].weight.grad
        # print(f"Decoder AdaLN grad norm: {torch.linalg.norm(adaln_grad).item() if adaln_grad is not None else 'None'}")
        # print("---------------------\n")
        
        
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # --- 6. EMA更新教师模型 ---
        # model.update_teacher()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)
    
if __name__ == '__main__':
    # --- 超参数设置 ---
    config = {
        'epanet_path': '/data/zsc/Pipeline/data/epaNet/Anytown.inp',
        'window_size': 6,
        'stride': 1,
        'batch_size': 1, # 对于图数据，batch通常是1，通过collate聚合
        'mask_ratio': 0.75,
        'epochs': 500,
        'lr': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        # CM 参数
        'cm_steps': 128,
        't_min': 0.002,
        't_max': 80.0,
        'rho': 7.0,
        'sigma_data': 1.0, # 需要从数据集中精确计算

        # 模型参数
        'F_stat_n': 4, # Example: elevations, demands, one-hot type(2) -> 4
        'F_dyn_n': 1, # Pressures
        'F_stat_e': 3, # diameters, lengths, roughnesses
        'F_cond': 128,
        'T_dim': 64,
        'd_model': 256,
        'num_encoder_gcn_layers': 2,
        'num_encoder_transformer_layers': 2,
        'd_model_transformer_encoder': 128, # Transformer内部的工作维度
        'nhead_transformer_encoder': 4,     # 现在 128 % 4 == 0, 没有问题了
        'num_decoder_layers': 4,
        'num_heads': 4,
        'k_eigvecs': 16,
    }
    
    # --- 1. 数据准备 ---
    print("--- Preparing Data ---")
    # 假设EpytHelper和WaterNetworkWindowedDataset可用
    # epa_helper = EpytHelper(config['epanet_path'], hrs=...)
    # raw_data = epa_helper.get_raw_data()
    dataset = WaterNetworkWindowedDataset(
        epanet_path=config['epanet_path'],
        window_size=config['window_size'],
        stride=config['stride'],
        k_eigvecs=config['k_eigvecs']
    )
    # # 精确计算 sigma_data
    # normalizers = torch.load(os.path.join(dataset.processed_dir, 'normalizers.pt'),weights_only=False)
    # config['sigma_data'] = normalizers['dynamic_node'].std.item()
    # print(f"Using sigma_data: {config['sigma_data']:.4f}")

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.8 * dataset_size)) # 80% 训练, 20% 验证

    train_indices, val_indices = indices[:split], indices[split:]

    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False) # 验证集不需要shuffle
    dynamic_normalizer = dataset.dynamic_node_normalizer

    # print(f"Dataset loaded with {len(dataset)} samples.")

    # --- 2. 模型、优化器、调度器初始化 ---
    print("--- Initializing Model ---")
    encoder_params = {
        'F_stat_n': config['F_stat_n'], 
        'F_dyn_n': config['F_dyn_n'], 
        'F_cond': config['F_cond'], 
        'T_dim': config['T_dim'], 
        'd_model_transformer': config['d_model_transformer_encoder'],
        'nhead_transformer': config['nhead_transformer_encoder'],
        'num_layers_gcn': config['num_encoder_gcn_layers'], 
        'num_layers_transformer': config['num_encoder_transformer_layers']
    }
    decoder_params = {
        'F_dyn_n': config['F_dyn_n'], 
        'F_cond': config['F_cond'], 
        'd_model': config['d_model'], 
        'num_layers': config['num_decoder_layers'], 
        'num_heads': config['num_heads'],
        'k_eigvecs': config['k_eigvecs'],
        'sigma_data': config['sigma_data']
    }
    simple_decoder_params = {
        'F_dyn_n': config['F_dyn_n'], 
        'F_cond': config['F_cond'], 
        'd_model': config['d_model'], 
        'num_layers': config['num_decoder_layers'], 
        'sigma_data': config['sigma_data']
    }
    
    simple_model_config = {
        'd_model': 128,
        'num_layers': 4,
        'F_dyn_n': config['F_dyn_n']
    }
    USE_SIMPLE_DECODER = True
    if USE_SIMPLE_DECODER:
        model = UltimateSimpleModel(**simple_model_config).to(config['device'])
        # decoder_params = simple_decoder_params
        # model = GC_MAE_Model_Simple(encoder_params, decoder_params).to(config['device'])
    else:
        model = GC_MAE_Model(encoder_params, decoder_params).to(config['device'])
    
    # 可选：加载预训练的MAE权重
    # load_pretrained_weights(...)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # --- 3. 计算时间步 ---
    t_min, t_max, rho, N_steps = config['t_min'], config['t_max'], config['rho'], config['cm_steps']
    time_steps = (t_max**(1/rho) + torch.arange(N_steps) / (N_steps - 1) * (t_min**(1/rho) - t_max**(1/rho)))**rho
    time_steps = time_steps.to(config['device'])

    # --- 4. 训练循环 ---
    print("--- Starting Training ---")
    val_metrics = {
        "rmse": float('inf'),
        "mae": float('inf'),
        "r2": float('-inf')
    }
    best_val_rmse = float('inf')
    
    fix_data = [next(iter(train_dataloader))] # 用于评估的固定数据
    losses = []
    r2_scores = []
    visualization_steps = [0, 50, 100, 200, 499] 

    
    for epoch in range(1, config['epochs'] + 1):
        avg_loss = train_one_epoch(
            model=model,
            dataloader=fix_data, # 确保dataloader已正确创建
            optimizer=optimizer,
            time_steps=time_steps,
            mask_ratio=config['mask_ratio'],
            device=config['device']
        )
        # --- 每隔N个epoch或每个epoch都进行评估 ---
        # if epoch % 5 == 0 or epoch == config['epochs']:
        #     val_metrics = evaluate(
        #         model=model,
        #         dataloader=fix_data,
        #         time_steps=time_steps,
        #         mask_ratio=config['mask_ratio'], # 可以使用与训练相同的mask_ratio
        #         device=config['device'],
        #         dynamic_normalizer=dataset.dynamic_node_normalizer
        #     )
        with torch.no_grad():
            model.eval()
            val_metrics = evaluate(
                model=model,
                dataloader=fix_data,
                time_steps=time_steps,
                mask_ratio=config['mask_ratio'], # 可以使用与训练相同的mask_ratio
                device=config['device'],
                dynamic_normalizer=dataset.dynamic_node_normalizer
            )
            r2_scores.append(val_metrics['r2'])
            model.train()
        
        # if epoch in visualization_steps:
        #     visualize_overfitting_progress(
        #     model, fix_data[0], time_steps, config['mask_ratio'],
        #     config['device'], dynamic_normalizer, epoch, losses, r2_scores
        # )
        
        print(f"Epoch {epoch:03d} | Validation RMSE: {val_metrics['rmse']:.4f} | Validation MAE: {val_metrics['mae']:.4f} | Validation R2: {val_metrics['r2']:.4f} | Train Loss: {avg_loss:.4f}")

        # --- 保存最佳模型 ---
        # if val_metrics['rmse'] < best_val_rmse:
        #     best_val_rmse = val_metrics['rmse']
        #     print(f"  -> New best model found! Saving to best_model.pt")
        #     torch.save(model.state_dict(), "model/CM_best_model.pt")
        
        scheduler.step()
        
        

    print("--- Training Finished ---")