# file: gma_evaluation.py (或者 pretrain_utils.py 的一部分)
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import * # 假设你的模型在model.py中
# from loss import physics_loss # 如果需要的话
from dataset import WaterEPANetDataset # 假设你的Dataset在这里
import os
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt # 导入matplotlib
from dataset import *
from utils import prepare_training_env

def get_parser():
    # ... (保持不变)
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, nargs="?", help="path to model")
    parser.add_argument("-d", "--data", type=str, nargs="?", help="path to epanet data")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs to train")
    parser.add_argument("-l", "--log_every_epoch", type=int, default=10, help="log every n epochs")
    parser.add_argument("--hours_analysis", type=int, default=72, help="hours to analysis")
    parser.add_argument("-p","--save_path" , type=str, default='figures/dual_net3_100/', help="path to save figures")
    return parser
# -----------------------------------------------------------------------------
# 1. 可视化函数 (与之前相同，但我们会在调用时传入重建值)
# -----------------------------------------------------------------------------
def plot_reconstruction(reconstructed_nodes, real_nodes, reconstructed_edges, real_edges,
                        node_indices, edge_indices, epoch, save_dir='figures/gma_recon'):
    """
    可视化GMA模型对被掩码部分的重建结果与真实值的对比。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.style.use('seaborn-v0_8-whitegrid')
    width = 0.35

    # --- 节点压力重建对比 ---
    if len(node_indices) > 0:
        fig1, ax1 = plt.subplots(figsize=(max(15, len(node_indices) * 0.5), 7))
        x_nodes = np.arange(len(node_indices))
        
        rects1 = ax1.bar(x_nodes - width/2, real_nodes, width, label='Real Value', color='royalblue')
        rects2 = ax1.bar(x_nodes + width/2, reconstructed_nodes, width, label='Reconstructed Value', color='lightcoral')

        ax1.set_ylabel('Pressure (Physical Scale)')
        ax1.set_title(f'Node Pressure Reconstruction (Epoch {epoch})')
        ax1.set_xticks(x_nodes)
        ax1.set_xticklabels([f'Node {i}' for i in node_indices], rotation=45, ha="right")
        ax1.legend()
        ax1.bar_label(rects1, padding=3, fmt='%.2f', fontsize=8)
        ax1.bar_label(rects2, padding=3, fmt='%.2f', fontsize=8)
        fig1.tight_layout()
        plt.savefig(os.path.join(save_dir, f'gma_node_recon_epoch_{epoch}.png'))
        plt.close(fig1)

    # --- 管道流量重建对比 ---
    if len(edge_indices) > 0:
        fig2, ax2 = plt.subplots(figsize=(max(15, len(edge_indices) * 0.5), 7))
        x_edges = np.arange(len(edge_indices))
        
        rects3 = ax2.bar(x_edges - width/2, real_edges, width, label='Real Value', color='darkgreen')
        rects4 = ax2.bar(x_edges + width/2, reconstructed_edges, width, label='Reconstructed Value', color='orange')

        ax2.set_ylabel('Flow (Physical Scale)')
        ax2.set_title(f'Pipe Flow Reconstruction (Epoch {epoch})')
        ax2.set_xticks(x_edges)
        ax2.set_xticklabels([f'Pipe {i}' for i in edge_indices], rotation=45, ha="right")
        ax2.legend()
        ax2.bar_label(rects3, padding=3, fmt='%.2f', fontsize=8)
        ax2.bar_label(rects4, padding=3, fmt='%.2f', fontsize=8)
        fig2.tight_layout()
        plt.savefig(os.path.join(save_dir, f'gma_pipe_recon_epoch_{epoch}.png'))
        plt.close(fig2)

# -----------------------------------------------------------------------------
# 2. 评估指标计算函数 (与之前相同，但我们传入重建值)
# -----------------------------------------------------------------------------
def reconstruction_metrics(y_true, y_pred, inverse_transform_func=None):
    """
    计算并返回一组回归任务的评估指标。

    Args:
        y_true (np.ndarray or torch.Tensor): 真实值.
        y_pred (np.ndarray or torch.Tensor): 预测值.
        inverse_transform_func (function, optional): 
            如果你的目标值经过了变换（如log(1+x)），传入其逆变换函数 
            (例如 lambda x: np.exp(x) - 1)。默认为None。

    Returns:
        dict: 包含MAE, RMSE, R2, MAPE指标的字典。
    """
    # 确保输入是numpy array
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # 如果有逆变换函数，先应用它
    if inverse_transform_func:
        y_true = inverse_transform_func(y_true)
        y_pred = inverse_transform_func(y_pred)

    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(y_true - y_pred))

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))

    # R-squared (R²)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    # 避免ss_tot为0的情况
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # MAPE (Mean Absolute Percentage Error)
    # 过滤掉y_true为0的情况，防止除以0错误
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = np.nan # 如果所有真实值都是0，则无法计算MAPE

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2 Score': r2,
        'MAPE (%)': mape
    }
    
    return metrics

# -----------------------------------------------------------------------------
# 3. 主评估函数 (为GMA预训练任务重写)
# -----------------------------------------------------------------------------
def evaluate_gma(model, data_loader, device, pressure_norm, flow_norm,
                 epoch=0, node_mask_ratio=0.75, edge_mask_ratio=0.75,
                 plot_every_n_batches=1, net_name = '',save_dir='figures/gma_recon'):
    """
    对 GraphMaskedAutoencoder 模型进行全面的重建评估。
    """
    model.eval()
    
    all_recon_nodes, all_real_nodes = [], []
    all_recon_edges, all_real_edges = [], []
    
    loss_fn = nn.MSELoss()
    total_loss_norm = 0.0
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc=f"Evaluating GMA Epoch {epoch}")):
            batch = batch.to(device)
            
            # --- 1. 生成固定的评估掩码 ---
            num_nodes = batch.num_nodes
            num_edges = batch.num_edges
            reservoir_indices = (batch.node_type == 1).nonzero(as_tuple=True)[0].tolist()

            num_masked_nodes = int(num_nodes * node_mask_ratio)
            num_masked_edges = int(num_edges * edge_mask_ratio)
            
            masked_node_indices_list = select_random_indices(num_nodes, num_masked_nodes, reservoir_indices, seed=epoch) # 用epoch做种子，每次评估掩码不同但可复现
            masked_edge_indices_list = select_random_indices(num_edges, num_masked_edges, [], seed=epoch)
            
            masked_node_indices = torch.tensor(masked_node_indices_list, dtype=torch.long, device=device)
            masked_edge_indices = torch.tensor(masked_edge_indices_list, dtype=torch.long, device=device)

            # --- 2. 模型前向传播 ---
            recon_pressures_norm, recon_flows_norm = model(batch, masked_node_indices, masked_edge_indices)
            
            # --- 3. 提取真实值 ---
            real_pressures_norm = batch.y_node
            real_flows_norm = batch.y_edge
            
            # --- 4. 累加被掩码部分的结果，用于计算总指标 ---
            if masked_node_indices.numel() > 0:
                all_recon_nodes.append(recon_pressures_norm[masked_node_indices].cpu())
                all_real_nodes.append(real_pressures_norm[masked_node_indices].cpu())

            if masked_edge_indices.numel() > 0:
                all_recon_edges.append(recon_flows_norm[masked_edge_indices].cpu())
                all_real_edges.append(real_flows_norm[masked_edge_indices].cpu())

            # --- 5. 计算当前批次的损失 (用于ReduceLROnPlateau) ---
            loss_node = 0.0
            if masked_node_indices.numel() > 0:
                loss_node = loss_fn(recon_pressures_norm[masked_node_indices], real_pressures_norm[masked_node_indices])
            
            loss_edge = 0.0
            if masked_edge_indices.numel() > 0:
                loss_edge = loss_fn(recon_flows_norm[masked_edge_indices], real_flows_norm[masked_edge_indices])

            loss = loss_node + loss_edge
            total_loss_norm += loss.item() if isinstance(loss, torch.Tensor) else loss

            # --- 6. 可视化当前批次的结果 ---
            fig_path = ""
            if save_dir is not None:
                fig_path = os.path.join('figures', save_dir, net_name)
            if plot_every_n_batches > 0 and i % plot_every_n_batches == 0:
                if masked_node_indices.numel() > 0 and masked_edge_indices.numel() > 0:
                    # 反归一化用于绘图
                    recon_nodes_phys = pressure_norm.inverse_transform(recon_pressures_norm.cpu())
                    real_nodes_phys = pressure_norm.inverse_transform(real_pressures_norm.cpu())
                    recon_edges_phys = flow_norm.inverse_transform(recon_flows_norm.cpu())
                    real_edges_phys = flow_norm.inverse_transform(real_flows_norm.cpu())
                    
                    plot_reconstruction(
                        recon_nodes_phys[masked_node_indices.cpu()].flatten().numpy(),
                        real_nodes_phys[masked_node_indices.cpu()].flatten().numpy(),
                        recon_edges_phys[masked_edge_indices.cpu()].flatten().numpy(),
                        real_edges_phys[masked_edge_indices.cpu()].flatten().numpy(),
                        masked_node_indices.cpu().numpy(),
                        masked_edge_indices.cpu().numpy(),
                        epoch=f"{epoch}_batch{i}",
                        save_dir=fig_path
                    )

    # --- 聚合所有批次的结果 ---
    avg_loss_norm = total_loss_norm / len(data_loader)
    
    print(f"\n--- GMA Evaluation Summary for Epoch {epoch} ---")
    print(f"Average Reconstruction MSE (Normalized space, on masked items): {avg_loss_norm:.6f}")
    
    if all_recon_nodes and all_real_nodes:
        all_recon_nodes = torch.cat(all_recon_nodes, dim=0)
        all_real_nodes = torch.cat(all_real_nodes, dim=0)
        print("\n--- Node Reconstruction Metrics on Full Validation Set (Original Scale) ---")
        pressure_metrics = reconstruction_metrics(all_real_nodes, all_recon_nodes)
        for name, value in pressure_metrics.items():
            print(f"  {name}: {value:.4f}")

    if all_recon_edges and all_real_edges:
        all_recon_edges = torch.cat(all_recon_edges, dim=0)
        all_real_edges = torch.cat(all_real_edges, dim=0)
        print("\n--- Pipe Reconstruction Metrics on Full Validation Set (Original Scale) ---")
        flow_metrics = reconstruction_metrics(all_real_edges, all_recon_edges)
        for name, value in flow_metrics.items():
            print(f"  {name}: {value:.4f}")
    
    return avg_loss_norm


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    model, optimizer, train_loader, val_loader, test_loader, pressure_norm, flow_norm, device ,_ = prepare_training_env(
        args.model_path, args.data, args.hours_analysis,mask_ratio=0.5
    )
    # evaluate_gma(model, 
    #          optimizer, 
    #          train_loader , 
    #          val_loader, 
    #          args.epochs , 
    #          device ,
    #          args.model_path , 
    #          scheduler=None,
    #          log_every_epoch=args.log_every_epoch )
    
    val_loss = evaluate_gma(model, 
                            test_loader, 
                            device=device,
                            pressure_norm=pressure_norm, 
                            flow_norm=flow_norm, 
                            epoch= 1, 
                            node_mask_ratio=0.75,
                            edge_mask_ratio=0.75,
                            plot_every_n_batches=1,
                            save_dir=args.save_path)
