import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import GNN_ChebConv, TGCN_MessageCoupling # 假设你的模型在model.py中
# from loss import physics_loss # 如果需要的话
from dataset import WaterEPANetDataset # 假设你的Dataset在这里
import os
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt # 导入matplotlib
from dataset import ZScoreNormalizer , LogZScoreNormalizer # 假设你的归一化类在这里
from utils import prepare_training_env

# ----------------------------------------------------------
# 新增的绘图函数
# ----------------------------------------------------------
def plot_predictions(pred_nodes, real_nodes, pred_edges, real_edges,
                     node_indices, edge_indices, epoch, save_dir='figures'):
    """
    可视化单个样本的预测结果与真实值的对比。
    
    Args:
        pred_nodes (np.array): 被mask节点的预测压力值 [num_masked_nodes]
        real_nodes (np.array): 被mask节点的真实压力值 [num_masked_nodes]
        pred_edges (np.array): 被mask管道的预测流量值 [num_masked_edges]
        real_edges (np.array): 被mask管道的真实流量值 [num_masked_edges]
        node_indices (np.array): 被mask节点的索引 [num_masked_nodes]
        edge_indices (np.array): 被mask管道的索引 [num_masked_edges]
        epoch (int): 当前的epoch数，用于文件名
        save_dir (str): 保存图片的目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- 1. 绘制节点压力对比图 ---
    plt.style.use('seaborn-v0_8-whitegrid') # 使用好看的样式
    fig, ax1 = plt.subplots(figsize=(15, 7))
    
    num_masked_nodes = len(node_indices)
    x_nodes = np.arange(num_masked_nodes)
    width = 0.35

    rects1 = ax1.bar(x_nodes - width/2, real_nodes, width, label='Real Pressure', color='royalblue')
    rects2 = ax1.bar(x_nodes + width/2, pred_nodes, width, label='Predicted Pressure', color='lightcoral')

    ax1.set_ylabel('Pressure')
    ax1.set_title(f'Node Pressure Prediction vs. Real (Epoch {epoch})')
    ax1.set_xticks(x_nodes)
    ax1.set_xticklabels([f'Node {i}' for i in node_indices], rotation=45, ha="right")
    ax1.legend()
    ax1.bar_label(rects1, padding=3, fmt='%.2f')
    ax1.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f'node_pressure_comparison_epoch_{epoch}.png'))
    plt.close(fig) # 关闭图形，防止在Jupyter中重复显示

    # --- 2. 绘制管道流量对比图 ---
    fig, ax2 = plt.subplots(figsize=(15, 7))

    num_masked_edges = len(edge_indices)
    x_edges = np.arange(num_masked_edges)
    # plt.ylim(-1000 , 3000)
    
    rects3 = ax2.bar(x_edges - width/2, real_edges, width, label='Real Flow', color='darkgreen')
    rects4 = ax2.bar(x_edges + width/2, pred_edges, width, label='Predicted Flow', color='orange')

    ax2.set_ylabel('Flow')
    ax2.set_title(f'Pipe Flow Prediction vs. Real (Epoch {epoch})')
    ax2.set_xticks(x_edges)
    ax2.set_xticklabels([f'Pipe {i}' for i in edge_indices], rotation=45, ha="right")
    ax2.legend()
    ax2.bar_label(rects3, padding=3, fmt='%.2f')
    ax2.bar_label(rects4, padding=3, fmt='%.2f')
    
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f'pipe_flow_comparison_epoch_{epoch}.png'))
    plt.close(fig)

    print(f"Prediction comparison plots saved to '{save_dir}/'")


# 你的 get_parser, gen_batch_filtered_loss, prepare_training_env 函数保持不变
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

def gen_batch_filtered_loss(node_pred, edge_pred, node_real, edge_real, 
                            node_masked_index, edge_masked_index,
                            node_normalizer, edge_normalizer):
    # ... (保持不变)
    inverse_pred_nodes = node_normalizer.inverse_transform(node_pred.cpu())
    inverse_real_nodes = node_normalizer.inverse_transform(node_real.cpu())
    inverse_pred_edge = edge_normalizer.inverse_transform(edge_pred.cpu())
    inverse_real_edge = edge_normalizer.inverse_transform(edge_real.cpu())
    
    # 提取被mask部分的值用于绘图
    node_pred_masked = inverse_pred_nodes[node_masked_index].flatten().numpy()
    node_real_masked = inverse_real_nodes[node_masked_index].flatten().numpy()
    edge_pred_masked = inverse_pred_edge[edge_masked_index].flatten().numpy()
    edge_real_masked = inverse_real_edge[edge_masked_index].flatten().numpy()
    
    node_loss = torch.nn.MSELoss(reduction='none')(inverse_pred_nodes, inverse_real_nodes)
    edge_loss = torch.nn.MSELoss(reduction='none')(inverse_pred_edge, inverse_real_edge)
    
    if node_masked_index is not None and len(node_masked_index) > 0:
        node_loss = node_loss[node_masked_index]
    else:
        node_loss = torch.tensor(0.0) # 如果没有mask，损失为0

    if edge_masked_index is not None and len(edge_masked_index) > 0:
        edge_loss = edge_loss[edge_masked_index]
    else:
        edge_loss = torch.tensor(0.0)
    
    return (node_loss.mean(), edge_loss.mean(), 
            node_pred_masked, node_real_masked,
            edge_pred_masked, edge_real_masked)


                
def eval(model, test_loader, pressure_norm, flow_norm, device, epoch=0, plot_sample=True , net_name='' , fig_path = None):
    model.eval()
    total_loss = 0
    total_node_loss = 0
    total_edge_loss = 0
    
    # 初始化用于累加filtered loss的变量
    sum_filtered_node_loss = 0
    sum_filtered_edge_loss = 0
    num_batches_with_masks = 0

    # 用于绘图的数据
    first_batch_plotted = 0
    
    pred_pressures = []
    real_pressures = []
    pred_flows = []
    real_flows = []

    with torch.no_grad():
        # 使用tqdm来显示进度条
        for batch in tqdm(test_loader, desc=f"Evaluating Epoch {epoch}"):
            # 你的batch解包和设备转移代码
            # ...
            x_seq, edge_index, edge_attr, y_node, y_edge, masked_node_index, masked_pipe_index = batch
            x_seq = x_seq[0].to(device)
            edge_index = edge_index[0].to(device)
            edge_attr = edge_attr[0].to(device)
            y_node = y_node[0].to(device)
            y_edge = y_edge[0].to(device)
            masked_node_index = masked_node_index[0]
            masked_pipe_index = masked_pipe_index[0]

            pred_nodes, pred_edges = model(x_seq, edge_index, edge_attr)
            
            # 计算整体损失（在归一化空间）
            node_loss = torch.nn.MSELoss()(pred_nodes, y_node)
            edge_loss = torch.nn.MSELoss()(pred_edges, y_edge)
            total_node_loss += node_loss.item()
            total_edge_loss += edge_loss.item()
            total_loss += (node_loss + edge_loss).item()

            # 计算并累加只针对masked部分的损失（在原始物理空间）
            (filtered_node_loss_mean, filtered_edge_loss_mean,
             node_pred_masked, node_real_masked,
             edge_pred_masked, edge_real_masked) = gen_batch_filtered_loss(
                pred_nodes, pred_edges, y_node, y_edge,
                masked_node_index, masked_pipe_index,
                pressure_norm, flow_norm
            )
            
            if filtered_node_loss_mean > 0 or filtered_edge_loss_mean > 0:
                sum_filtered_node_loss += filtered_node_loss_mean.item()
                sum_filtered_edge_loss += filtered_edge_loss_mean.item()
                num_batches_with_masks += 1

            pred_pressures.append(pred_nodes.cpu())
            real_pressures.append(y_node.cpu())
            pred_flows.append(pred_edges.cpu())
            real_flows.append(y_edge.cpu())

            # --- 绘图逻辑 ---
            save_dir = ""
            if fig_path is not None:
                save_dir = os.path.join('figures', fig_path, net_name)
                print(f"Saving figures to {save_dir}")
                if os.path.exists(save_dir) == False:
                    os.makedirs(save_dir , mode= 0o755 , exist_ok=True)
            if plot_sample and len(masked_node_index) < 100:
                if len(masked_node_index) > 0 and len(masked_pipe_index) > 0:
                    plot_predictions(
                        node_pred_masked, node_real_masked,
                        edge_pred_masked, edge_real_masked,
                        masked_node_index.cpu().numpy(),
                        masked_pipe_index.cpu().numpy(),
                        epoch,
                        save_dir= save_dir + f"/{first_batch_plotted}"
                    )
                    first_batch_plotted = first_batch_plotted + 1

    # 计算并打印平均损失
    avg_loss = total_loss / len(test_loader)
    avg_node_loss = total_node_loss / len(test_loader)
    avg_edge_loss = total_edge_loss / len(test_loader)
    
    avg_filtered_node_loss = sum_filtered_node_loss / num_batches_with_masks if num_batches_with_masks > 0 else 0
    avg_filtered_edge_loss = sum_filtered_edge_loss / num_batches_with_masks if num_batches_with_masks > 0 else 0
    
    pred_pressures_tensor = torch.cat(pred_pressures, dim=0)
    real_pressures_tensor = torch.cat(real_pressures, dim=0)
    pred_flows_tensor = torch.cat(pred_flows, dim=0)
    real_flows_tensor = torch.cat(real_flows, dim=0)
    
    pressure_metrics = regression_metrics(
        real_pressures_tensor, pred_pressures_tensor
    )
    flow_metrics = regression_metrics(
        real_flows_tensor, pred_flows_tensor
    )
    
    print(f"\n--- Evaluation Summary for Epoch {epoch} ---")
    print(f"Overall Loss (Normalized space): {avg_loss:.6f}")
    print(f"  - Node Pressure Loss: {avg_node_loss:.6f}")
    print(f"  - Pipe Flow Loss: {avg_edge_loss:.6f}")
    print("-" * 20)
    print(f"Filtered MSE Loss (Original scale, on masked items only):")
    print(f"  - Node Pressure MSE: {avg_filtered_node_loss:.6f}")
    print(f"  - Pipe Flow MSE: {avg_filtered_edge_loss:.6f}")
    print("-" * 20)
    print("Pressure Metrics:")
    for name, value in pressure_metrics.items():
        print(f"  {name}: {value:.4f}")
    print("Flow Metrics:")
    for name, value in flow_metrics.items():
        print(f"  {name}: {value:.4f}")

    return avg_loss

def regression_metrics(y_true, y_pred, inverse_transform_func=None):
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
    

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # 模拟一个训练循环，在每个epoch后进行评估和绘图
    # 这里为了演示，我们只在第0个epoch进行一次评估
    # 在你的实际训练脚本中，你会把eval放在训练循环的末尾
    
    model, optimizer, train_loader, val_loader, test_loader, pressure_norm, flow_norm, device ,_ = prepare_training_env(
        args.model_path, args.data, args.hours_analysis,mask_ratio=0.5
    )
    
    net_name = args.data.split('/')[-1]
    
    # eval(model, test_loader, pressure_norm, flow_norm, device, epoch=0)
    
    # --- 这是一个如何在你的训练循环中使用eval的例子 ---
    # for epoch in range(args.epochs):
    #     # ... 你的训练代码 ...
    #     train_one_epoch(...)
        
    #     # 在每个epoch或每隔几个epoch进行评估
    #     if (epoch + 1) % args.log_every_epoch == 0:
    print(f"val_loader: {len(val_loader)}")
    val_loss = eval(model, test_loader, pressure_norm, flow_norm, device, epoch= 1, plot_sample=True , net_name=net_name , fig_path=args.save_path)
    #         print(f"Validation loss at epoch {epoch + 1}: {val_loss}")
    #         # ... 保存模型的逻辑 ...
    
    # 假设这是你模型在测试集上的输出
    # y_true_tensor = torch.tensor([10, 20, 150, 40, 50]) # 150是一个极值
    # y_pred_tensor = torch.tensor([12, 18, 120, 43, 48]) # 模型对极值的预测有较大差距
    
    # # 场景1：没有对数变换
    # print("--- 原始值评估 ---")
    # metrics_raw = regression_metrics(y_true_tensor, y_pred_tensor)
    # for name, value in metrics_raw.items():
    #     print(f"{name}: {value:.4f}")
    
    # # 场景2：你的目标值使用了 log(1+x) 变换
    # # 1. 先对数据进行变换，模拟你的训练过程
    # y_true_log = torch.log(1 + y_true_tensor.float())
    # y_pred_log = torch.log(1 + y_pred_tensor.float()) # 实际上pred是模型直接输出的log值

    # # 2. 定义逆变换函数
    # def inverse_log1p(x):
    #     return np.exp(x) - 1

    # print("\n--- 对数变换后评估 (传入逆变换函数) ---")
    # # 将模型输出的log值和真实的log值传入，函数内部会负责转换回原始尺度再计算
    # metrics_log = regression_metrics(y_true_log, y_pred_log, inverse_transform_func=inverse_log1p)
    # for name, value in metrics_log.items():
    #     print(f"{name}: {value:.4f}")
    pass # 移除旧的调用