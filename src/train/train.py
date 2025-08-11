import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import GNN_ChebConv , TGCN_MessageCoupling , TGCN_MessageCoupling_Deep 
from loss import physics_loss
from dataset import WaterEPANetDataset , GraphNormalizer , ZScoreNormalizer , LogZScoreNormalizer
import os
import argparse
from tqdm import tqdm
import numpy as np
from utils import prepare_training_env , unified_loss


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        nargs="?",
        help="path to model",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        nargs="?",
        help="path to epanet data",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train",
    )
    parser.add_argument(
        "-l",
        "--log_every_epoch",
        type=int,
        default=10,
        help="log every n epochs",
    )
    parser.add_argument(
        "--hours_analysis",
        type=int,
        default=72,
        help="hours to analysis",
    )
    
    
    return parser

def plot_loss(train_losses, val_losses , fig_path):
    # ----------------- 步骤 3: 训练结束后绘制损失曲线 -----------------
    plt.figure(figsize=(10, 5))
    # 筛选掉None值，以便绘图
    val_epochs = [i for i, v in enumerate(val_losses) if v is not None]
    valid_val_losses = [v for v in val_losses if v is not None]

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_epochs, valid_val_losses, label='Validation Loss', linestyle='--')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_path)
def evaluate(model, data_loader, device):
    """计算模型在给定数据集上的损失"""
    model.eval()  # 非常重要：将模型切换到评估模式（禁用Dropout等）
    total_loss = 0
    with torch.no_grad():  # 非常重要：在评估时不需要计算梯度
        for batch in data_loader:
            x_seq, edge_index, edge_attr_seq, y_node, y_edge, _, _ = batch
            x_seq = x_seq[0].to(device)
            edge_index = edge_index[0].to(device)
            y_node = y_node[0].to(device)
            y_edge = y_edge[0].to(device)
            edge_attr = edge_attr_seq[0].to(device)

            pred_nodes, pred_edges = model(x_seq, edge_index, edge_attr)
            
            # 使用与训练时相同的损失函数
            loss = unified_loss(pred_nodes, y_node, pred_edges, y_edge)
            total_loss += loss.item()

    return total_loss / len(data_loader) # 返回该epoch的平均损失

# ----------------- 步骤 2: 修改你的主训练函数 -----------------
def training(model, optimizer, train_loader, val_loader, epochs, device, ckpt_path, scheduler=None ,  log_every_epoch=1):
    # 用于记录损失的列表
    train_losses = []
    val_losses = []

    # 用于早停的变量（可选，但推荐）
    best_val_loss = float('inf')

    iterator = tqdm(range(epochs), desc='Training', total=epochs)
    
    for epoch in iterator:
        # --- 训练部分 ---
        model.train()  # 确保模型处于训练模式
        epoch_train_loss = 0
        for batch in train_loader:
            x_seq, edge_index, edge_attr_seq, y_node, y_edge, _, _ = batch
            x_seq = x_seq[0].to(device)
            edge_index = edge_index[0].to(device)
            y_node = y_node[0].to(device)
            y_edge = y_edge[0].to(device)
            edge_attr = edge_attr_seq[0].to(device)

            optimizer.zero_grad()
            
            pred_nodes, pred_edges = model(x_seq, edge_index, edge_attr)
            
            loss = unified_loss(pred_nodes, y_node, pred_edges, y_edge)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # 计算并记录当前epoch的平均训练损失
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- 验证部分 ---
        # 我们在每个epoch结束时进行验证
        if epoch % log_every_epoch == 0:
            avg_val_loss = evaluate(model, val_loader, device)
            val_losses.append(avg_val_loss)
            if scheduler:
                scheduler.step(avg_val_loss)

            # 更新tqdm进度条的描述信息
            iterator.set_description(
                f'Epoch: {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}'
            )

            # 保存模型（只保存在验证集上表现最好的模型）
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), ckpt_path)
                # print(f"Model saved at epoch {epoch+1} with validation loss {best_val_loss:.4f}")
        else:
            # 如果不验证，也记录一个占位符或前一个值，以保持列表长度一致
            if val_losses: # 如果列表不为空
                val_losses.append(val_losses[-1])
            else: # 如果是第一个epoch且不验证
                val_losses.append(None)

        plot_loss(train_losses, val_losses, 'loss.png')
    
    
    return train_losses, val_losses
# def training(model , optimizer, train_loader , val_loader , epochs , device ,ckpt_path, log_every_epoch=10):
#     model.train()
#     total_loss = 0
#     iterator = tqdm(range(epochs), desc=f'train_loss:{total_loss}', total= epochs)
#     for i , epoch in enumerate(iterator):
#         for batch in train_loader:
            
#             x_seq, edge_index, edge_attr_seq , y_node, y_edge , _ , _ = batch  # x_seq: [B, T, N, F]
#             x_seq = x_seq[0].to(device)          # [T, N, F]
#             edge_index = edge_index[0].to(device)  # [2, E]
#             y_node = y_node[0].to(device)        # [N, 1]
#             y_edge = y_edge[0].to(device)        # [E, 1]
#             edge_attr = edge_attr_seq[0].to(device)

#             # 调用模型
#             pred_nodes , pred_edges = model(x_seq, edge_index , edge_attr)  # pred_node: [B, N, 1]
#             # pred_nodes = model(batch)
#             # 常规重建损失
#             loss_node = torch.nn.MSELoss()(pred_nodes, y_node) + torch.nn.MSELoss()(pred_edges, y_edge)
#             # loss_edge = torch.nn.MSELoss()(pred_edges, batch.edge_attr)
#             # 物理约束损失
#             # loss_physics = physics_loss(pred_nodes, pred_edges, batch)
#             # 总损失
#             total_loss = loss_node
            
#             optimizer.zero_grad()
#             total_loss.backward()
#             total_norm = 0
#             # for p in model.parameters():
#             #     if p.grad is not None:
#             #         param_norm = p.grad.data.norm(2)
#             #         total_norm += param_norm.item() ** 2
#             # total_norm = total_norm ** 0.5
#             # print(f"Gradient Norm: {total_norm}")
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
#             optimizer.step()
            
#             if i % log_every_epoch == 0:
#                 # print(f"pred: {pred_nodes[:5]}, true: {batch.y_node[:5]}")
#                 iterator.set_description(f'train_loss:{total_loss.item():.2f}')
#                 torch.save(model.state_dict(), ckpt_path)
                
def eval(model , test_loader , pressure_norm , flow_norm , device):
    model.eval()
    total_loss = 0
    np.set_printoptions(precision=10, suppress=True)
    with torch.no_grad():
        for batch in test_loader:
            x_seq, edge_index, edge_attr, y_node, y_edge , node_indices , edge_indices = batch  # x_seq: [B, T, N, F]
            x_seq = x_seq[0].to(device)          # [T, N, F]
            edge_index = edge_index[0].to(device)  # [2, E]
            y_node = y_node[0].to(device)        # [N, 1]
            y_edge = y_edge[0].to(device)        # [E, 1]
            edge_attr = edge_attr[0].to(device)

            # 调用模型
            pred_nodes , pred_edges = model(x_seq, edge_index , edge_attr)  # pred_node: [B, N, 1]
            inverse_pred_nodes = pressure_norm.inverse_transform(pred_nodes.cpu())
            inverse_real_nodes = pressure_norm.inverse_transform(y_node.cpu())
            print(f"pred_nodes: {inverse_pred_nodes[node_indices].numpy()}")
            print(f"real_nodes: {inverse_real_nodes[node_indices].numpy()}")
            inverse_pred_edge = flow_norm.inverse_transform(pred_edges.cpu())
            inverse_real_edge = flow_norm.inverse_transform(y_edge.cpu())
            print(f"pred_edge: {inverse_pred_edge[edge_indices].numpy()}")
            print(f"real_edge: {inverse_real_edge[edge_indices].numpy()}")
            loss_node = unified_loss(pred_nodes, y_node, pred_edges, y_edge)
            total_loss += loss_node.item()
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss
            
            
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    model, optimizer , train_loader , val_loader , test_loader , pressure_norm , flow_norm , device  , scheduler = prepare_training_env(args.model_path, 
                                                                                                                           args.data, 
                                                                                                                           args.hours_analysis,
                                                                                                                           mask_ratio=0.5)
    training(model, 
             optimizer, 
             train_loader , 
             val_loader, 
             args.epochs , 
             device ,
             args.model_path , 
             scheduler=None,
             log_every_epoch=args.log_every_epoch )
    eval(model, val_loader , pressure_norm , flow_norm ,device)