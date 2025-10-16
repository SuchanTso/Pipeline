import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import *
from dataset import *
import argparse
from tqdm import tqdm
import numpy as np
from utils import prepare_training_env , unified_loss , scaled_cosine_loss


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model_path",type=str,nargs="?",help="path to model",)
    parser.add_argument("-d","--data",type=str,nargs="?",help="path to epanet data",)
    parser.add_argument("-e","--epochs",type=int,default=100,help="number of epochs to train",)
    parser.add_argument("-l","--log_every_epoch",type=int,default=10,help="log every n epochs",)
    parser.add_argument("--hours_analysis",type=int,default=72,help="hours to analysis",)
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
def evaluate_gma(model, data_loader, device, node_mask_ratio, edge_mask_ratio,
                 pressure_col_idx, flow_col_idx , grad_weight=0):
    model.eval()
    total_loss = 0
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)

            num_nodes_total = batch.num_nodes
            num_edges_total = batch.num_edges
            
            num_masked_nodes = int(num_nodes_total * node_mask_ratio)
            num_masked_edges = int(num_edges_total * edge_mask_ratio)
            
            reservoir_indices = (batch.node_type == 1).nonzero(as_tuple=True)[0].tolist()
            
            # --- 核心修改 1: 确保索引是Tensor ---
            masked_node_indices_list = select_random_indices(num_nodes_total, num_masked_nodes, reservoir_indices, seed=42)
            masked_edge_indices_list = select_random_indices(num_edges_total, num_masked_edges, [], seed=42)
            
            masked_node_indices = torch.tensor(masked_node_indices_list, dtype=torch.long, device=device)
            masked_edge_indices = torch.tensor(masked_edge_indices_list, dtype=torch.long, device=device)
            
            reconstructed_pressures, reconstructed_flows = model(batch, masked_node_indices, masked_edge_indices)
            
            true_pressures = batch.x_dynamic
            true_flows =batch.edge_attr_dynamic

            loss_node = 0.0
            # 确保列表不为空再计算损失
            if masked_node_indices.numel() > 0:
                pred = reconstructed_pressures[masked_node_indices]
                true = true_pressures[masked_node_indices]
                loss_node = loss_fn(pred, true)
                
                src , dst = batch.edge_index
                true_grads = true_pressures[src] - true_pressures[dst]
                # loss_grad = nn.MSELoss()(reconstructed_grads[masked_edge_indices] , true_grads[masked_edge_indices])
            
            loss_edge = 0.0
            if masked_edge_indices.numel() > 0:
                pred = reconstructed_flows[masked_edge_indices]
                true = true_flows[masked_edge_indices]
                loss_edge = loss_fn(pred, true)

            loss = loss_node + loss_edge 
            total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss

    return total_loss / len(data_loader)
def pretrain_gma(model, optimizer, train_loader, val_loader, epochs, device, ckpt_path, 
                 mask_ratio_range=(0.35, 0.95), scheduler=None, log_every_epoch=1,
                 pressure_col_idx=None, flow_col_idx=None): # 新增列索引参数
    """
    用于预训练 GraphMaskedAutoencoder 的主函数。
    已适配 PyG DataLoader 的 Batch 对象。
    """
    # if pressure_col_idx is None or flow_col_idx is None:
    #     raise ValueError("pressure_col_idx and flow_col_idx must be provided.")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    loss_fn = nn.HuberLoss(delta=1.0) # 使用自定义的损失函数

    iterator = tqdm(range(epochs), desc='Pre-training GMA', total=epochs)
    
    model_name = ckpt_path.split('/')[-1].replace('.pt','')
    
    for epoch in iterator:
        model.train()
        epoch_train_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # --- 动态生成用于训练的掩码 (作用于整个Batch) ---
            node_mask_ratio = random.uniform(*mask_ratio_range)
            edge_mask_ratio = random.uniform(*mask_ratio_range)
            # node_mask_ratio = 0.5
            # edge_mask_ratio = 0.5
            w = 0.1
            
            num_masked_nodes = int(batch.num_nodes * node_mask_ratio)
            num_masked_edges = int(batch.num_edges * edge_mask_ratio)
            
            reservoir_indices = (batch.node_type == 1).nonzero(as_tuple=True)[0]
            masked_node_indices_list = select_random_indices(batch.num_nodes, num_masked_nodes, reservoir_indices)
            masked_edge_indices_list = select_random_indices(batch.num_edges, num_masked_edges, [])
            
            masked_node_indices = torch.tensor(masked_node_indices_list, dtype=torch.long, device=device)
            masked_edge_indices = torch.tensor(masked_edge_indices_list, dtype=torch.long, device=device)

            reconstructed_pressures, reconstructed_flows = model(batch, masked_node_indices, masked_edge_indices)
            
            true_pressures = batch.x_dynamic
            true_flows =batch.edge_attr_dynamic
            # print(f"pred_node: {reconstructed_pressures} , true_node: {true_pressures}")

            loss_node = 0.0
            if masked_node_indices.numel() > 0:
                pred = reconstructed_pressures[masked_node_indices]
                true = true_pressures[masked_node_indices]
                loss_node = loss_fn(pred, true)
                
            loss_edge = 0.0
            if masked_edge_indices.numel() > 0:
                pred = reconstructed_flows[masked_edge_indices]
                true = true_flows[masked_edge_indices]
                loss_edge = loss_fn(pred, true)

            loss = loss_node + loss_edge

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- 验证部分 ---
        if epoch % log_every_epoch == 0:
            avg_val_loss = evaluate_gma(model, val_loader, device, 
                                        node_mask_ratio=0.75, edge_mask_ratio=0.75,
                                        pressure_col_idx=pressure_col_idx, flow_col_idx=flow_col_idx,
                                        grad_weight=w)
            val_losses.append(avg_val_loss)
            
            if scheduler:
                scheduler.step(avg_val_loss)

            iterator.set_description(
                f'Epoch: {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}'
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), ckpt_path)
            #     torch.save(model.encoder.state_dict(), ckpt_path.replace('.pt', '_encoder.pt'))
        else:
            if val_losses: val_losses.append(val_losses[-1])
            else: val_losses.append(None)
            
        plot_loss(train_losses, val_losses, f'pretrain_{model_name}_loss.png') # plot_loss函数无需修改
    
    return train_losses, val_losses
            
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    model, optimizer , train_loader , val_loader , test_loader , pressure_norm , flow_norm , device  , scheduler = prepare_training_env(args.model_path, 
                                                                                                                           args.data, 
                                                                                                                           args.hours_analysis,
                                                                                                                           mask_ratio=0.5)
    if len(optimizer) == 1:
        optimizer = optimizer[0]
    pretrain_gma(model, 
             optimizer, 
             train_loader , 
             val_loader, 
             args.epochs , 
             device ,
             args.model_path , 
             scheduler=None,
             log_every_epoch=args.log_every_epoch )