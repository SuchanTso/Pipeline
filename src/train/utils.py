import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import GNN_ChebConv , TGCN_MessageCoupling , TGCN_MessageCoupling_Deep , TGCN_PrimalDual , GAT_RegressionModel , TGAT_MessageCoupling_Deep
from loss import physics_loss
from dataset import WaterEPANetDataset , GraphNormalizer , ZScoreNormalizer ,LogZScoreNormalizer
import os
import argparse
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn


def prepare_training_env(ckpt_path , data_path , hr , mask_ratio=0.2):
    #node_in_feats, edge_in_feats, gcn_hidden, node_gru_hidden, edge_gru_hidden, edge_mlp_hidden, out_node_feats=1, out_edge_feats=1
    # model = TGCN_MessageCoupling(node_in_feats=4 , edge_in_feats=4 , gcn_hidden=32, node_gru_hidden=32 ,edge_gru_hidden=32 ,edge_mlp_hidden=32, out_node_feats=1,out_edge_feats=1)
    # model = TGCN_MessageCoupling_Deep(node_in_feats=4 , edge_in_feats=5 , gcn_hidden=64, node_gru_hidden=64 ,edge_gru_hidden=64 ,edge_mlp_hidden=64, out_node_feats=1,out_edge_feats=1)
    # model = TGCN_PyG(in_feats=4, gcn_hidden=32, gru_hidden=32, out_feats=1)
    # gat_config = {
    #     'n_layers': 1,  # GAT层数（1个输入层 + 1个输出层）
    #     'n_heads': [8], # 输入层8个头，输出层1个头
    #     'node_hid_feats': 16, # 每个头的隐藏维度
    #     'node_out_feats': 64, # 最终节点嵌入的维度
    #     'edge_hid_feats': 16, # 边特征变换后的维度
    #     'alpha': 0.2
    # }

    # model = GAT_RegressionModel(
    #     node_in_feats=4,
    #     edge_in_feats=5,
    #     node_gru_hidden=64,
    #     edge_gru_hidden=64,
    #     gat_config=gat_config,
    #     edge_mlp_hidden=128,
    #     dropout=0.3
    # )
    model = TGAT_MessageCoupling_Deep(
        node_in_feats=4, 
        edge_in_feats=5, 
        gcn_hidden=64,       # GAT每头的隐藏维度
        node_gru_hidden=128,
        edge_gru_hidden=64,
        edge_mlp_hidden=128,
        gat_layers=2,        # 可以从2层开始
        gru_layers=2,
        dropout_rate=0.2,    # GAT对dropout更敏感，可以适当调高
        heads=4              # 4或8是常见选择
    )
    # model = TGCN_PrimalDual(node_in_feats=4 , edge_in_feats=5 , gcn_hidden=32, node_gru_hidden=32 ,edge_gru_hidden=32,edge_gcn_hidden=32 ,edge_mlp_hidden=32, out_node_feats=1,out_edge_feats=1)
    # model = GNN_ChebConv(hid_channels=32, edge_features=3, node_features=3, edge_channels=32, dropout_rate=0.2, CC_K=2,
    #                      emb_aggr='max', depth=2, normalize=True)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        
    initial_lr = 1e-4 
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-5)

    # 2. 定义学习率调度器
    # 监控 'val_loss'，当它停止下降时，降低学习率
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',          # 'min' 模式，监控的指标越小越好 (如loss)
        factor=0.5,          # 学习率衰减的乘法因子 (new_lr = lr * factor)，0.5表示减半
        patience=10,         # 容忍10个epoch指标没有改善
        # verbose=True,        # 当学习率更新时，在控制台打印一条消息
        threshold=0.0001,    # 用于衡量“改善”的阈值
        threshold_mode='rel',# 'rel'表示相对变化，'abs'表示绝对变化
        cooldown=5,          # 降低学习率后，冷却5个epoch再重新开始监控
        min_lr=1e-7          # 学习率的下限
    )    
        
        
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # normalizer = GraphNormalizer()
    # x_norm = ZScoreNormalizer()
    # y_node_normalizer = ZScoreNormalizer()
    # y_edge_normalizer = ZScoreNormalizer()
    pressure_norm = ZScoreNormalizer()
    flow_norm = LogZScoreNormalizer()
    dataset = WaterEPANetDataset(data_path, hr ,pressure_normalizer=pressure_norm , flow_normalizer= flow_norm,masked_ratio=mask_ratio , window_size=6)
    # print(f"dataset load:{dataset[0]}")
    
    train_loader , val_loader , test_loader = dataset.gen_train_loader(batch_size=1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, optimizer ,train_loader , val_loader , test_loader , pressure_norm , flow_norm , device , scheduler

def preprocess_data_with_dual_graph(edge_index):
    """
    一个辅助函数，用于在数据加载阶段计算对偶图的edge_index。
    
    Args:
        edge_index (LongTensor): 原始图的边索引。

    Returns:
        LongTensor: 对偶图的边索引。
    """
    # PyG的LineGraph变换要求图是无向的，且不含自环。
    # 你的edge_index可能已经是无向的，但这里为了确保，我们强制一下。
    # 注意：这可能会改变边的数量和顺序，需要与你的edge_attr对齐。
    # 如果你的图很大，且已经是无向的，可以跳过to_undirected。
    # temp_edge_index, _ = to_undirected(edge_index)
    
    # 创建一个临时的PyG Data对象来使用transform
    temp_data = Data(edge_index=edge_index)
    
    # 初始化LineGraph变换器
    line_graph_transform = LineGraph(force_directed=False) # force_directed=False适用于无向图
    
    # 应用变换
    line_graph_data = line_graph_transform(temp_data)
    
    return line_graph_data.edge_index

def unified_loss(pred_nodes, y_node, pred_edges, y_edge):
    """
    统一的损失函数，计算节点和边的预测损失。
    
    Args:
        pred_nodes (Tensor): 模型预测的节点特征。
        y_node (Tensor): 实际的节点特征。
        pred_edges (Tensor): 模型预测的边特征。
        y_edge (Tensor): 实际的边特征。

    Returns:
        Tensor: 计算得到的总损失。
    """
    criterion = nn.HuberLoss(delta=1.0)
    node_loss = criterion(pred_nodes, y_node)
    edge_loss = criterion(pred_edges, y_edge)
    return node_loss + edge_loss