import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import *
# from loss import physics_loss
from dataset import *
import argparse
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset



def create_data_loader(raw_data, batch_size=32, train_ratio=0.7, val_ratio=0.15, window_size=12):
    # --- Step 1: 预处理和标准化 ---
    pretrain_data = PretrainDataset_ET(raw_data)
    
    # --- Step 2: 创建时序窗口数据集 ---
    main_dataset = EST_MAE_Dataset(pretrain_data.processed_data_list, window_size=window_size)
    
    # --- Step 3: 划分训练/验证/测试集 ---
    total_len = len(main_dataset)
    train_end = int(total_len * train_ratio)
    val_end = train_end + int(total_len * val_ratio)

    train_dataset = Subset(main_dataset, range(0, train_end))
    val_dataset   = Subset(main_dataset, range(train_end, val_end))
    test_dataset  = Subset(main_dataset, range(val_end, total_len))

    # 使用 PyG 的 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 返回加载器和标准化器，以便在训练和评估时使用
    return train_loader, val_loader, test_loader, pretrain_data.pressure_norm, pretrain_data.flow_norm

def load_model(model_name , model_path):
    optimizers = []
    initial_lr = 1e-4
    if model_name == "TGCN_MessageCoupling":
        model = TGCN_MessageCoupling(node_in_feats=4 , edge_in_feats=4 , gcn_hidden=32, node_gru_hidden=32 ,edge_gru_hidden=32 ,edge_mlp_hidden=32, out_node_feats=1,out_edge_feats=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        optimizers.append(optimizer)
    elif model_name == "TGCN_MessageCoupling_Deep":
        model = TGCN_MessageCoupling_Deep(node_in_feats=4 , edge_in_feats=5 , gcn_hidden=64, node_gru_hidden=64 ,edge_gru_hidden=64 ,edge_mlp_hidden=64, out_node_feats=1,out_edge_feats=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        optimizers.append(optimizer)
    elif model_name == "TGCN_PrimalDual":
        model = TGCN_PrimalDual(node_in_feats=4 , edge_in_feats=5 , gcn_hidden=32, node_gru_hidden=32 ,edge_gru_hidden=32,edge_gcn_hidden=32 ,edge_mlp_hidden=32, out_node_feats=1,out_edge_feats=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        optimizers.append(optimizer)
    elif model_name == "GAT_RegressionModel":
        gat_config = {
            'n_layers': 1,  # GAT层数（1个输入层 + 1个输出层）
            'n_heads': [8], # 输入层8个头，输出层1个头
            'node_hid_feats': 16, # 每个头的隐藏维度
            'node_out_feats': 64, # 节点嵌入的维度
            'edge_hid_feats': 16, # 边特征变换后的维度
            'alpha': 0.2
        }
        model = GAT_RegressionModel(
            node_in_feats=4,
            edge_in_feats=5,
            node_gru_hidden=64,
            edge_gru_hidden=64,
            gat_config=gat_config,
            edge_mlp_hidden=128,
            dropout=0.3
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        optimizers.append(optimizer)
        
    elif model_name == "TGAT_MessageCoupling_Deep":
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
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        optimizers.append(optimizer)
        
    elif model_name == "T_GraphFormer":
        model = T_GraphFormer(
            node_in_feats=5,
            edge_in_feats=5,
            d_model=256,
            num_heads=4,
            num_transformer_layers=4,
            gru_hidden=256,
            gru_layers=2
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        optimizers.append(optimizer)
        
    elif model_name == "GraphMaskedAutoencoder":
        model = GraphMaskedAutoencoder(
            node_in_feats=4,
            edge_in_feats=4,
            d_model=256,
            num_heads=4,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dropout=0.2
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        optimizers.append(optimizer)
        
    elif model_name == "W_GraphMAE":
        model = W_GraphMAE(
            node_in_feats=4,
            edge_in_feats=4,
            d_model=128,
            num_heads=4,
            num_encoder_layers=8, # 深编码器
            num_decoder_layers=4  # 浅解码器
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        optimizers.append(optimizer)
        
    elif model_name == "DecoupledFusionEGT_GraphMAE":
        model = DecoupledFusionEGT_GraphMAE(
            node_in_feats=4,
            edge_in_feats=4,
            d_model=128,          # 嵌入维度
            num_heads=4,          # 注意力头数
            num_encoder_layers=8, # 编码器层数
            num_decoder_layers=4, # 解码器层数
            dropout=0.1
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        optimizers.append(optimizer)
        
    elif model_name == "FinalPipeline":
        model = FinalPipeline(
            stage_one_args={
                'static_node_feats':3,
                'dynamic_node_feats':1,
                'static_edge_feats':3,
                'dynamic_edge_feats':1,
                'd_model':128,
                'num_heads':4,
                'num_encoder_layers':8, # 深编码器
                'num_decoder_layers':4,  # 浅解码器
                'dropout':0.1,
                'max_spd':10
            },
            stage_two_args={
                'node_feature_dim':4,
                'edge_feature_dim':3,
                'num_layers': 3,           # Transformer的层数，可以调参
                'd_model':128,
                'num_heads':4,
                'dropout':0.1
            }
        )
        optimizer_1 = torch.optim.AdamW(model.stage_one.parameters(), lr=initial_lr, weight_decay=1e-5)
        optimizer_2 = torch.optim.AdamW(model.stage_two.parameters(), lr=initial_lr, weight_decay=1e-5)
        optimizers.append(optimizer_1)
        optimizers.append(optimizer_2)
        
    elif model_name == "EST_MAE":
        model = EST_MAE(
            static_node_feats=3,
            dynamic_node_feats=1,
            static_edge_feats=3,
            dynamic_edge_feats=1,
            d_model=128,
            num_heads=4,
            # num_encoder_layers=8, # 深编码器
            num_decoder_layers=4,  # 浅解码器
            dropout=0.1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        optimizers.append(optimizer)
    
    elif model_name == "EST_MAE_v5":
        model = EST_MAE_v5(
            static_node_feats=3,
            dynamic_node_feats=1,
            static_edge_feats=3,
            dynamic_edge_feats=1,
            d_model=128,
            num_heads=4,
            # num_encoder_layers=8, # 深编码器
            num_decoder_layers=4,  # 浅解码器
            dropout=0.1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        optimizers.append(optimizer)
    elif model_name == "EST_GEN_MAE":
        model = EST_ProGen(
            static_node_feats=3,
            dynamic_node_feats=1,
            static_edge_feats=3,
            dynamic_edge_feats=1,
            d_model=128,
            num_heads=4,
            # num_encoder_layers=8, # 深编码器
            num_decoder_layers=4,  # 浅解码
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    if os.path.exists(model_path):
        print(f"load model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    return model , optimizers
def load_data(data_name , data_path, hr ,batch_size , mask_ratio=0.2):
    pressure_norm = LogZScoreNormalizer()
    flow_norm = LogZScoreNormalizer()
    epa = EpytHelper(data_path , hr)
    if data_name == "PretrainDataset":
        dataset = PretrainDataset(
            raw_data=epa.get_raw_data(),
            fit_ratio=1.0,
            fit_node_mask_ratio=mask_ratio,
            pressure_norm=pressure_norm,
            flow_norm=flow_norm,
            fit_pipe_mask_ratio=mask_ratio
        )
        train_loader , val_loader , test_loader = dataset.gen_train_loader(batch_size=batch_size)
    elif data_name == "PretrainDataset_ET":
        dataset = PretrainDataset_ET(
            raw_data=epa.get_raw_data(),
            fit_ratio=1.0,
            fit_node_mask_ratio=mask_ratio,
            pressure_norm=pressure_norm,
            flow_norm=flow_norm,
            fit_pipe_mask_ratio=mask_ratio
        )
        main_dataset = EST_MAE_Dataset(dataset.processed_data_list, window_size=6)
        train_loader , val_loader , test_loader = main_dataset.gen_train_loader(batch_size=batch_size)
    elif data_name == "WaterEPANetDataset":
            dataset = WaterEPANetDataset(data_path, 
                hr,
                pressure_normalizer=pressure_norm,
                flow_normalizer= flow_norm,
                fit_node_mask_ratio=mask_ratio,
                fit_pipe_mask_ratio=mask_ratio,
                augment_node_mask_ratio_range=(mask_ratio - 0.3 , mask_ratio),
                augment_pipe_mask_ratio_range=(mask_ratio - 0.3 , mask_ratio),
                window_size=6)
            train_loader , val_loader , test_loader = dataset.gen_train_loader(batch_size=batch_size)
    else:
        raise ValueError(f"Unknown dataset name: {data_name}")
    epa.destroy()
    
    return train_loader , val_loader , test_loader , pressure_norm , flow_norm

def prepare_training_env(ckpt_path , data_path , hr , mask_ratio=0.2):
    model , optimizers = load_model("EST_MAE_v5",ckpt_path)

    # 2. 定义学习率调度器
    # 监控 'val_loss'，当它停止下降时，降低学习率
    # scheduler = ReduceLROnPlateau(
    #     optimizer_1,
    #     mode='min',          # 'min' 模式，监控的指标越小越好 (如loss)
    #     factor=0.5,          # 学习率衰减的乘法因子 (new_lr = lr * factor)，0.5表示减半
    #     patience=10,         # 容忍10个epoch指标没有改善
    #     # verbose=True,        # 当学习率更新时，在控制台打印一条消息
    #     threshold=0.0001,    # 用于衡量“改善”的阈值
    #     threshold_mode='rel',# 'rel'表示相对变化，'abs'表示绝对变化
    #     cooldown=5,          # 降低学习率后，冷却5个epoch再重新开始监控
    #     min_lr=1e-7          # 学习率的下限
    # )    
    train_loader , val_loader , test_loader , pressure_norm , flow_norm = load_data("PretrainDataset_ET" , data_path , hr , 1 , mask_ratio)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, optimizers ,train_loader , val_loader , test_loader , pressure_norm , flow_norm , device , None

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

def scaled_cosine_loss(x, y, alpha=2.0): # beta是可调超参数
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

def visualize_overfitting_progress(model, fixed_batch, time_steps, mask_ratio, device, dynamic_normalizer, step, losses, r2_scores):
    """
    在过拟合训练的某个步骤，对模型的性能和内部状态进行全面可视化。

    Args:
        model (nn.Module): 正在训练的模型.
        fixed_batch (Data): 用于过拟合的固定数据批次.
        time_steps (Tensor): 时间步.
        mask_ratio (float): 掩码比例.
        device (torch.device): 设备.
        dynamic_normalizer: 用于反归一化的Normalizer.
        step (int): 当前的训练步数.
        losses (list): 记录了每一步损失的列表.
        r2_scores (list): 记录了每一步R²的列表.
    """
    print(f"\n--- Running Visualization at Step {step} ---")
    model.eval() # 切换到评估模式进行可视化

    # --- 1. 准备数据 (与 evaluate 函数类似) ---
    data = fixed_batch
    t_max = time_steps[0]
    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes, device=device) # 每次可视化可以用不同的mask
    masked_node_count = int(num_nodes * mask_ratio)
    
    if masked_node_count == 0:
        print("No nodes masked, skipping visualization.")
        return

    node_masked_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    node_masked_mask[perm[:masked_node_count]] = True
    node_obs_mask = ~node_masked_mask

    x_masked_true_norm = data.x_dynamic_window[node_masked_mask, -1, :]
    x_obs_true_norm = data.x_dynamic_window[node_obs_mask, -1, :]
    initial_noise = torch.randn_like(x_masked_true_norm) * t_max

    canvas_noise = torch.zeros(num_nodes, data.x_dynamic_window.size(-1), device=device)
    canvas_noise[node_obs_mask] = x_obs_true_norm
    canvas_noise[node_masked_mask] = initial_noise

    # --- 2. 获取模型预测 ---
    with torch.no_grad():
        prediction_norm = model(data, node_obs_mask, canvas_noise, t_max, use_teacher=True, train_encoder=False)
    
    pred_masked_norm = prediction_norm[node_masked_mask]

    # --- 3. 反归一化 ---
    pred_masked_orig = dynamic_normalizer.inverse_transform(pred_masked_norm).cpu().numpy().flatten()
    x_masked_true_orig = dynamic_normalizer.inverse_transform(x_masked_true_norm).cpu().numpy().flatten()
    initial_noise_orig = dynamic_normalizer.inverse_transform(initial_noise).cpu().numpy().flatten()

    # --- 4. 开始绘图 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Overfitting Diagnosis - Step {step}', fontsize=16)

    # a) 损失和R²曲线
    ax = axes[0, 0]
    ax.plot(losses, label='Loss', color='tab:red')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss', color='tab:red')
    ax.tick_params(axis='y', labelcolor='tab:red')
    ax.grid(True)
    
    ax2 = ax.twinx()
    ax2.plot(r2_scores, label='R² Score', color='tab:blue')
    ax2.set_ylabel('R² Score', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylim(-1.5, 1.0) # 固定R2的范围
    ax.set_title('Loss and R² Curve')

    # b) 预测值 vs. 真实值 散点图
    ax = axes[0, 1]
    ax.scatter(x_masked_true_orig, pred_masked_orig, alpha=0.5, label='Predictions')
    min_val = min(x_masked_true_orig.min(), pred_masked_orig.min())
    max_val = max(x_masked_true_orig.max(), pred_masked_orig.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Prediction vs. True Scatter Plot')
    ax.legend()
    ax.grid(True)

    # c) 预测值、真实值、噪声值的分布
    ax = axes[1, 0]
    ax.hist(x_masked_true_orig, bins=30, alpha=0.6, label='True Values', density=True)
    ax.hist(pred_masked_orig, bins=30, alpha=0.6, label='Predicted Values', density=True)
    ax.hist(initial_noise_orig, bins=30, alpha=0.4, label='Initial Noise (denorm)', density=True)
    ax.set_title('Distribution of Values')
    ax.set_xlabel('Pressure')
    ax.set_ylabel('Density')
    ax.legend()
    
    # d) 预测值在图上的空间分布
    ax = axes[1, 1]
    # 创建一个完整的节点值向量用于可视化
    full_node_values = np.zeros(num_nodes)
    full_node_values[node_masked_mask.cpu().numpy()] = pred_masked_orig
    # 观测节点的值也反归一化以供参考
    obs_values_orig = dynamic_normalizer.inverse_transform(x_obs_true_norm).cpu().numpy().flatten()
    full_node_values[node_obs_mask.cpu().numpy()] = obs_values_orig

    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42) # 使用固定种子保证布局一致
    
    nodes = nx.draw_networkx_nodes(G, pos, node_color=full_node_values, cmap=plt.cm.viridis, node_size=50, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
    plt.colorbar(nodes, ax=ax, label='Predicted/Observed Pressure')
    ax.set_title('Spatial Distribution of Predictions')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'debug_overfitting_step_{step:04d}.png')
    plt.close()
    
    model.train() # 恢复到训练模式