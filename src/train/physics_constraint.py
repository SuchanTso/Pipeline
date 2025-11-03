# file: physics_loss.py

import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter

def continuity_loss_undirected(p_real_pred, q_real_pred_magnitude, batch_data):
    """
    在无向图上计算质量守恒损失。
    
    Args:
        p_real_pred (Tensor): [num_nodes] 真实尺度的压力预测。
        q_real_pred_magnitude (Tensor): [num_undirected_edges] 真实尺度的流量大小预测。
        batch_data: 包含无向图 edge_index 和其他静态属性。
    """
    edge_index = batch_data.edge_index
    num_nodes = batch_data.num_nodes
    
    # --- 1. 动态确定流量方向 ---
    # 计算每个节点的总水头 H = p + z
    total_head_pred = p_real_pred.flatten() + batch_data.elevations_real.flatten()
    
    # 获取每条边的起点和终点
    # edge_index 的形状是 [2, num_edges]，row是起点，col是终点
    row, col = edge_index
    
    # 计算水头差
    head_diff = total_head_pred[row] - total_head_pred[col]
    
    # 确定方向：如果 head_diff > 0, 水从 row->col, 流量为正。
    #           如果 head_diff < 0, 水从 col->row, 流量应视为从 row->col 为负。
    # sign() 会完美地处理这个问题。
    flow_direction = torch.sign(head_diff)
    
    # 为流量大小赋予方向
    # [关键]：这里的 q_real_pred_magnitude 必须是模型预测的流量大小（非负）
    # 如果你的模型预测的是有符号的流量，需要先取绝对值
    q_directed = torch.abs(q_real_pred_magnitude.flatten()) * flow_direction
    
    # --- 2. 使用 scatter 聚合 ---
    # 现在 q_directed 是一个有方向的流量向量，其方向与 edge_index (row->col) 一致
    # 流出 row 的流量
    flow_out = scatter(q_directed, row, dim=0, dim_size=num_nodes, reduce='sum')
    # 流入 col 的流量 (等价于流出col的流量的相反数)
    flow_in = scatter(-q_directed, col, dim=0, dim_size=num_nodes, reduce='sum')
    
    # 净流出 = 所有以该节点为起点的有向流量之和 + 所有以该节点为终点的有向流量之和
    # (因为流向终点的流量，对于终点来说是流入，即负的流出)
    net_flow_out = flow_out + flow_in
    
    scale = torch.var(batch_data.demands_real) + 1e-8
    
    # 损失是净流出量和节点需求之间的差异
    loss = F.mse_loss(net_flow_out, batch_data.demands_real.flatten()) / scale
    
    return loss

def hazen_williams_loss_undirected(p_real_pred, q_real_pred_magnitude, batch_data, unit_conversion_factor=10.67):
    """
    在无向图上计算能量守恒损失。
    """
    epsilon = 1e-8
    edge_index = batch_data.edge_index
    
    is_pipe_mask = (batch_data.diameters_real > 0) & (batch_data.roughnesses_real > 0)
    
    # 如果一个batch里全是水泵（虽然不太可能），直接返回0损失
    if not torch.any(is_pipe_mask):
        return torch.tensor(0.0, device=p_real_pred.device)

    # --- 2. 只在普通管道上计算损失 ---
    # 使用掩码来筛选所有相关的张量
    
    # 计算阻力系数 K (只为普通管道)
    K = (unit_conversion_factor * batch_data.lengths_real[is_pipe_mask]) / \
        (torch.pow(batch_data.roughnesses_real[is_pipe_mask], 1.852) * 
         torch.pow(batch_data.diameters_real[is_pipe_mask], 4.87) + epsilon)

    # 根据流量大小计算理论水头损失的大小 (RHS)
    q_abs_pipes = torch.abs(q_real_pred_magnitude.flatten()[is_pipe_mask])
    head_loss_from_flow_magnitude = K * torch.pow(q_abs_pipes, 1.852)

    # 根据压力和高程计算实际水头差的大小 (LHS)
    total_head_pred = p_real_pred.flatten() + batch_data.elevations_real.flatten()
    
    # 筛选出普通管道的边索引
    edge_index_pipes = edge_index[:, is_pipe_mask]
    row, col = edge_index_pipes
    
    head_src = total_head_pred[row]
    head_dst = total_head_pred[col]
    head_loss_from_pressure_magnitude = torch.abs(head_src - head_dst)
    
    scale = torch.var(head_loss_from_pressure_magnitude) + epsilon
    
    # 计算两者之间的均方误差
    loss = F.mse_loss(head_loss_from_pressure_magnitude, head_loss_from_flow_magnitude) / scale
    
    return loss

def calculate_physics_loss_undirected(
    p_real_pred, q_real_pred,
    batch_data,
    lambda_cont=1.0,
    lambda_eng=0.1):
    """
    计算总的物理损失 (无向图版本)。
    """
    # [重要] 假设 q_real_pred 是模型直接输出，可正可负。
    # 我们用它的绝对值作为流量大小。
    q_real_pred_magnitude = torch.abs(q_real_pred)

    

    loss_c = continuity_loss_undirected(p_real_pred, q_real_pred_magnitude, batch_data)
    loss_e = hazen_williams_loss_undirected(p_real_pred, q_real_pred_magnitude, batch_data)
    
    return lambda_cont * loss_c + lambda_eng * loss_e

def project_gradients(g_phys, g_data):
    """
    将物理梯度 g_phys 投影到数据梯度 g_data 的正交空间上。
    这个版本始终移除平行分量，以确保物理约束的更新
    永远不会干扰数据学习任务的梯度方向和步长。
    
    Args:
        g_phys (iterable): 物理梯度的可迭代对象。
        g_data (iterable): 数据梯度的可迭代对象。
        
    Returns:
        iterable: 投影后的、与数据梯度正交的物理梯度。
    """
    # 将梯度列表展平为单个向量
    g_phys_flat = torch.cat([grad.flatten() for grad in g_phys])
    g_data_flat = torch.cat([grad.flatten() for grad in g_data])

    # 计算 g_phys 在 g_data 上的投影分量: proj(g_phys onto g_data)
    dot_product = torch.dot(g_phys_flat, g_data_flat)
    data_norm_sq = torch.dot(g_data_flat, g_data_flat)
    
    # 避免除以零
    if data_norm_sq == 0:
        # 如果数据梯度为0，物理梯度无需投影
        g_phys_projected_flat = g_phys_flat
    else:
        proj_vector = (dot_product / data_norm_sq) * g_data_flat
    
        # [核心修改] 无论梯度是否冲突，都减去平行分量，只保留正交分量
        g_phys_projected_flat = g_phys_flat - proj_vector
        
    # 将扁平化的正交梯度恢复成原始参数的形状
    g_phys_projected = []
    offset = 0
    # 使用 model.parameters() 来获取正确的形状和设备信息
    # 注意：这里需要传入模型参数列表来恢复形状，我们假设g_data的形状和模型参数一致
    ref_params = [p for p in g_data] # g_data is an iterable, convert to list if needed
    
    for param in ref_params:
        numel = param.numel()
        # 确保恢复的张量在正确的设备上
        g_phys_projected.append(g_phys_projected_flat[offset:offset+numel].view_as(param))
        offset += numel
        
    return g_phys_projected