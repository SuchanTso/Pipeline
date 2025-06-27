import torch
def physics_loss(pred_pressure, pred_flow, real_data):
    # 质量守恒: 节点流入流出平衡
    inflow = pred_flow[real_data.edge_index[0]].sum(dim=0)
    outflow = pred_flow[real_data.edge_index[1]].sum(dim=0)
    mass_conservation = torch.abs(inflow - outflow).mean()
    
    # 能量方程: 压力差驱动流量
    src_pressure = pred_pressure[real_data.edge_index[0]]
    dst_pressure = pred_pressure[real_data.edge_index[1]]
    pressure_drop = src_pressure - dst_pressure
    energy_loss = torch.abs(pred_flow - 0.01 * pressure_drop).mean()  # 简化的线性关系
    
    return mass_conservation + energy_loss