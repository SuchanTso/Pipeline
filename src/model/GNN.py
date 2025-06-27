from torch_geometric.nn import MessagePassing
from torch.nn import Sequential, Linear, ReLU
import torch

class HydraulicGNN(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')  # 聚合函数: 均值/求和/最大值
        # 节点更新MLP
        self.node_mlp = Sequential(
            Linear(3, 64),  # 输入: 自身特征 + 邻居聚合特征
            ReLU(),
            Linear(64, 1)   # 输出重建的节点压力
        )
        # 边更新MLP
        self.edge_mlp = Sequential(
            Linear(3, 64),  # 输入: 源节点、目标节点、原始边特征
            ReLU(),
            Linear(64, 1)   # 输出重建的管道流量
        )
    
    def forward(self, data):
        # 节点嵌入初始化
        x = data.x
        edge_index, edge_attr = data.edge_index, data.edge_attr
        # print(f"edge_index: {edge_index}")
        # 消息传递（两轮迭代）
        for _ in range(2):
            # 1. 更新边特征
            edge_updates = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
            # 2. 聚合邻居信息更新节点
            x = self.propagate(edge_index, x=x, edge_attr=edge_updates)
        
        return x, edge_updates
    
    def edge_updater(self, edge_index, x, edge_attr):
        # 拼接源节点、目标节点、原始边特征
        src, dst = edge_index
        edge_input = torch.cat([x[src], x[dst], edge_attr], dim=1)
        # print(f"edge_input: {edge_input}")
        return self.edge_mlp(edge_input)
    
    def message(self, x_j, edge_attr):
        # 消息 = 邻居节点特征 + 边特征
        return torch.cat([x_j, edge_attr], dim=1)
    
    def update(self, aggr_out, x):
        # 节点更新: 自身特征 + 聚合消息
        node_input = torch.cat([x, aggr_out], dim=1)
        return self.node_mlp(node_input)