from torch_geometric.nn import MessagePassing,ChebConv
from torch.nn import Sequential, Linear, ReLU
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.nn.inits import reset
from typing import Union
from torch import Tensor
from torch.nn import Dropout

class GraphEmbed(MessagePassing):
    def __init__(self, x_num, ea_num, emb_channels, aggr, dropout_rate=0):
        super(GraphEmbed, self).__init__(aggr=aggr)

        self.x_num = x_num
        self.ea_num = ea_num
        self.emb_channels = emb_channels
        self.nn = Sequential(Linear(2 * x_num + ea_num, emb_channels), Dropout(p=dropout_rate), ReLU())
        self.aggr = aggr
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        return out

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.nn(z)

    def __repr__(self):
        return '{}(aggr="{}", nn={})'.format(self.__class__.__name__, self.aggr, self.nn)
    
class GNN_ChebConv(nn.Module):
    def __init__(self, hid_channels, edge_features, node_features, edge_channels=32, dropout_rate=0, CC_K=2,
                 emb_aggr='max', depth=2, normalize=True):
        super(GNN_ChebConv, self).__init__()
        self.hid_channels = hid_channels
        self.dropout = dropout_rate
        self.normalize = normalize

        # embedding of node/edge features with NN
        self.embedding = GraphEmbed(node_features, edge_features, edge_channels, aggr=emb_aggr)

        # CB convolutions (with normalization)
        self.convs = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                self.convs.append(ChebConv(edge_channels, hid_channels, CC_K, normalization='sym'))
            else:
                self.convs.append(ChebConv(hid_channels, hid_channels, CC_K, normalization='sym'))

        # output layer (so far only a 1 layer MLP, make more?)
        if depth == 0:
            self.lin = Linear(edge_channels, 1)
        else:
            self.lin = Linear(hid_channels, 1)

    def forward(self, data):

        # retrieve model device (for LayerNorm to work)
        device = next(self.parameters()).device

        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # 1. Pre-process data (nodes and edges) with MLP
        x = self.embedding(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # 2. Do convolutions
        for i in range(len(self.convs)):
            x = self.convs[i](x=x, edge_index=edge_index)
            if self.normalize:
                x = nn.LayerNorm(self.hid_channels, eps=1e-5, device=device)(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = nn.ReLU()(x)

        # 3. Output
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        # print(f"liner_x.shape = {x.shape}")
        # Mask over storage nodes (which have pressure=0)
        x = x.squeeze(1)  # [num_nodes, 1] -> [num_nodes]
        # print(f"output x.shape = {x.shape}")
        return x

class HydraulicGNN(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')  # 聚合函数: 均值/求和/最大值
        # 节点更新MLP
        self.node_mlp = Sequential(
            Linear(7, 64),  # 输入: 自身特征 + 邻居聚合特征
            ReLU(),
            Linear(64, 1)   # 输出重建的节点压力
        )
        # 边更新MLP
        self.edge_mlp = Sequential(
            Linear(5, 64),  # 输入: 源节点、目标节点、原始边特征
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