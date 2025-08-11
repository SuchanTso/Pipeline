from torch_geometric.nn import MessagePassing,ChebConv,GATConv
from torch.nn import Sequential, Linear, ReLU
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.nn.inits import reset
from typing import Union
from torch import Tensor
from torch.nn import Dropout
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
from .layers import *

# 一个能同时更新节点和边的MPNN层
class NodeEdgeUpdateLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr='add')
        # 节点更新网络: f_n(h_v, aggregated_message)
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim) # 输出维度可以和输入一样，或者新的维度
        )
        # 边更新网络: f_e(h_e, h_u, h_v)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim) # 输出维度可以和输入一样
        )
        # 消息网络: f_m(h_v, h_e)
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        # x: [N, node_dim], edge_attr: [E, edge_dim]
        
        # 1. 消息传递和聚合 (用于更新节点)
        aggregated_messages = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # 2. 更新节点表示
        x_new = self.node_mlp(torch.cat([x, aggregated_messages], dim=1))
        
        # 3. 更新边表示
        src, dst = edge_index
        edge_attr_new = self.edge_mlp(torch.cat([edge_attr, x[src], x[dst]], dim=1))
        
        return x_new, edge_attr_new

    def message(self, x_j, edge_attr):
        # x_j 是邻居节点特征，edge_attr是连接的边特征
        return self.message_mlp(torch.cat([x_j, edge_attr], dim=1))


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

    def forward(self, x , edge_attr , edge_index):

        # retrieve model device (for LayerNorm to work)
        device = next(self.parameters()).device

        # x = data.x
        # edge_index = data.edge_index
        # edge_attr = data.edge_attr

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

class GCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__(aggr='add')  # 可选 mean / max / add
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        
        # Step 1: 加入自环
        # print(f"edge_index.shape: {edge_index.shape} , num_nodes: {x.size(0)}")
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Step 2: 线性变换
        x = self.linear(x)
        
        # Step 3: 归一化（D^-0.5 A D^-0.5）
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Step 4: 触发消息传递（调用 message(), aggregate(), update()）
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j 是邻居节点发来的特征
        return norm.view(-1, 1) * x_j
    
class TGCN_MessageCoupling(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, gcn_hidden, node_gru_hidden, edge_gru_hidden, edge_mlp_hidden, out_node_feats=1, out_edge_feats=1):
        """
        Args:
            node_in_feats (int): 输入节点特征的维度 (F_node)
            edge_in_feats (int): 输入边特征的维度 (F_edge)
            gcn_hidden (int): GCN层的隐藏维度
            node_gru_hidden (int): 节点GRU的隐藏维度
            edge_gru_hidden (int): 边GRU的隐藏维度
            edge_mlp_hidden (int): 边预测MLP的隐藏维度
            out_node_feats (int): 输出节点特征的维度（通常是1，代表压力）
            out_edge_feats (int): 输出边特征的维度（通常是1，代表流量）
        """
        super().__init__()
        
        # 1. 空间信息提取 (GCN)
        self.gcn = GCNLayer(node_in_feats, gcn_hidden)
        
        # 2. 时序信息提取 (GRUs)
        # 节点时序GRU
        self.node_gru = nn.GRU(input_size=gcn_hidden, hidden_size=node_gru_hidden, batch_first=True)
        # 边时序GRU (新增)
        self.edge_gru = nn.GRU(input_size=edge_in_feats, hidden_size=edge_gru_hidden, batch_first=True)

        # 3. 预测器 (MLPs)
        # 边流量预测器：输入来自 [源节点嵌入, 目标节点嵌入, 边嵌入]
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_gru_hidden * 2 + edge_gru_hidden, edge_mlp_hidden),
            nn.ReLU(),
            nn.Linear(edge_mlp_hidden, out_edge_feats)
        )

        # 节点压力预测器：输入来自 [节点嵌入, 汇聚的邻边预测流量]
        self.pressure_mlp = nn.Sequential(
            nn.Linear(node_gru_hidden, node_gru_hidden), # 输入维度变化
            nn.ReLU(),
            nn.Linear(node_gru_hidden, out_node_feats)
        )

    def forward(self, x_seq, edge_index, edge_attr_seq):
        """
        x_seq: 节点时序输入, Tensor, shape [T, N, F_node]
        edge_index: 图结构, LongTensor, shape [2, E]
        edge_attr_seq: 边时序输入, Tensor, shape [T, E, F_edge]
        """
        T, N, _ = x_seq.shape
        _, E, _ = edge_attr_seq.shape
        device = x_seq.device

        # --- 节点时空嵌入 ---
        # Step 1: 每个时间步独立通过GCN提取空间特征
        gcn_out_seq = []
        for t in range(T):
            x_t = x_seq[t]  # [N, F_node]
            gcn_out = self.gcn(x_t, edge_index)  # [N, gcn_hidden]
            gcn_out_seq.append(gcn_out)
        
        # Step 2: GCN输出序列通过GRU进行时序建模
        gcn_out_seq = torch.stack(gcn_out_seq, dim=0)  # [T, N, gcn_hidden]
        gcn_out_seq = gcn_out_seq.permute(1, 0, 2)     # [N, T, gcn_hidden] (GRU需要batch_first)
        
        _, h_n_node = self.node_gru(gcn_out_seq) # h_n_node shape: [1, N, node_gru_hidden]
        node_embed = h_n_node.squeeze(0) # [N, node_gru_hidden], 每个节点的最终时空嵌入

        # --- 边时序嵌入 (新增部分) ---
        # Step 3: 边特征序列通过GRU进行时序建模
        edge_attr_seq_permuted = edge_attr_seq.permute(1, 0, 2) # [E, T, F_edge] (GRU需要batch_first)
        _, h_n_edge = self.edge_gru(edge_attr_seq_permuted) # h_n_edge shape: [1, E, edge_gru_hidden]
        edge_embed = h_n_edge.squeeze(0) # [E, edge_gru_hidden], 每条边的最终时序嵌入

        # --- 耦合预测 ---
        # Step 4: 边流量预测
        # 使用节点和边的最终嵌入来预测流量
        src, dst = edge_index[0], edge_index[1]
        src_embed = node_embed[src]  # [E, node_gru_hidden]
        dst_embed = node_embed[dst]  # [E, node_gru_hidden]
        
        # 拼接源节点、目标节点和边自身的嵌入
        edge_prediction_input = torch.cat([src_embed, dst_embed, edge_embed], dim=-1) # [E, node_gru_hidden*2 + edge_gru_hidden]
        pred_edge = self.edge_mlp(edge_prediction_input)  # [E, 1], 预测的边流量

        # Step 5: 汇聚预测的边流量到节点
        # 计算每个节点的净流入流量（可根据物理意义调整为 in-flow - out-flow，这里简单地汇聚入度流量）
        # node_flow_agg = torch.zeros((N, pred_edge.shape[-1]), device=device)  # [N, 1]
        # node_flow_agg = node_flow_agg.index_add(0, dst, pred_edge)  # 将所有入边流量加到目标节点上

        # Step 6: 节点压力预测
        # 使用节点的时空嵌入和汇聚的流量信息来预测压力
        # pressure_prediction_input = torch.cat([node_embed], dim=-1)  # [N, node_gru_hidden + 1]
        pred_node = self.pressure_mlp(node_embed)  # [N, 1], 预测的节点压力

        return pred_node, pred_edge
    
# class TGCN_MessageCoupling(nn.Module):
#     def __init__(self, in_feats, gcn_hidden, gru_hidden, edge_hidden, out_node_feats=1, out_edge_feats=1):
#         super().__init__()
#         # self.gcn = GNN_ChebConv(hid_channels=gcn_hidden, edge_features=4, node_features=in_feats)
#         self.gcn = GCNLayer(in_feats, gcn_hidden)
#         self.gru = nn.GRU(input_size=gcn_hidden, hidden_size=gru_hidden, batch_first=True)

#         # 边流量预测器：输入 2 个节点嵌入 + 边属性 (含 masked_flow)
#         self.edge_mlp = nn.Sequential(
#             nn.Linear(4 + 1, edge_hidden),  # 3个原始边属性 + 1个masked_flow
#             # nn.Linear(gru_hidden * 2 + 4 + 1, edge_hidden),  # 3个原始边属性 + 1个masked_flow
#             nn.ReLU(),
#             nn.Linear(edge_hidden, out_edge_feats)
#         )

#         # 节点压力预测器：节点嵌入 + 汇聚的边流量
#         self.pressure_mlp = nn.Sequential(
#             nn.Linear(gru_hidden, gru_hidden),
#             nn.ReLU(),
#             nn.Linear(gru_hidden, out_node_feats)
#         )

#     def forward(self, x_seq, edge_index, edge_attr):
#         """
#         x_seq: Tensor, shape [T, N, F]       # 节点时序输入
#         edge_index: LongTensor [2, E]         # 图结构
#         edge_attr: Tensor, shape [E, 4]       # 边属性 + masked_flow
#         """
#         T, N, F = x_seq.shape
#         device = x_seq.device

#         # Step 1: 每个时间步做 GCN
#         gcn_out_seq = []
#         for t in range(T):
#             x_t = x_seq[t]  # [N, F]
#             # gcn_out = self.gcn(x_t, edge_index.T.int(), edge_attr.int())
#             gcn_out = self.gcn(x_t, edge_index)  # [N, gcn_hidden]
#             gcn_out_seq.append(gcn_out)

#         # Step 2: 组装 [N, T, H] 输入 GRU
#         gcn_out_seq = torch.stack(gcn_out_seq, dim=0)  # [T, N, H]
#         gcn_out_seq = gcn_out_seq.permute(1, 0, 2)     # [N, T, H]

#         # Step 3: 每个节点过 GRU，得到最终表示
#         _ , h_n = self.gru(gcn_out_seq)
#         node_embed = h_n.squeeze(0)  # [N, H]
#         # node_embed = torch.stack(node_embed, dim=0)  # [N, H]

#         # Step 4: 边预测（根据 node_embed + edge_attr）
#         src, dst = edge_index[0], edge_index[1]  # [E]
#         last_t_pressure = x_seq[-1 , : , -1].unsqueeze(1)  # [N, 1]，最后时刻的压力
#         src_pressure = last_t_pressure[src]  # [E, 1]
#         dst_pressure = last_t_pressure[dst]  # [E, 1]
#         pressure_diff = src_pressure - dst_pressure
#         src_feat = node_embed[src]  # [E, H]
#         dst_feat = node_embed[dst]  # [E, H]
#         edge_input = torch.cat([edge_attr , pressure_diff], dim=-1)  # [E, 2H + 4]
#         # edge_input = torch.cat([src_feat, dst_feat, edge_attr , pressure_diff], dim=-1)  # [E, 2H + 4]
#         pred_edge = self.edge_mlp(edge_input)  # [E, 1]，预测边流量

#         # Step 5: 汇聚边流量到节点（反向 message passing）
#         node_flow_agg = torch.zeros((N, pred_edge.shape[-1]), device=device)  # [N, 1]
#         node_flow_agg = node_flow_agg.index_add(0, dst, pred_edge)  # sum of incoming flows

#         # Step 6: 节点压力预测（节点表示 + 边信息）
#         pressure_input = torch.cat([node_embed, node_flow_agg], dim=-1)  # [N, H + 1]
#         # pred_node = self.pressure_mlp(pressure_input)  # [N, 1]，预测节点压力
#         pred_node = self.pressure_mlp(node_embed)  # [N, 1]，预测节点压力

#         return pred_node, pred_edge

class TGCN_MessageCoupling_Deep(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, gcn_hidden, node_gru_hidden, edge_gru_hidden, 
                 edge_mlp_hidden, out_node_feats=1, out_edge_feats=1, 
                 gcn_layers=2, gru_layers=2, dropout_rate=0.1): # 新增超参数
        """
        Args:
            gcn_layers (int): GCN的层数.
            gru_layers (int): GRU的层数.
            dropout_rate (float): Dropout比率，防止过拟合.
        """
        super().__init__()
        
        # --- 1. 空间信息提取 (Deep GCN with Residuals) ---
        self.gcn_layers = nn.ModuleList()
        # 第一层：从输入特征维度到隐藏维度
        self.gcn_layers.append(GCNLayer(node_in_feats, gcn_hidden))
        # 中间层：从隐藏维度到隐藏维度
        for _ in range(gcn_layers - 1):
            self.gcn_layers.append(GCNLayer(gcn_hidden, gcn_hidden))

        # --- 2. 时序信息提取 (Deep GRU) ---
        # 节点时序GRU
        self.node_gru = nn.GRU(
            input_size=gcn_hidden, 
            hidden_size=node_gru_hidden, 
            num_layers=gru_layers, # 增加GRU层数
            batch_first=True,
            dropout=dropout_rate if gru_layers > 1 else 0 # 只有多层时才使用dropout
        )
        # 边时序GRU
        self.edge_gru = nn.GRU(
            input_size=edge_in_feats, 
            hidden_size=edge_gru_hidden, 
            num_layers=gru_layers, # 增加GRU层数
            batch_first=True,
            dropout=dropout_rate if gru_layers > 1 else 0
        )
        
        # Dropout层，用于GCN之后
        self.dropout = nn.Dropout(p=dropout_rate)

        # --- 3. 预测器 (MLPs) ---
        # 边流量预测器
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_gru_hidden * 2 + edge_gru_hidden + 1, edge_mlp_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(edge_mlp_hidden, out_edge_feats)
        )

        # 节点压力预测器
        self.pressure_mlp = nn.Sequential(
            nn.Linear(node_gru_hidden, node_gru_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(node_gru_hidden, out_node_feats)
        )

    def forward(self, x_seq, edge_index, edge_attr_seq):
        T, N, _ = x_seq.shape
        device = x_seq.device

        # --- 节点时空嵌入 ---
        # Step 1: 每个时间步独立通过 Deep GCN 提取空间特征
        gcn_out_seq = []
        for t in range(T):
            x_t = x_seq[t]
            # 通过多层GCN
            for i, layer in enumerate(self.gcn_layers):
                x_t_out = F.relu(layer(x_t, edge_index))
                # 残差连接 (除了第一层，因为维度可能不匹配)
                if i > 0:
                    x_t = x_t + x_t_out
                else:
                    x_t = x_t_out
                x_t = nn.LayerNorm(x_t.size()[1:], eps=1e-5, device=device)(x_t)  # LayerNorm
                x_t = self.dropout(x_t) # 在每层GCN后应用dropout
            gcn_out_seq.append(x_t)
        
        gcn_out_seq = torch.stack(gcn_out_seq, dim=0)  # [T, N, gcn_hidden]
        gcn_out_seq = gcn_out_seq.permute(1, 0, 2)     # [N, T, gcn_hidden]
        
        # Step 2: GCN输出序列通过 Deep GRU 进行时序建模
        # GRU的h0默认为0，无需手动传入
        _, h_n_node = self.node_gru(gcn_out_seq) # h_n_node shape: [gru_layers, N, node_gru_hidden]
        # 我们只需要最后一层的隐藏状态
        node_embed = h_n_node[-1, :, :] # [N, node_gru_hidden]
        pred_node = self.pressure_mlp(node_embed)

        # --- 边时序嵌入 ---
        # Step 3: 边特征序列通过 Deep GRU 进行时序建模
        edge_attr_seq_permuted = edge_attr_seq.permute(1, 0, 2) # [E, T, F_edge]
        _, h_n_edge = self.edge_gru(edge_attr_seq_permuted) # h_n_edge shape: [gru_layers, E, edge_gru_hidden]
        edge_embed = h_n_edge[-1, :, :] # [E, edge_gru_hidden]

        # --- 耦合预测 (这部分逻辑不变) ---
        # Step 4: 边流量预测
        src, dst = edge_index[0], edge_index[1]
        src_embed = node_embed[src]
        dst_embed = node_embed[dst]
        pressure_src = pred_node[src]
        pressure_dst = pred_node[dst]
        pressure_diff = pressure_src - pressure_dst
        edge_prediction_input = torch.cat([src_embed, dst_embed, edge_embed , pressure_diff], dim=-1)
        pred_edge = self.edge_mlp(edge_prediction_input)

        # Step 5: 汇聚预测的边流量到节点
        # node_flow_agg = torch.zeros((N, pred_edge.shape[-1]), device=device)
        # node_flow_agg = node_flow_agg.index_add(0, dst, pred_edge)

        # Step 6: 节点压力预测

        return pred_node, pred_edge
    
class TGAT_MessageCoupling_Deep(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, gcn_hidden, node_gru_hidden, edge_gru_hidden, 
                 edge_mlp_hidden, out_node_feats=1, out_edge_feats=1, 
                 gat_layers=2, gru_layers=2, dropout_rate=0.1, heads=4): # 修改和新增超参数
        """
        Args:
            gat_layers (int): GAT的层数.
            gru_layers (int): GRU的层数.
            dropout_rate (float): Dropout比率.
            heads (int): GAT多头注意力的头数.
        """
        super().__init__()
        
        # --- 1. 空间信息提取 (Multi-Head GAT) ---
        self.gat_layers = nn.ModuleList()
        
        # GAT的输入和输出维度处理与GCN不同，因为多头注意力会拼接输出
        # 输出维度 = heads * gcn_hidden
        
        # 第一层：从输入特征维度到隐藏维度
        self.gat_layers.append(
            GATConv(node_in_feats, gcn_hidden, heads=heads, dropout=dropout_rate, concat=True)
        )
        # GAT第一层的输出维度是 gcn_hidden * heads

        # 中间层：从 (gcn_hidden * heads) 维度到 (gcn_hidden * heads) 维度
        for _ in range(gat_layers - 2):
            self.gat_layers.append(
                GATConv(gcn_hidden * heads, gcn_hidden, heads=heads, dropout=dropout_rate, concat=True)
            )

        # 最后一层：为了输出维度统一，通常concat=False，使用平均而不是拼接
        # 这样输出维度就是 gcn_hidden，方便后续接入GRU
        self.gat_layers.append(
            GATConv(gcn_hidden * heads, gcn_hidden, heads=heads, dropout=dropout_rate, concat=False)
        )

        # --- 2. 时序信息提取 (Deep GRU) ---
        # 节点时序GRU
        self.node_gru = nn.GRU(
            input_size=gcn_hidden, # GRU的输入维度现在是GAT最后一层的输出维度
            hidden_size=node_gru_hidden, 
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout_rate if gru_layers > 1 else 0
        )
        # 边时序GRU (这部分不变)
        self.edge_gru = nn.GRU(
            input_size=edge_in_feats, 
            hidden_size=edge_gru_hidden, 
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout_rate if gru_layers > 1 else 0
        )
        
        # Dropout层
        self.dropout = nn.Dropout(p=dropout_rate)

        # --- 3. 预测器 (MLPs) ---
        # 边流量预测器 (这部分不变)
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_gru_hidden * 2 + edge_gru_hidden + 1, edge_mlp_hidden), # 输入维度调整
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(edge_mlp_hidden, out_edge_feats)
        )

        # 节点压力预测器 (这部分不变)
        self.pressure_mlp = nn.Sequential(
            nn.Linear(node_gru_hidden, node_gru_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(node_gru_hidden, out_node_feats)
        )

    def forward(self, x_seq, edge_index, edge_attr_seq):
        T, N, _ = x_seq.shape
        device = x_seq.device

        # --- 节点时空嵌入 ---
        # Step 1: 每个时间步独立通过 GAT 提取空间特征
        gat_out_seq = []
        for t in range(T):
            x_t = x_seq[t]
            # 通过多层GAT
            for i, layer in enumerate(self.gat_layers):
                x_t = F.elu(layer(x_t, edge_index)) # GAT通常使用ELU作为激活函数
                # 注意：GATConv内部已经集成了dropout，这里可以不再手动加
                # 残差连接在GAT中也可以使用，但要确保维度匹配
                # 这里为了简化，我们暂时不加残差连接，因为GAT的多头机制本身就很强大

            gat_out_seq.append(x_t)
        
        gat_out_seq = torch.stack(gat_out_seq, dim=0)  # [T, N, gcn_hidden]
        gat_out_seq = gat_out_seq.permute(1, 0, 2)     # [N, T, gcn_hidden]
        
        # Step 2: GAT输出序列通过 GRU 进行时序建模 (这部分逻辑不变)
        _, h_n_node = self.node_gru(gat_out_seq) 
        node_embed = h_n_node[-1, :, :] 
        pred_node = self.pressure_mlp(node_embed)

        # --- 边时序嵌入 --- (这部分逻辑不变)
        edge_attr_seq_permuted = edge_attr_seq.permute(1, 0, 2)
        _, h_n_edge = self.edge_gru(edge_attr_seq_permuted)
        edge_embed = h_n_edge[-1, :, :]

        # --- 耦合预测 ---
        # Step 4: 边流量预测
        src, dst = edge_index[0], edge_index[1]
        src_embed = node_embed[src]
        dst_embed = node_embed[dst]
        pressure_diff = pred_node[src] - pred_node[dst]

        # ！！！重要修改！！！
        # 在你之前的代码中，你将预测出的pressure_diff作为输入
        # 这会导致一个循环依赖：为了预测流量，需要先预测压力，但压力的准确性又依赖于整个模型的输出
        # 一个更稳健、更常见的数据驱动做法是，直接使用从GRU出来的节点嵌入(embedding)来预测流量
        # 这些嵌入已经编码了时空信息，包括了与压力相关的信息
        # 让我们先去掉显式的pressure_diff，让模型从嵌入中学习关系
        
        # 原来的输入: torch.cat([src_embed, dst_embed, edge_embed, pressure_diff], dim=-1)
        # 修改后的输入:
        edge_prediction_input = torch.cat([src_embed, dst_embed, edge_embed , pressure_diff], dim=-1)
        pred_edge = self.edge_mlp(edge_prediction_input)
        
        # Step 5 & 6: 返回预测值
        return pred_node, pred_edge
    
    
class TGCN_PrimalDual(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, gcn_hidden, node_gru_hidden, edge_gru_hidden, 
                 edge_gcn_hidden, edge_mlp_hidden, out_node_feats=1, out_edge_feats=1, 
                 gcn_layers=2, gru_layers=2, dropout_rate=0.3, tanh_scale=3.0):
        """
        一个同时在原始图(Primal Graph)和对偶图(Dual Graph)上传播信息的时空图网络。

        Args:
            node_in_feats (int): 节点输入特征维度
            edge_in_feats (int): 边输入特征维度
            gcn_hidden (int): 原始图GCN的隐藏维度
            node_gru_hidden (int): 节点GRU的隐藏维度
            edge_gru_hidden (int): 边GRU的隐藏维度
            edge_gcn_hidden (int): 对偶图GCN的隐藏维度 (新增)
            edge_mlp_hidden (int): 边预测MLP的隐藏维度
            out_node_feats (int): 节点输出维度 (压力)
            out_edge_feats (int): 边输出维度 (流量)
            gcn_layers (int): GCN层数
            gru_layers (int): GRU层数
            dropout_rate (float): Dropout比率
            tanh_scale (float): TanhScaler的缩放因子
        """
        super().__init__()

        # --- 1. 原始图路径 (Primal Path for Nodes) ---
        self.primal_gcn_layers = nn.ModuleList()
        self.primal_gcn_layers.append(GCNLayer(node_in_feats, gcn_hidden))
        for _ in range(gcn_layers - 1):
            self.primal_gcn_layers.append(GCNLayer(gcn_hidden, gcn_hidden))

        self.node_gru = nn.GRU(
            input_size=gcn_hidden, 
            hidden_size=node_gru_hidden, 
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout_rate if gru_layers > 1 else 0
        )

        # --- 2. 对偶图路径 (Dual Path for Edges) ---
        # 边的时序GRU，为对偶GCN提供输入
        self.edge_gru = nn.GRU(
            input_size=edge_in_feats, 
            hidden_size=edge_gru_hidden, 
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout_rate if gru_layers > 1 else 0
        )
        
        # 对偶图GCN，用于边-边信息传播
        self.dual_gcn_layers = nn.ModuleList()
        self.dual_gcn_layers.append(GCNLayer(edge_gru_hidden, edge_gcn_hidden))
        for _ in range(gcn_layers - 1):
            self.dual_gcn_layers.append(GCNLayer(edge_gcn_hidden, edge_gcn_hidden))

        self.dropout = nn.Dropout(p=dropout_rate)

        # --- 3. 预测器 (Predictors) ---
        # 节点压力预测器
        self.pressure_mlp = nn.Sequential(
            nn.Linear(node_gru_hidden, node_gru_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(node_gru_hidden, out_node_feats)
        )

        # 边流量预测器
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_gru_hidden * 2 + edge_gcn_hidden + 1, edge_mlp_hidden), # 输入维度更新
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(edge_mlp_hidden, out_edge_feats)
            # TanhScaler(scale=tanh_scale) # 保证流量预测也稳定
        )

    def forward(self, x_seq, edge_index, edge_attr_seq):
        """
        Args:
            x_seq (Tensor): 节点特征序列 [T, N, F_node]
            edge_index (LongTensor): 原始图的边索引 [2, E]
            edge_attr_seq (Tensor): 边特征序列 [T, E, F_edge]
            dual_edge_index (LongTensor): 对偶图的边索引 [2, E_dual] (预先计算好)
        """
        T, N, _ = x_seq.shape
        device = x_seq.device
        src, dst = edge_index[0], edge_index[1]
        # print(f"edge_index: {edge_index.shape}")
        dual_edge_index = self.preprocess_data_with_dual_graph(edge_index , N)  # 计算对偶图的边索引
        # --- 节点路径处理 ---
        gcn_out_seq = []
        for t in range(T):
            x_t = x_seq[t]
            for i, layer in enumerate(self.primal_gcn_layers):
                x_t_out = F.relu(layer(x_t, edge_index))
                if i > 0: x_t = x_t + x_t_out  # 残差连接
                else: x_t = x_t_out
                x_t = self.dropout(x_t)
            gcn_out_seq.append(x_t)
        
        gcn_out_seq = torch.stack(gcn_out_seq, dim=0).permute(1, 0, 2)
        _, h_n_node = self.node_gru(gcn_out_seq)
        node_embed = h_n_node[-1, :, :]

        # --- 边路径处理 (核心改进) ---
        # Step 2a: 边自身时序信息提取 (GRU)
        edge_attr_seq_permuted = edge_attr_seq.permute(1, 0, 2)
        _, h_n_edge = self.edge_gru(edge_attr_seq_permuted)
        edge_temporal_embed = h_n_edge[-1, :, :]
        
        # Step 2b: 边-边空间信息传播 (Dual GCN)
        edge_spatial_embed = edge_temporal_embed
        for i, layer in enumerate(self.dual_gcn_layers):
            # 将边的时序嵌入作为对偶图的节点特征进行图卷积
            edge_spatial_out = F.relu(layer(edge_spatial_embed, dual_edge_index))
            if i > 0: edge_spatial_embed = edge_spatial_embed + edge_spatial_out # 残差连接
            else: edge_spatial_embed = edge_spatial_out
            edge_spatial_embed = self.dropout(edge_spatial_embed)

        # --- 耦合预测 ---
        # Step 3a: 优先预测节点压力
        pred_node = self.pressure_mlp(node_embed)

        # Step 3b: 利用预测的压力差和强化后的边嵌入来预测流量
        predicted_pressure_diff = pred_node[src] - pred_node[dst]
        # (可选) 对压力差进行归一化，以平衡特征尺度
        predicted_pressure_diff = F.layer_norm(predicted_pressure_diff, predicted_pressure_diff.shape)
        
        edge_prediction_input = torch.cat([
            node_embed[src], 
            node_embed[dst], 
            edge_spatial_embed,  # 使用经过对偶GCN强化的边嵌入
            predicted_pressure_diff
        ], dim=-1)
        
        pred_edge = self.edge_mlp(edge_prediction_input)

        return pred_node, pred_edge

# ----------------------------------------------------
# 数据预处理：如何生成 dual_edge_index
# ----------------------------------------------------

    def preprocess_data_with_dual_graph(self , edge_index , node_nums):
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
        temp_data = Data(edge_index=edge_index ,num_nodes=node_nums)
        
        # 初始化LineGraph变换器
        line_graph_transform = LineGraph(force_directed=False) # force_directed=False适用于无向图
        
        # 应用变换
        line_graph_data = line_graph_transform(temp_data)
        
        return line_graph_data.edge_index
    
    


# ----------------------------------------------------
# Layer 2: 辅助模块 (用于稳定性和输出)
# ----------------------------------------------------

# ----------------------------------------------------
# Layer 3: 主模型 GAT_EdgeModel
# ----------------------------------------------------
class GAT_RegressionModel(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, node_gru_hidden, edge_gru_hidden, 
                 gat_config, edge_mlp_hidden, dropout=0, tanh_scale=3.0):
        super(GAT_RegressionModel, self).__init__()

        # 时序编码器
        self.node_gru = nn.GRU(input_size=node_in_feats, hidden_size=node_gru_hidden, batch_first=False)
        self.edge_gru = nn.GRU(input_size=edge_in_feats, hidden_size=edge_gru_hidden, batch_first=False)
        
        # 空间编码器 (使用新的MultiLayerGATEncoder)
        self.gat_encoder = MultiLayerGATEncoder(
            n_layers=gat_config['n_layers'],
            n_heads=gat_config['n_heads'],
            node_in_feats=node_gru_hidden,
            node_hid_feats=gat_config['node_hid_feats'],
            node_out_feats=gat_config['node_out_feats'],
            edge_in_feats=edge_gru_hidden,
            edge_hid_feats=gat_config['edge_hid_feats'],
            dropout=dropout,
            alpha=gat_config['alpha']
        )
        
        gat_output_dim = gat_config['node_out_feats']

        # 预测器
        self.pressure_mlp = nn.Sequential(
            nn.Linear(gat_output_dim, gat_output_dim),
            nn.ReLU(),
            nn.Linear(gat_output_dim, 1)
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(gat_output_dim * 2 + edge_in_feats + 1, 1)
        )

    def _pyg_to_adj_format(self, edge_index, edge_attr, num_nodes):
        # ... (代码同上) ...
        adj_mask = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        edge_features_tensor = torch.zeros(num_nodes, num_nodes, edge_attr.shape[1], device=edge_attr.device)
        src, dst = edge_index[0], edge_index[1]
        adj_mask[src, dst] = 1
        edge_features_tensor[src, dst] = edge_attr
        return adj_mask, edge_features_tensor

    def forward(self, x_seq, edge_index, edge_attr_seq):
        T, N, _ = x_seq.shape
        device = x_seq.device
        src, dst = edge_index[0], edge_index[1]

        # 时序编码
        _, h_node = self.node_gru(x_seq)
        node_temporal_embed = h_node.squeeze(0)
        
        _, h_edge = self.edge_gru(edge_attr_seq)
        edge_temporal_embed = h_edge.squeeze(0)
        # print(f"edge_attr_seq: {edge_attr_seq.shape}")
        
        # 空间编码
        adj_mask, edge_features_tensor = self._pyg_to_adj_format(edge_index, edge_temporal_embed, N)
        final_node_embed = self.gat_encoder(node_temporal_embed, adj_mask, edge_features_tensor)

        # 回归预测
        pred_node = self.pressure_mlp(final_node_embed)
        
        pred_pressure_diff = pred_node[src] - pred_node[dst]
        # pred_pressure_diff = y_node[src] - y_node[dst]
        
        edge_prediction_input = torch.cat([
            final_node_embed[src],
            final_node_embed[dst],
            edge_attr_seq,
            pred_pressure_diff
        ], dim=-1)
        
        pred_edge = self.edge_mlp(edge_prediction_input)

        return pred_node, pred_edge