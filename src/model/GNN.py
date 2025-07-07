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
from torch_geometric.utils import add_self_loops, degree


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
    
class TGCN_PyG(nn.Module):
    def __init__(self, in_feats, gcn_hidden, gru_hidden, out_feats):
        super(TGCN_PyG, self).__init__()
        self.gcn = GCNLayer(in_feats, gcn_hidden)
        self.gru = nn.GRU(input_size=gcn_hidden, hidden_size=gru_hidden, batch_first=True)
        self.out_layer = nn.Linear(gru_hidden, out_feats)

    def forward(self, x_seq, edge_index):
        # x_seq: [T, N, F]
        T, N, F = x_seq.shape
        # print(f"x_seq.shape: {x_seq.shape}")
        outputs = []


        gcn_out_seq = []
        for t in range(T):
            x_t = x_seq[t]  # [N, F]
            gcn_out = self.gcn(x_t, edge_index)  # [N, gcn_hidden]
            gcn_out_seq.append(gcn_out)
        gcn_out_seq = torch.stack(gcn_out_seq, dim=0)  # [T, N, gcn_hidden]
        gcn_out_seq = gcn_out_seq.permute(1, 0, 2)  # [N, T, gcn_hidden]

        node_outputs = []
        for n in range(N):
            node_seq = gcn_out_seq[n]  # [1, T, gcn_hidden]
            _, h = self.gru(node_seq)               # h: [1, 1, gru_hidden]
            node_outputs.append(self.out_layer(h))  # [1, out_feats]

        outputs.append(torch.cat(node_outputs, dim=0))  # [N, out_feats]

        return torch.stack(outputs, dim=0)[0]  # [N, out_feats]
    
class TGCN_MessageCoupling(nn.Module):
    def __init__(self, in_feats, gcn_hidden, gru_hidden, edge_hidden, out_node_feats=1, out_edge_feats=1):
        super().__init__()
        self.gcn = GCNLayer(in_feats, gcn_hidden)
        self.gru = nn.GRU(input_size=gcn_hidden, hidden_size=gru_hidden, batch_first=True)

        # 边流量预测器：输入 2 个节点嵌入 + 边属性 (含 masked_flow)
        self.edge_mlp = nn.Sequential(
            nn.Linear(gru_hidden * 2 + 4, edge_hidden),  # 3个原始边属性 + 1个masked_flow
            nn.ReLU(),
            nn.Linear(edge_hidden, out_edge_feats)
        )

        # 节点压力预测器：节点嵌入 + 汇聚的边流量
        self.pressure_mlp = nn.Sequential(
            nn.Linear(gru_hidden + out_edge_feats, gru_hidden),
            nn.ReLU(),
            nn.Linear(gru_hidden, out_node_feats)
        )

    def forward(self, x_seq, edge_index, edge_attr):
        """
        x_seq: Tensor, shape [T, N, F]       # 节点时序输入
        edge_index: LongTensor [2, E]         # 图结构
        edge_attr: Tensor, shape [E, 4]       # 边属性 + masked_flow
        """
        T, N, F = x_seq.shape
        device = x_seq.device

        # Step 1: 每个时间步做 GCN
        gcn_out_seq = []
        for t in range(T):
            x_t = x_seq[t]  # [N, F]
            gcn_out = self.gcn(x_t, edge_index)  # [N, gcn_hidden]
            gcn_out_seq.append(gcn_out)

        # Step 2: 组装 [N, T, H] 输入 GRU
        gcn_out_seq = torch.stack(gcn_out_seq, dim=0)  # [T, N, H]
        gcn_out_seq = gcn_out_seq.permute(1, 0, 2)     # [N, T, H]

        # Step 3: 每个节点过 GRU，得到最终表示
        node_embed = []
        for n in range(N):
            node_seq = gcn_out_seq[n].unsqueeze(0)  # [1, T, H]
            _, h = self.gru(node_seq)               # h: [1, 1, H]
            h = h.squeeze(0).squeeze(0)             # [H]
            node_embed.append(h)
        node_embed = torch.stack(node_embed, dim=0)  # [N, H]

        # Step 4: 边预测（根据 node_embed + edge_attr）
        src, dst = edge_index[0], edge_index[1]  # [E]
        src_feat = node_embed[src]  # [E, H]
        dst_feat = node_embed[dst]  # [E, H]
        edge_input = torch.cat([src_feat, dst_feat, edge_attr], dim=-1)  # [E, 2H + 4]
        pred_edge = self.edge_mlp(edge_input)  # [E, 1]，预测边流量

        # Step 5: 汇聚边流量到节点（反向 message passing）
        node_flow_agg = torch.zeros((N, pred_edge.shape[-1]), device=device)  # [N, 1]
        node_flow_agg = node_flow_agg.index_add(0, dst, pred_edge)  # sum of incoming flows

        # Step 6: 节点压力预测（节点表示 + 边信息）
        pressure_input = torch.cat([node_embed, node_flow_agg], dim=-1)  # [N, H + 1]
        pred_node = self.pressure_mlp(pressure_input)  # [N, 1]，预测节点压力

        return pred_node, pred_edge
