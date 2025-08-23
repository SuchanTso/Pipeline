import torch
import torch.nn as nn
from dataset import *

# file: layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphTransformerLayer(nn.Module):
    # ... (__init__ 不变) ...
    def __init__(self, d_model, num_heads, dropout=0.1, max_spd=10):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.degree_encoder = nn.Embedding(20, d_model, padding_idx=0)
        self.spd_encoder = nn.Embedding(max_spd + 2, num_heads)
        self.edge_encoder = nn.Linear(d_model, num_heads)

        self.out_proj = nn.Linear(d_model, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_proj, edge_attr_proj, degree_enc, spd_matrix, edge_index):
        N, _ = x_proj.shape
        device = x_proj.device
        # print("done projecting node features.")
        x = x_proj
        if degree_enc is not None:
        #     print(f"degree_enc: {degree_enc.cpu()}")
            degree_indices = degree_enc.squeeze(-1).long()
            x = x + self.degree_encoder(degree_indices.squeeze(-1))
        # print("done degree encoding.")
        q = self.q_proj(x).view(N, self.num_heads, self.d_head)
        k = self.k_proj(x).view(N, self.num_heads, self.d_head)
        v = self.v_proj(x).view(N, self.num_heads, self.d_head)

        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        content_attn = torch.bmm(q, k.transpose(1, 2)) / (self.d_head ** 0.5)
        attn_scores = content_attn

        if spd_matrix is not None:
            structural_bias = self.spd_encoder(spd_matrix).permute(2, 0, 1)
            attn_scores = attn_scores + structural_bias

        # --- 核心修复：增加一个绝对安全的卫语句 ---
        if edge_attr_proj is not None and edge_index is not None and edge_index.numel() > 0:
            # 检查 edge_attr_proj 和 edge_index 的边数是否匹配
            if edge_attr_proj.shape[0] != edge_index.shape[1]:
                # 如果不匹配，打印警告并跳过边偏置计算。这能防止崩溃。
                print(f"Warning: Mismatch in edge count between edge_attr_proj ({edge_attr_proj.shape[0]}) and edge_index ({edge_index.shape[1]}). Skipping edge bias.")
            else:
                edge_bias_proj = self.edge_encoder(edge_attr_proj).permute(1, 0)
                edge_bias = torch.zeros(self.num_heads, N, N, device=device)
                edge_bias[:, edge_index[0], edge_index[1]] = edge_bias_proj
                attn_scores = attn_scores + edge_bias
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.bmm(attn_probs, v)
        out = out.transpose(0, 1).contiguous().view(N, -1)
        
        x = self.layer_norm1(x + self.dropout(self.out_proj(out)))
        x = self.layer_norm2(x + self.dropout(self.ffn(x)))
        
        return x

class W_GraphMAE(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, d_model, num_heads,
                 num_encoder_layers, num_decoder_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 1. 输入投影层
        self.node_in_proj = nn.Linear(node_in_feats, d_model)
        self.edge_in_proj = nn.Linear(edge_in_feats, d_model)
        
        # 2. 编码器
        self.encoder = nn.ModuleList(
            [GraphTransformerLayer(d_model, num_heads, dropout) for _ in range(num_encoder_layers)]
        )
        
        # 3. 解码器
        self.decoder = nn.ModuleList(
            [GraphTransformerLayer(d_model, num_heads, dropout) for _ in range(num_decoder_layers)]
        )
        
        # 4. 掩码标记
        self.node_mask_token = nn.Parameter(torch.randn(1, d_model))
        
        # 5. 重建头
        self.node_reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )
        self.edge_reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        
        # self.grad_reconstruction_head = nn.Linear(d_model * 2, 1)

    def forward(self, batch, node_mask_indices, edge_mask_indices):
        x, full_edge_index, full_edge_attr = batch.x, batch.edge_index, batch.edge_attr
        device = x.device
        
        # 1. 投影到 d_model 空间
        x_proj = self.node_in_proj(x)
        edge_attr_proj = self.edge_in_proj(full_edge_attr)

        # --- 2. 准备编码器的输入 (核心逻辑修改) ---
        # (a) 确定可见的节点和边
        num_nodes = batch.num_nodes
        num_edges = batch.num_edges
        
        node_visible_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        if node_mask_indices.numel() > 0:
            node_visible_mask[node_mask_indices] = False
        visible_node_indices = torch.where(node_visible_mask)[0]
        
        edge_visible_mask = torch.ones(num_edges, dtype=torch.bool, device=device)
        if edge_mask_indices.numel() > 0:
            edge_visible_mask[edge_mask_indices] = False
        
        # (b) 构建编码器子图：只包含可见节点，以及连接这些可见节点的可见边
        src, dst = full_edge_index
        # 编码器能看到的边，必须满足两个条件：
        # 1. 这条边本身是可见的 (未被掩码)
        # 2. 这条边的两个端点都必须是可见的
        encoder_edge_mask = edge_visible_mask & node_visible_mask[src] & node_visible_mask[dst]
        
        # (c) 提取子图拓扑和特征
        encoded_visible_nodes = torch.empty((0, self.d_model), device=device)
        if visible_node_indices.numel() > 0:
            # 创建节点映射，用于重新标记(relabel)
            node_map = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
            node_map[visible_node_indices] = torch.arange(visible_node_indices.numel(), device=device)
            
            subgraph_edge_index = node_map[full_edge_index[:, encoder_edge_mask]]
            
            visible_nodes_features_proj = x_proj[visible_node_indices]
            visible_edge_features_proj = edge_attr_proj[encoder_edge_mask]
            
            # (d) 通过编码器
            encoded_visible_nodes = visible_nodes_features_proj
            for layer in self.encoder:
                encoded_visible_nodes = layer(
                    x_proj=encoded_visible_nodes, 
                    edge_attr_proj=visible_edge_features_proj,
                    edge_index=subgraph_edge_index,
                    degree_enc=None, 
                    spd_matrix=None
                )
        
        # --- 3. 解码阶段 (与之前类似，但现在更清晰) ---
        # 准备解码器输入：可见节点用编码后的表示，掩码节点用node_mask_token
        decoder_input_x = x_proj.clone()
        if node_mask_indices.numel() > 0:
            decoder_input_x[node_mask_indices] = self.node_mask_token
        
        if visible_node_indices.numel() > 0:
            decoder_input_x[visible_node_indices] = encoded_visible_nodes

        # 通过解码器，解码器可以看到完整的图拓扑，但边特征也被掩码了
        # 我们创建一个带掩码的边特征给解码器
        decoder_edge_attr_proj = edge_attr_proj.clone()
        if edge_mask_indices.numel() > 0:
             # 为了简化，我们直接置零，也可以用可学习的edge_mask_token
             decoder_edge_attr_proj[edge_mask_indices] = 0.0

        decoded_x = decoder_input_x
        for layer in self.decoder:
            decoded_x = layer(
                x_proj=decoded_x, 
                edge_attr_proj=decoder_edge_attr_proj,
                edge_index=full_edge_index,
                degree_enc=batch.degree_encoding, # 解码器可以使用结构信息
                spd_matrix=batch.spd_matrix
            )
            
        # --- 4. 重建 ---
        reconstructed_pressures = self.node_reconstruction_head(decoded_x)
        
        src, dst = full_edge_index[0], full_edge_index[1]
        decoded_src_nodes = decoded_x[src]
        decoded_dst_nodes = decoded_x[dst]
        edge_reconstruction_input = torch.cat([decoded_src_nodes, decoded_dst_nodes], dim=-1)
        reconstructed_flows = self.edge_reconstruction_head(edge_reconstruction_input)
        
        return reconstructed_pressures, reconstructed_flows