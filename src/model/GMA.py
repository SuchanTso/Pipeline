import torch
import torch.nn as nn
from .layers import *

class GraphMaskedAutoencoder(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, d_model, num_heads, num_encoder_layers,
                 num_decoder_layers, dropout=0.1):
        super().__init__()
        
        self.node_in_proj = nn.Linear(node_in_feats, d_model)
        self.edge_in_proj = nn.Linear(edge_in_feats, d_model)
        
        self.node_mask_token = nn.Parameter(torch.randn(1, d_model))
        self.edge_mask_token = nn.Parameter(torch.randn(1, d_model))

        # --- 核心修改 ---
        # GraphTransformerLayer的初始化不再需要edge_in_feats
        self.encoder = nn.ModuleList(
            [GraphTransformerLayer(d_model, num_heads, dropout) for _ in range(num_encoder_layers)]
        )
        self.decoder_transformer = nn.ModuleList(
            [GraphTransformerLayer(d_model, num_heads, dropout) for _ in range(num_decoder_layers)]
        )
        
        self.node_reconstruction_head = nn.Linear(d_model, 1)
        self.edge_reconstruction_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, batch, node_mask_indices, edge_mask_indices):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        degree_enc, spd_matrix, edge_map = batch.degree_encoding, batch.spd_matrix, batch.edge_map
        
        # --- 编码过程 ---
        # 1. 输入投影 (在Layer外部完成)
        x_proj = self.node_in_proj(x)
        edge_attr_proj = self.edge_in_proj(edge_attr)
        
        # 2. 应用掩码
        x_masked = x_proj.clone()
        if node_mask_indices is not None and len(node_mask_indices) > 0:
            x_masked[node_mask_indices, :] = self.node_mask_token
            
        edge_attr_masked_proj = edge_attr_proj.clone()
        if edge_mask_indices is not None and len(edge_mask_indices) > 0:
            edge_attr_masked_proj[edge_mask_indices, :] = self.edge_mask_token
        
        # 3. 通过编码器
        #    我们将掩码后的节点特征和掩码后的边特征都传入
        #    Layer内部的注意力偏置会使用被掩码的边特征
        encoded_x = x_masked
        temp_edge_attr_proj = edge_attr_masked_proj
        for layer in self.encoder:
            encoded_x = layer(encoded_x, temp_edge_attr_proj, degree_enc, spd_matrix, edge_map, edge_index)

        # --- 解码过程 ---
        # 解码器接收编码器的输出，并尝试重建
        decoded_x = encoded_x
        for layer in self.decoder_transformer:
            decoded_x = layer(decoded_x, temp_edge_attr_proj, degree_enc, spd_matrix, edge_map, edge_index)
        
        # --- 重建 ---
        reconstructed_pressures = self.node_reconstruction_head(decoded_x)
        
        src, dst = edge_index[0], edge_index[1]
        decoded_src_nodes = decoded_x[src]
        decoded_dst_nodes = decoded_x[dst]
        edge_reconstruction_input = torch.cat([decoded_src_nodes, decoded_dst_nodes], dim=-1)
        reconstructed_flows = self.edge_reconstruction_head(edge_reconstruction_input)
        
        return reconstructed_pressures, reconstructed_flows
    
    
