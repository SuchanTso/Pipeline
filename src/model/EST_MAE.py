import torch
import torch.nn as nn
import torch.nn.functional as F

class EGT_Layer_v3(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, max_spd=10, max_degree=20):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads, self.d_head = num_heads, d_model // num_heads
        self.degree_encoder = nn.Embedding(max_degree + 1, d_model, padding_idx=0)
        self.spd_encoder = nn.Embedding(max_spd + 2, num_heads)
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.edge_bias_proj, self.node_out_proj = nn.Linear(d_model, num_heads), nn.Linear(d_model, d_model)
        self.ln_node_1, self.ln_node_2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn_node = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))
        self.edge_enhancer = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.ln_edge, self.dropout = nn.LayerNorm(d_model), nn.Dropout(dropout)
    def forward(self, h_node, h_edge, edge_index, degree_enc=None, spd_matrix=None):
        N, device = h_node.shape[0], h_node.device
        x = h_node
        if degree_enc is not None:
            degree_indices = degree_enc.squeeze(-1).long()
            if degree_indices.max().item() >= self.degree_encoder.num_embeddings: raise IndexError(f"Degree index out of bounds")
            x = x + self.degree_encoder(degree_indices)
        h_node_res = x
        q, k, v = self.q_proj(x).view(N, self.num_heads, self.d_head).transpose(0, 1), self.k_proj(x).view(N, self.num_heads, self.d_head).transpose(0, 1), self.v_proj(x).view(N, self.num_heads, self.d_head).transpose(0, 1)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (self.d_head ** 0.5)
        if spd_matrix is not None:
            if spd_matrix.max().item() >= self.spd_encoder.num_embeddings: raise IndexError(f"SPD index out of bounds")
            attn_scores = attn_scores + self.spd_encoder(spd_matrix).permute(2, 0, 1)
        if h_edge is not None and h_edge.numel() > 0 and h_edge.shape[0] == edge_index.shape[1]:
            edge_bias = torch.zeros(self.num_heads, N, N, device=device); edge_bias[:, edge_index[0], edge_index[1]] = self.edge_bias_proj(h_edge).permute(1, 0); attn_scores = attn_scores + edge_bias
        attn_probs = self.dropout(F.softmax(attn_scores, dim=-1))
        out_node = torch.bmm(attn_probs, v).transpose(0, 1).contiguous().view(N, -1)
        h_node_updated = self.ln_node_1(h_node_res + self.dropout(self.node_out_proj(out_node)))
        h_node_updated = self.ln_node_2(h_node_updated + self.dropout(self.ffn_node(h_node_updated)))
        if h_edge is not None and h_edge.numel() > 0:
            h_edge_res = h_edge; src, dst = edge_index
            enhancer_input = torch.cat([h_node_updated[src], h_node_updated[dst], h_edge_res], dim=-1)
            h_edge_updated = self.ln_edge(h_edge_res + self.dropout(self.edge_enhancer(enhancer_input)))
        else: h_edge_updated = h_edge
        return h_node_updated, h_edge_updated

class TemporalEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers=1, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model * 2, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x_seq):
        """
        Args:
            x_seq (Tensor): [num_entities, window_size, d_model]
        Returns:
            Tensor: [num_entities, d_model] <--- [CRITICAL FIX] 输出维度改变
        """
        temporal_encodings = self.transformer_encoder(x_seq)
        
        # [CRITICAL FIX] 只返回最后一个时间步的输出作为摘要
        # 这将输入的三维张量 "降维" 成了二维张量
        return temporal_encodings[:, -1, :]

# =========================================================
# 2. 核心组件: LiftingLayer (现在逻辑正确)
# =========================================================
class LiftingLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        # 动态特征编码器 (使用修正后的 TemporalEncoder)
        self.temporal_encoder = TemporalEncoder(d_model, num_heads, num_layers=1, dropout=dropout)
        
        self.static_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU()
        )
        
        self.fusion_projector = nn.Linear(d_model * 2, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_node_static, h_node_dynamic_seq):
        # 1. 并行编码
        # dynamic_summary 现在是 [num_nodes, d_model] (二维)
        dynamic_summary = self.temporal_encoder(h_node_dynamic_seq)
        # static_summary 现在是 [num_nodes, d_model] (二维)
        static_summary = self.static_encoder(h_node_static)
        
        # 2. 后期融合
        # [FIXED] 两个二维张量现在可以安全地拼接了
        fused_embedding = torch.cat([static_summary, dynamic_summary], dim=-1)
        lifted_representation = self.fusion_projector(fused_embedding)
        
        return self.fusion_norm(h_node_static + self.dropout(lifted_representation))

class LiftingLayer_v2(nn.Module):
    def __init__(self, d_model_static, d_model_dynamic, d_model_out, num_heads, dropout=0.1):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(d_model_dynamic, num_heads, dropout=dropout)
        self.static_encoder = nn.Sequential(
            nn.LayerNorm(d_model_static),
            nn.Linear(d_model_static, d_model_static),
            nn.GELU()
        )
        self.fusion_projector = nn.Linear(d_model_static + d_model_dynamic, d_model_out)
        self.fusion_norm = nn.LayerNorm(d_model_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_node_static, h_node_dynamic_seq):
        dynamic_summary = self.temporal_encoder(h_node_dynamic_seq)
        static_summary = self.static_encoder(h_node_static)
        fused = torch.cat([static_summary, dynamic_summary], dim=-1)
        lifted = self.fusion_projector(fused)
        return self.fusion_norm(lifted)

class EST_MAE(nn.Module):
    # __init__ 方法基本不变，但我们确保参数清晰
    def __init__(self,
                 static_node_feats, dynamic_node_feats, static_edge_feats,
                 dynamic_edge_feats,
                 d_model, num_heads, 
                 num_decoder_layers,
                 dropout=0.1, max_spd=10, max_degree=20):
        super().__init__()
        
        self.static_node_proj = nn.Linear(static_node_feats, d_model)
        self.dynamic_node_proj = nn.Linear(dynamic_node_feats, d_model)
        self.static_edge_proj = nn.Linear(static_edge_feats, d_model)
        
        self.node_mask_token = nn.Parameter(torch.randn(1, d_model))
        
        self.encoder = LiftingLayer(d_model, num_heads, dropout)
        
        self.decoder = nn.ModuleList(
            [EGT_Layer_v3(d_model, num_heads, dropout, max_spd, max_degree) for _ in range(num_decoder_layers)]
        )

        self.node_reconstruction_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, dynamic_node_feats))
        self.edge_reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model * 2 + static_edge_feats),
            nn.Linear(d_model * 2 + static_edge_feats, d_model), nn.GELU(),
            nn.Linear(d_model, dynamic_edge_feats)
        )

    def forward(self, batch, node_mask_indices, edge_mask_indices):
        # 1. 解包数据
        x_static, x_dynamic_window = batch.x_static, batch.x_dynamic_window
        edge_attr_static = batch.edge_attr_static
        full_edge_index, degree_enc, spd_matrix = batch.edge_index, batch.degree_encoding, batch.spd_matrix
        device = x_static.device
        num_nodes = batch.num_nodes
        
        # --- [CRITICAL FIX] 统一数据维度约定 ---
        # 我们的约定: 所有时序张量的形状都为 [实体数, 时间窗口, 特征数]
        # x_dynamic_window 的输入形状应为 [N_batch, T, D_dyn]
        # (这要求Dataset和DataLoader已经正确处理)
        
        # 2. 特征投影
        h_node_static = self.static_node_proj(x_static)
        # 现在 h_node_dynamic_seq 的形状是 [N_batch, T, d_model]
        h_node_dynamic_seq = self.dynamic_node_proj(x_dynamic_window)
        h_edge_static = self.static_edge_proj(edge_attr_static)

        # 3. 准备编码器输入 (只在可见节点上)
        node_visible_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        if node_mask_indices.numel() > 0:
            node_visible_mask[node_mask_indices] = False
        visible_node_indices = torch.where(node_visible_mask)[0]

        H_encoded_visible = None
        if visible_node_indices.numel() > 0:
            h_node_static_visible = h_node_static[visible_node_indices]
            
            # --- [CRITICAL FIX] 正确的索引操作 ---
            # h_node_dynamic_seq 的第0维是节点维，所以可以直接索引
            h_node_dynamic_seq_visible = h_node_dynamic_seq[visible_node_indices, :, :]
            
            # 4. 通过编码器
            H_encoded_visible = self.encoder(h_node_static_visible, h_node_dynamic_seq_visible)

        # 5. 准备解码器输入 (与v3.2版本相同，因为那部分逻辑是正确的)
        # (a) 构建目标帧 t 的基础嵌入
        # 从时序窗口中取出最后一帧的动态特征
        h_node_dynamic_t = self.dynamic_node_proj(x_dynamic_window[:, -1, :])
        dec_input_h_node_base = h_node_static + h_node_dynamic_t
        
        # (b) 创建带掩码的画布
        dec_input_h_node = dec_input_h_node_base.clone()
        if node_mask_indices.numel() > 0:
            dec_input_h_node[node_mask_indices] = self.node_mask_token
        
        # (c) 粘贴编码器输出
        if H_encoded_visible is not None:
            dec_input_h_node[visible_node_indices] = H_encoded_visible

        # 6. 通过解码器
        dec_h_node = dec_input_h_node
        dec_h_edge = h_edge_static # 静态边特征
        
        for layer in self.decoder:
            dec_h_node, dec_h_edge = layer(dec_h_node, dec_h_edge, full_edge_index, degree_enc, spd_matrix)

        # 7. 重建
        pred_pressures = self.node_reconstruction_head(dec_h_node)
        
        src_full, dst_full = full_edge_index
        edge_recon_input = torch.cat([dec_h_node[src_full], dec_h_node[dst_full], edge_attr_static], dim=-1)
        pred_flows = self.edge_reconstruction_head(edge_recon_input)
        
        return pred_pressures, pred_flows
    
class EST_MAE_v5(nn.Module):
    def __init__(self,
                 static_node_feats, dynamic_node_feats, static_edge_feats, dynamic_edge_feats,
                 d_model, num_heads, 
                 num_decoder_layers,
                 dropout=0.1, max_spd=10, max_degree=20,
                 d_model_static_ratio=0.5):
        super().__init__()
        
        d_model_static = int(d_model * d_model_static_ratio)
        d_model_dynamic = d_model - d_model_static

        self.static_node_proj = nn.Linear(static_node_feats, d_model_static)
        self.dynamic_node_proj = nn.Linear(dynamic_node_feats, d_model_dynamic)
        self.static_edge_proj = nn.Linear(static_edge_feats, d_model)
        
        self.node_mask_token = nn.Parameter(torch.randn(1, d_model))
        
        self.encoder = LiftingLayer_v2(d_model_static, d_model_dynamic, d_model, num_heads, dropout)
        
        self.decoder = nn.ModuleList(
            [EGT_Layer_v3(d_model, num_heads, dropout, max_spd, max_degree) for _ in range(num_decoder_layers)]
        )

        self.node_reconstruction_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, dynamic_node_feats))
        
        # [OPTIMIZATION 2] 边重建头现在接收原始静态边特征
        self.edge_reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model * 2 + static_edge_feats),
            nn.Linear(d_model * 2 + static_edge_feats, d_model), nn.GELU(),
            nn.Linear(d_model, dynamic_edge_feats)
        )

    def forward(self, batch, node_mask_indices, edge_mask_indices):
        x_static, x_dynamic_window = batch.x_static, batch.x_dynamic_window
        edge_attr_static = batch.edge_attr_static
        full_edge_index, degree_enc, spd_matrix = batch.edge_index, batch.degree_encoding, batch.spd_matrix
        device = x_static.device
        num_nodes = batch.num_nodes
        
        # 1. 投影
        h_node_static = self.static_node_proj(x_static)
        h_node_dynamic_seq = self.dynamic_node_proj(x_dynamic_window) # Shape: [N, T, D_dyn]
        
        # 2. 准备编码器输入
        node_visible_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        if node_mask_indices.numel() > 0:
            node_visible_mask[node_mask_indices] = False
        visible_node_indices = torch.where(node_visible_mask)[0]

        H_encoded_visible = None
        if visible_node_indices.numel() > 0:
            h_node_static_visible = h_node_static[visible_node_indices]
            h_node_dynamic_seq_visible = h_node_dynamic_seq[visible_node_indices, :, :]
            H_encoded_visible = self.encoder(h_node_static_visible, h_node_dynamic_seq_visible)

        # 3. 准备解码器输入 (标准MAE流程)
        #    创建一个全尺寸的画布，可见部分用编码器输出，掩码部分用mask_token
        dec_input_h_node = self.node_mask_token.expand(num_nodes, -1).clone()
        if H_encoded_visible is not None:
            dec_input_h_node[visible_node_indices] = H_encoded_visible
            
        # 4. 通过解码器
        #    解码器只在单帧上操作，它的边信息应该是目标帧的静态边信息
        dec_h_edge = self.static_edge_proj(edge_attr_static)
        dec_h_node = dec_input_h_node
        
        for layer in self.decoder:
            dec_h_node, dec_h_edge = layer(dec_h_node, dec_h_edge, full_edge_index, degree_enc, spd_matrix)

        # 5. 重建
        pred_pressures = self.node_reconstruction_head(dec_h_node)
        
        src_full, dst_full = full_edge_index
        # 使用原始的、未投影的静态边特征，信息更直接
        edge_recon_input = torch.cat([dec_h_node[src_full], dec_h_node[dst_full], edge_attr_static], dim=-1)
        pred_flows = self.edge_reconstruction_head(edge_recon_input)
        
        return pred_pressures, pred_flows