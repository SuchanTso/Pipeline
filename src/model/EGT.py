import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing,ChebConv,GATConv

class EGT_Layer(nn.Module):
    """
    Enhanced Graph Transformer (EGT) Layer.
    This layer follows a node-dominant update scheme.
    1. Node representations are updated using a standard Graph Transformer attention mechanism,
       where edge features serve as attention biases.
    2. Edge representations are then enhanced using the newly updated node representations
       of their endpoints.
    This creates a clear, one-way information flow: h_edge -> h_node -> h_edge_new
    
    (这个核心层保持不变，因为它本身的设计是合理的，问题出在如何使用它)
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # --- Components for Node Update (Standard Graph Transformer) ---
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.edge_bias_proj = nn.Linear(d_model, num_heads)
        self.node_out_proj = nn.Linear(d_model, d_model)
        
        self.ln_node_1 = nn.LayerNorm(d_model)
        self.ln_node_2 = nn.LayerNorm(d_model)
        self.ffn_node = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        # --- Components for Edge Enhancement ---
        self.edge_enhancer = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        self.ln_edge = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_node, h_edge, edge_index):
        N = h_node.shape[0]
        device = h_node.device
        
        # === 1. Node Update Stream (Post-LN) ===
        h_node_res = h_node
        q = self.q_proj(h_node).view(N, self.num_heads, self.d_head).transpose(0, 1)
        k = self.k_proj(h_node).view(N, self.num_heads, self.d_head).transpose(0, 1)
        v = self.v_proj(h_node).view(N, self.num_heads, self.d_head).transpose(0, 1)

        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (self.d_head ** 0.5)
        
        if h_edge.numel() > 0:
            edge_bias = self.edge_bias_proj(h_edge).permute(1, 0)
            attn_bias_matrix = torch.zeros(self.num_heads, N, N, device=device)
            attn_bias_matrix[:, edge_index[0], edge_index[1]] = edge_bias
            attn_scores = attn_scores + attn_bias_matrix

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out_node = torch.bmm(attn_probs, v).transpose(0, 1).contiguous().view(N, -1)
        
        h_node = self.ln_node_1(h_node_res + self.dropout(self.node_out_proj(out_node)))
        h_node = self.ln_node_2(h_node + self.dropout(self.ffn_node(h_node)))
        
        # === 2. Edge Enhancement Stream ===
        if h_edge.numel() > 0:
            h_edge_res = h_edge
            src, dst = edge_index
            enhancer_input = torch.cat([h_node[src], h_node[dst], h_edge_res], dim=-1)
            edge_update = self.edge_enhancer(enhancer_input)
            h_edge = self.ln_edge(h_edge_res + self.dropout(edge_update))
        
        return h_node, h_edge

class EGT_GraphMAE(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, d_model, num_heads,
                 num_encoder_layers, num_decoder_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 1. Input Projection
        self.node_in_proj = nn.Linear(node_in_feats, d_model)
        self.edge_in_proj = nn.Linear(edge_in_feats, d_model)

        # 2. Mask Tokens
        self.node_mask_token = nn.Parameter(torch.randn(1, d_model))
        self.edge_mask_token = nn.Parameter(torch.randn(1, d_model))

        # 3. Encoder and Decoder using the EGT_Layer
        self.encoder = nn.ModuleList(
            [EGT_Layer(d_model, num_heads, dropout) for _ in range(num_encoder_layers)]
        )
        self.decoder = nn.ModuleList(
            [EGT_Layer(d_model, num_heads, dropout) for _ in range(num_decoder_layers)]
        )

        # 4. Reconstruction Heads
        self.node_reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )
        # Directly reconstruct from the final enhanced edge representations
        self.edge_reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, batch, node_mask_indices, edge_mask_indices):
        x, full_edge_index, full_edge_attr = batch.x, batch.edge_index, batch.edge_attr
        device = x.device
        num_nodes = batch.num_nodes
        num_edges = batch.num_edges

        # 1. Project inputs to d_model space
        h_node = self.node_in_proj(x)
        h_edge = self.edge_in_proj(full_edge_attr)

        # --- 2. ENCODER PHASE (on visible subgraph) ---
        node_visible_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        if node_mask_indices.numel() > 0:
            node_visible_mask[node_mask_indices] = False
        visible_node_indices = torch.where(node_visible_mask)[0]

        edge_visible_mask = torch.ones(num_edges, dtype=torch.bool, device=device)
        if edge_mask_indices.numel() > 0:
            edge_visible_mask[edge_mask_indices] = False
        
        src, dst = full_edge_index
        encoder_edge_mask = edge_visible_mask & node_visible_mask[src] & node_visible_mask[dst]
        
        node_map = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        node_map[visible_node_indices] = torch.arange(visible_node_indices.numel(), device=device)
        
        subgraph_edge_index = node_map[full_edge_index[:, encoder_edge_mask]]
        
        enc_h_node = h_node[visible_node_indices]
        enc_h_edge = h_edge[encoder_edge_mask]

        for layer in self.encoder:
            enc_h_node, enc_h_edge = layer(enc_h_node, enc_h_edge, subgraph_edge_index)

        # --- 3. DECODER PHASE (on full graph with masks) ---
        dec_input_h_node = h_node.clone()
        dec_input_h_edge = h_edge.clone()

        if node_mask_indices.numel() > 0:
            dec_input_h_node[node_mask_indices] = self.node_mask_token
        if edge_mask_indices.numel() > 0:
            dec_input_h_edge[edge_mask_indices] = self.edge_mask_token

        dec_input_h_node[visible_node_indices] = enc_h_node
        dec_input_h_edge[encoder_edge_mask] = enc_h_edge

        dec_h_node, dec_h_edge = dec_input_h_node, dec_input_h_edge
        for layer in self.decoder:
            dec_h_node, dec_h_edge = layer(dec_h_node, dec_h_edge, full_edge_index)
            
        # --- 4. RECONSTRUCTION ---
        reconstructed_pressures = self.node_reconstruction_head(dec_h_node)
        reconstructed_flows = self.edge_reconstruction_head(dec_h_edge)
        
        return reconstructed_pressures, reconstructed_flows
    
# file: layers.py (or wherever you define EGT)

class EGT_Layer_v3(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, max_spd=10):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # --- Re-integrated Structural Encoders ---
        self.degree_encoder = nn.Embedding(20, d_model, padding_idx=0)
        self.spd_encoder = nn.Embedding(max_spd + 2, num_heads)

        # --- Components for Node Update ---
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.edge_bias_proj = nn.Linear(d_model, num_heads) # Renamed from edge_encoder for clarity
        self.node_out_proj = nn.Linear(d_model, d_model)
        
        self.ln_node_1 = nn.LayerNorm(d_model)
        self.ln_node_2 = nn.LayerNorm(d_model)
        self.ffn_node = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        # --- Components for Edge Enhancement ---
        self.edge_enhancer = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        self.ln_edge = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_node, h_edge, edge_index, degree_enc=None, spd_matrix=None):
        N, _ = h_node.shape
        device = h_node.device
        
        # === 1. Inject Degree Information ===
        x = h_node
        if degree_enc is not None:
            degree_indices = degree_enc.squeeze(-1).long()
            x = x + self.degree_encoder(degree_indices)

        # === 2. Node Update Stream ===
        h_node_res = x
        q = self.q_proj(x).view(N, self.num_heads, self.d_head).transpose(0, 1)
        k = self.k_proj(x).view(N, self.num_heads, self.d_head).transpose(0, 1)
        v = self.v_proj(x).view(N, self.num_heads, self.d_head).transpose(0, 1)

        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (self.d_head ** 0.5)
        
        # --- Add Structural Biases (SPD and Edge Features) ---
        if spd_matrix is not None:
            structural_bias = self.spd_encoder(spd_matrix).permute(2, 0, 1)
            attn_scores = attn_scores + structural_bias

        if h_edge is not None and h_edge.numel() > 0:
            if h_edge.shape[0] == edge_index.shape[1]:
                edge_bias_proj = self.edge_bias_proj(h_edge).permute(1, 0)
                edge_bias = torch.zeros(self.num_heads, N, N, device=device)
                edge_bias[:, edge_index[0], edge_index[1]] = edge_bias_proj
                attn_scores = attn_scores + edge_bias
            else:
                 print(f"Warning: Mismatch in EGT edge count. Skipping edge bias.")

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out_node = torch.bmm(attn_probs, v).transpose(0, 1).contiguous().view(N, -1)
        
        h_node_updated = self.ln_node_1(h_node_res + self.dropout(self.node_out_proj(out_node)))
        h_node_updated = self.ln_node_2(h_node_updated + self.dropout(self.ffn_node(h_node_updated)))
        
        # === 3. Edge Enhancement Stream ===
        if h_edge is not None and h_edge.numel() > 0:
            h_edge_res = h_edge
            src, dst = edge_index
            enhancer_input = torch.cat([h_node_updated[src], h_node_updated[dst], h_edge_res], dim=-1)
            edge_update = self.edge_enhancer(enhancer_input)
            h_edge_updated = self.ln_edge(h_edge_res + self.dropout(edge_update))
        else:
            h_edge_updated = h_edge # Pass through if no edges
        
        return h_node_updated, h_edge_updated
    
# file: model.py (or wherever you define the main model)

class EGT_GraphMAE_v3(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, d_model, num_heads,
                 num_encoder_layers, num_decoder_layers, dropout=0.1, max_spd=10):
        super().__init__()
        self.d_model = d_model
        
        # 1. Input Projection
        self.node_in_proj = nn.Linear(node_in_feats, d_model)
        self.edge_in_proj = nn.Linear(edge_in_feats, d_model)
        
        # 2. Encoders and Decoders (using the new v3 layer)
        self.encoder = nn.ModuleList(
            [EGT_Layer_v3(d_model, num_heads, dropout, max_spd) for _ in range(num_encoder_layers)]
        )
        self.decoder = nn.ModuleList(
            [EGT_Layer_v3(d_model, num_heads, dropout, max_spd) for _ in range(num_decoder_layers)]
        )
        
        # 3. Mask Tokens
        self.node_mask_token = nn.Parameter(torch.randn(1, d_model))
        self.edge_mask_token = nn.Parameter(torch.randn(1, d_model))
        
        # 4. Reconstruction Heads
        self.node_reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )
        self.edge_reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model), # Changed from d_model*2
            nn.Linear(d_model, 1)
        )

    def forward(self, batch, node_mask_indices, edge_mask_indices):
        x, full_edge_index, full_edge_attr = batch.x, batch.edge_index, batch.edge_attr
        degree_enc = getattr(batch, 'degree_encoding', None)
        spd_matrix = getattr(batch, 'spd_matrix', None)
        device = x.device
        
        # 1. Project inputs
        h_node_initial = self.node_in_proj(x)
        h_edge_initial = self.edge_in_proj(full_edge_attr)

        # --- 2. ENCODER on SUBGRAPH (The winning strategy) ---
        num_nodes = batch.num_nodes
        node_visible_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        if node_mask_indices.numel() > 0:
            node_visible_mask[node_mask_indices] = False
        visible_node_indices = torch.where(node_visible_mask)[0]
        
        edge_visible_mask = torch.ones(batch.num_edges, dtype=torch.bool, device=device)
        if edge_mask_indices.numel() > 0:
            edge_visible_mask[edge_mask_indices] = False
        
        src, dst = full_edge_index
        encoder_edge_mask = edge_visible_mask & node_visible_mask[src] & node_visible_mask[dst]
        
        enc_h_node, enc_h_edge, enc_edge_index, enc_degree, enc_spd = None, None, None, None, None
        
        if visible_node_indices.numel() > 0:
            node_map = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
            node_map[visible_node_indices] = torch.arange(visible_node_indices.numel(), device=device)
            
            enc_edge_index = node_map[full_edge_index[:, encoder_edge_mask]]
            
            enc_h_node = h_node_initial[visible_node_indices]
            enc_h_edge = h_edge_initial[encoder_edge_mask]

            if degree_enc is not None:
                enc_degree = degree_enc[visible_node_indices]
            if spd_matrix is not None:
                # Create subgraph SPD matrix by indexing
                enc_spd = spd_matrix[visible_node_indices, :][:, visible_node_indices]

            for layer in self.encoder:
                enc_h_node, enc_h_edge = layer(
                    h_node=enc_h_node, 
                    h_edge=enc_h_edge,
                    edge_index=enc_edge_index,
                    degree_enc=enc_degree, 
                    spd_matrix=enc_spd
                )
        
        # --- 3. DECODER on FULL GRAPH ---
        dec_input_h_node = h_node_initial.clone()
        dec_input_h_edge = h_edge_initial.clone()

        if node_mask_indices.numel() > 0:
            dec_input_h_node[node_mask_indices] = self.node_mask_token
        if visible_node_indices.numel() > 0 and enc_h_node is not None:
            dec_input_h_node[visible_node_indices] = enc_h_node
        
        if edge_mask_indices.numel() > 0:
            dec_input_h_edge[edge_mask_indices] = self.edge_mask_token
        # Note: We don't need to put back encoded edge features, as the decoder
        # will dynamically build them from the decoded node features.

        dec_h_node, dec_h_edge = dec_input_h_node, dec_input_h_edge
        for layer in self.decoder:
            dec_h_node, dec_h_edge = layer(
                h_node=dec_h_node, 
                h_edge=dec_h_edge,
                edge_index=full_edge_index,
                degree_enc=degree_enc, 
                spd_matrix=spd_matrix
            )
            
        # --- 4. Reconstruction ---
        reconstructed_pressures = self.node_reconstruction_head(dec_h_node)
        reconstructed_flows = self.edge_reconstruction_head(dec_h_edge)
        
        return reconstructed_pressures, reconstructed_flows
    
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================================
# 1. Core Layer: EGT_Layer_v3 (This remains unchanged from the successful v3)
# =====================================================================================
class EGT_Layer_v3(nn.Module):
    """
    Enhanced Graph Transformer (EGT) Layer with Structural Information.
    - Updates nodes using Graphormer-style attention (with degree and SPD bias).
    - Enhances edge representations based on the updated node states.
    """
    def __init__(self, d_model, num_heads, dropout=0.1, max_spd=10):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # --- Structural Encoders ---
        self.degree_encoder = nn.Embedding(20, d_model, padding_idx=0)
        self.spd_encoder = nn.Embedding(max_spd + 2, num_heads)

        # --- Components for Node Update ---
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.edge_bias_proj = nn.Linear(d_model, num_heads)
        self.node_out_proj = nn.Linear(d_model, d_model)
        
        self.ln_node_1 = nn.LayerNorm(d_model)
        self.ln_node_2 = nn.LayerNorm(d_model)
        self.ffn_node = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        # --- Components for Edge Enhancement ---
        self.edge_enhancer = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        self.ln_edge = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_node, h_edge, edge_index, degree_enc=None, spd_matrix=None):
        N, _ = h_node.shape
        device = h_node.device
        
        # === 1. Inject Degree Information ===
        x = h_node
        if degree_enc is not None:
            degree_indices = degree_enc.squeeze(-1).long()
            x = x + self.degree_encoder(degree_indices)

        # === 2. Node Update Stream ===
        h_node_res = x
        q = self.q_proj(x).view(N, self.num_heads, self.d_head).transpose(0, 1)
        k = self.k_proj(x).view(N, self.num_heads, self.d_head).transpose(0, 1)
        v = self.v_proj(x).view(N, self.num_heads, self.d_head).transpose(0, 1)

        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (self.d_head ** 0.5)
        
        if spd_matrix is not None:
            structural_bias = self.spd_encoder(spd_matrix).permute(2, 0, 1)
            attn_scores = attn_scores + structural_bias

        if h_edge is not None and h_edge.numel() > 0 and h_edge.shape[0] == edge_index.shape[1]:
            edge_bias_proj = self.edge_bias_proj(h_edge).permute(1, 0)
            edge_bias = torch.zeros(self.num_heads, N, N, device=device)
            edge_bias[:, edge_index[0], edge_index[1]] = edge_bias_proj
            attn_scores = attn_scores + edge_bias

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out_node = torch.bmm(attn_probs, v).transpose(0, 1).contiguous().view(N, -1)
        
        h_node_updated = self.ln_node_1(h_node_res + self.dropout(self.node_out_proj(out_node)))
        h_node_updated = self.ln_node_2(h_node_updated + self.dropout(self.ffn_node(h_node_updated)))
        
        # === 3. Edge Enhancement Stream ===
        if h_edge is not None and h_edge.numel() > 0:
            h_edge_res = h_edge
            src, dst = edge_index
            enhancer_input = torch.cat([h_node_updated[src], h_node_updated[dst], h_edge_res], dim=-1)
            edge_update = self.edge_enhancer(enhancer_input)
            h_edge_updated = self.ln_edge(h_edge_res + self.dropout(edge_update))
        else:
            h_edge_updated = h_edge
        
        return h_node_updated, h_edge_updated


# =====================================================================================
# 2. Main Model: FusionEGT_GraphMAE (The final, upgraded version)
# =====================================================================================
class FusionEGT_GraphMAE(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, d_model, num_heads,
                 num_encoder_layers, num_decoder_layers, dropout=0.1, max_spd=10):
        super().__init__()
        self.d_model = d_model
        
        # 1. Input Projection
        self.node_in_proj = nn.Linear(node_in_feats, d_model)
        self.edge_in_proj = nn.Linear(edge_in_feats, d_model)
        
        # 2. Encoders and Decoders (using the proven v3 layer)
        self.encoder = nn.ModuleList(
            [EGT_Layer_v3(d_model, num_heads, dropout, max_spd) for _ in range(num_encoder_layers)]
        )
        self.decoder = nn.ModuleList(
            [EGT_Layer_v3(d_model, num_heads, dropout, max_spd) for _ in range(num_decoder_layers)]
        )
        
        # 3. Mask Tokens
        self.node_mask_token = nn.Parameter(torch.randn(1, d_model))
        self.edge_mask_token = nn.Parameter(torch.randn(1, d_model))
        
        # 4. Reconstruction Heads
        self.node_reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )
        
        # --- [CRITICAL UPGRADE] Fusion Edge Reconstruction Head ---
        # Input dimension is d_model * 3 (src_node + dst_node + edge)
        self.edge_reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, batch, node_mask_indices, edge_mask_indices):
        x, full_edge_index, full_edge_attr = batch.x, batch.edge_index, batch.edge_attr
        degree_enc = getattr(batch, 'degree_encoding', None)
        spd_matrix = getattr(batch, 'spd_matrix', None)
        device = x.device
        
        # 1. Project inputs
        h_node_initial = self.node_in_proj(x)
        h_edge_initial = self.edge_in_proj(full_edge_attr)

        # --- 2. ENCODER on SUBGRAPH (The winning strategy) ---
        num_nodes = batch.num_nodes
        node_visible_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        if node_mask_indices.numel() > 0:
            node_visible_mask[node_mask_indices] = False
        visible_node_indices = torch.where(node_visible_mask)[0]
        
        edge_visible_mask = torch.ones(batch.num_edges, dtype=torch.bool, device=device)
        if edge_mask_indices.numel() > 0:
            edge_visible_mask[edge_mask_indices] = False
        
        src, dst = full_edge_index
        encoder_edge_mask = edge_visible_mask & node_visible_mask[src] & node_visible_mask[dst]
        
        enc_h_node, enc_h_edge = None, None
        
        if visible_node_indices.numel() > 0:
            node_map = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
            node_map[visible_node_indices] = torch.arange(visible_node_indices.numel(), device=device)
            
            enc_edge_index = node_map[full_edge_index[:, encoder_edge_mask]]
            
            enc_h_node = h_node_initial[visible_node_indices]
            enc_h_edge = h_edge_initial[encoder_edge_mask]

            enc_degree = degree_enc[visible_node_indices] if degree_enc is not None else None
            enc_spd = spd_matrix[visible_node_indices, :][:, visible_node_indices] if spd_matrix is not None else None

            for layer in self.encoder:
                enc_h_node, enc_h_edge = layer(enc_h_node, enc_h_edge, enc_edge_index, enc_degree, enc_spd)
        
        # --- 3. DECODER on FULL GRAPH ---
        dec_input_h_node = h_node_initial.clone()
        dec_input_h_edge = h_edge_initial.clone()

        if node_mask_indices.numel() > 0:
            dec_input_h_node[node_mask_indices] = self.node_mask_token
        if visible_node_indices.numel() > 0 and enc_h_node is not None:
            dec_input_h_node[visible_node_indices] = enc_h_node
        
        if edge_mask_indices.numel() > 0:
            dec_input_h_edge[edge_mask_indices] = self.edge_mask_token

        dec_h_node, dec_h_edge = dec_input_h_node, dec_input_h_edge
        for layer in self.decoder:
            dec_h_node, dec_h_edge = layer(dec_h_node, dec_h_edge, full_edge_index, degree_enc, spd_matrix)
            
        # --- 4. [CRITICAL UPGRADE] Fusion Reconstruction ---
        reconstructed_pressures = self.node_reconstruction_head(dec_h_node)
        
        src_full, dst_full = full_edge_index
        
        # Create the fused input for the edge reconstruction head
        edge_reconstruction_input = torch.cat([
            dec_h_node[src_full], 
            dec_h_node[dst_full], 
            dec_h_edge
        ], dim=-1)
        
        reconstructed_flows = self.edge_reconstruction_head(edge_reconstruction_input)
        
        return reconstructed_pressures, reconstructed_flows
    
# The EGT_Layer_v3 remains exactly the same.

class DecoupledFusionEGT_GraphMAE(nn.Module):
    # __init__ is EXACTLY THE SAME as FusionEGT_GraphMAE
    def __init__(self, node_in_feats, edge_in_feats, d_model, num_heads,
                 num_encoder_layers, num_decoder_layers, dropout=0.1, max_spd=10):
        super().__init__()
        # ... (copy the entire __init__ from FusionEGT_GraphMAE)
        self.d_model = d_model
        self.node_in_proj = nn.Linear(node_in_feats, d_model)
        self.edge_in_proj = nn.Linear(edge_in_feats, d_model)
        self.encoder = nn.ModuleList(
            [EGT_Layer_v3(d_model, num_heads, dropout, max_spd) for _ in range(num_encoder_layers)]
        )
        self.decoder = nn.ModuleList(
            [EGT_Layer_v3(d_model, num_heads, dropout, max_spd) for _ in range(num_decoder_layers)]
        )
        self.node_mask_token = nn.Parameter(torch.randn(1, d_model))
        self.edge_mask_token = nn.Parameter(torch.randn(1, d_model))
        self.node_reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )
        self.edge_reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    # The ONLY change is in the forward pass.
    def forward(self, batch, node_mask_indices, edge_mask_indices):
        # ... (The first 3 parts: Projection, Encoder, Decoder are THE SAME as FusionEGT)
        x, full_edge_index, full_edge_attr = batch.x, batch.edge_index, batch.edge_attr
        degree_enc = getattr(batch, 'degree_encoding', None)
        spd_matrix = getattr(batch, 'spd_matrix', None)
        device = x.device
        
        h_node_initial = self.node_in_proj(x)
        h_edge_initial = self.edge_in_proj(full_edge_attr)

        # --- ENCODER on SUBGRAPH ---
        num_nodes = batch.num_nodes
        node_visible_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        if node_mask_indices.numel() > 0:
            node_visible_mask[node_mask_indices] = False
        visible_node_indices = torch.where(node_visible_mask)[0]
        
        edge_visible_mask = torch.ones(batch.num_edges, dtype=torch.bool, device=device)
        if edge_mask_indices.numel() > 0:
            edge_visible_mask[edge_mask_indices] = False
        
        src, dst = full_edge_index
        encoder_edge_mask = edge_visible_mask & node_visible_mask[src] & node_visible_mask[dst]
        
        enc_h_node, enc_h_edge = None, None
        if visible_node_indices.numel() > 0:
            node_map = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
            node_map[visible_node_indices] = torch.arange(visible_node_indices.numel(), device=device)
            enc_edge_index = node_map[full_edge_index[:, encoder_edge_mask]]
            enc_h_node = h_node_initial[visible_node_indices]
            enc_h_edge = h_edge_initial[encoder_edge_mask]
            enc_degree = degree_enc[visible_node_indices] if degree_enc is not None else None
            enc_spd = spd_matrix[visible_node_indices, :][:, visible_node_indices] if spd_matrix is not None else None
            for layer in self.encoder:
                enc_h_node, enc_h_edge = layer(enc_h_node, enc_h_edge, enc_edge_index, enc_degree, enc_spd)
        
        # --- DECODER on FULL GRAPH ---
        dec_input_h_node = h_node_initial.clone()
        dec_input_h_edge = h_edge_initial.clone()
        if node_mask_indices.numel() > 0:
            dec_input_h_node[node_mask_indices] = self.node_mask_token
        if visible_node_indices.numel() > 0 and enc_h_node is not None:
            dec_input_h_node[visible_node_indices] = enc_h_node
        if edge_mask_indices.numel() > 0:
            dec_input_h_edge[edge_mask_indices] = self.edge_mask_token
        dec_h_node, dec_h_edge = dec_input_h_node, dec_input_h_edge
        for layer in self.decoder:
            dec_h_node, dec_h_edge = layer(dec_h_node, dec_h_edge, full_edge_index, degree_enc, spd_matrix)
            
        # --- 4. [CRITICAL UPGRADE] Decoupled Fusion Reconstruction ---
        reconstructed_pressures = self.node_reconstruction_head(dec_h_node)
        
        src_full, dst_full = full_edge_index
        
        # THE ONLY LINE THAT CHANGES:
        edge_reconstruction_input = torch.cat([
            dec_h_node[src_full].detach(), # DETACH to block gradients
            dec_h_node[dst_full].detach(), # DETACH to block gradients
            dec_h_edge
        ], dim=-1)
        
        reconstructed_flows = self.edge_reconstruction_head(edge_reconstruction_input)
        
        return reconstructed_pressures, reconstructed_flows
    
class StageOne_Imputer(nn.Module):
    # This is the HydroMAE model
    def __init__(self, static_node_feats, dynamic_node_feats, static_edge_feats, dynamic_edge_feats,
                 d_model, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.1, max_spd=10):
        super().__init__()
        self.static_node_proj = nn.Linear(static_node_feats, d_model)
        self.dynamic_node_proj = nn.Linear(dynamic_node_feats, d_model)
        self.static_edge_proj = nn.Linear(static_edge_feats, d_model)
        self.dynamic_edge_proj = nn.Linear(dynamic_edge_feats, d_model)
        self.node_mask_token = nn.Parameter(torch.randn(1, d_model))
        self.edge_mask_token = nn.Parameter(torch.randn(1, d_model))
        self.encoder = nn.ModuleList([EGT_Layer_v3(d_model, num_heads, dropout, max_spd) for _ in range(num_encoder_layers)])
        self.decoder = nn.ModuleList([EGT_Layer_v3(d_model, num_heads, dropout, max_spd) for _ in range(num_decoder_layers)])
        self.node_reconstruction_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, dynamic_node_feats))
        self.edge_reconstruction_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, dynamic_edge_feats))

    def forward(self, batch, node_mask_indices, edge_mask_indices):
        x_static, x_dynamic = batch.x_static, batch.x_dynamic
        edge_attr_static, edge_attr_dynamic = batch.edge_attr_static, batch.edge_attr_dynamic
        full_edge_index, degree_enc, spd_matrix = batch.edge_index, getattr(batch, 'degree_encoding', None), getattr(batch, 'spd_matrix', None)
        device = x_static.device
        h_node_static, h_node_dynamic = self.static_node_proj(x_static), self.dynamic_node_proj(x_dynamic)
        h_edge_static, h_edge_dynamic = self.static_edge_proj(edge_attr_static), self.dynamic_edge_proj(edge_attr_dynamic)
        h_node_dynamic_masked = h_node_dynamic.clone()
        if node_mask_indices.numel() > 0: h_node_dynamic_masked[node_mask_indices] = self.node_mask_token
        h_edge_dynamic_masked = h_edge_dynamic.clone()
        if edge_mask_indices.numel() > 0: h_edge_dynamic_masked[edge_mask_indices] = self.edge_mask_token
        h_node_initial, h_edge_initial = h_node_static + h_node_dynamic_masked, h_edge_static + h_edge_dynamic_masked
        num_nodes = batch.num_nodes
        node_visible_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        if node_mask_indices.numel() > 0: node_visible_mask[node_mask_indices] = False
        visible_node_indices = torch.where(node_visible_mask)[0]
        edge_visible_mask = torch.ones(batch.num_edges, dtype=torch.bool, device=device)
        if edge_mask_indices.numel() > 0: edge_visible_mask[edge_mask_indices] = False
        src, dst = full_edge_index
        encoder_edge_mask = edge_visible_mask & node_visible_mask[src] & node_visible_mask[dst]
        enc_h_node, enc_h_edge = None, None
        if visible_node_indices.numel() > 0:
            node_map = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
            node_map[visible_node_indices] = torch.arange(visible_node_indices.numel(), device=device)
            enc_edge_index = node_map[full_edge_index[:, encoder_edge_mask]]
            enc_h_node, enc_h_edge = h_node_initial[visible_node_indices], h_edge_initial[encoder_edge_mask]
            enc_degree = degree_enc[visible_node_indices] if degree_enc is not None else None
            enc_spd = spd_matrix[visible_node_indices, :][:, visible_node_indices] if spd_matrix is not None else None
            for layer in self.encoder: enc_h_node, enc_h_edge = layer(enc_h_node, enc_h_edge, enc_edge_index, enc_degree, enc_spd)
        dec_input_h_node, dec_input_h_edge = h_node_initial.clone(), h_edge_initial.clone()
        if node_mask_indices.numel() > 0 and visible_node_indices.numel() > 0 and enc_h_node is not None:
            dec_input_h_node[visible_node_indices] = enc_h_node
        dec_h_node, dec_h_edge = dec_input_h_node, dec_input_h_edge
        for layer in self.decoder: dec_h_node, dec_h_edge = layer(dec_h_node, dec_h_edge, full_edge_index, degree_enc, spd_matrix)
        reconstructed_pressures = self.node_reconstruction_head(dec_h_node)
        reconstructed_flows = self.edge_reconstruction_head(dec_h_edge)
        return reconstructed_pressures, reconstructed_flows

class FlowPredictor(nn.Module):
    # ... (代码与之前完全相同)
    def __init__(self, node_feature_dim, edge_feature_dim, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.node_in_proj = nn.Linear(node_feature_dim, d_model)
        self.gat_layer1 = GATConv(d_model, d_model, heads=num_heads, dropout=dropout)
        self.gat_layer2 = GATConv(d_model * num_heads, d_model, heads=1, concat=False, dropout=dropout)
        head_input_dim = d_model * 2 + edge_feature_dim
        self.flow_head = nn.Sequential(nn.LayerNorm(head_input_dim), nn.Linear(head_input_dim, d_model), nn.GELU(), nn.Linear(d_model, 1))
    def forward(self, imputed_node_features, static_edge_attr, edge_index):
        h_node = self.node_in_proj(imputed_node_features)
        h_node = F.elu(self.gat_layer1(h_node, edge_index))
        h_node = self.gat_layer2(h_node, edge_index)
        src, dst = edge_index
        flow_predictor_input = torch.cat([h_node[src], h_node[dst], static_edge_attr], dim=-1)
        predicted_flows = self.flow_head(flow_predictor_input)
        return predicted_flows


# =========================================================
# 2. 最终的统一模型：TwoStagePipeline
# =========================================================
class TwoStagePipeline(nn.Module):
    def __init__(self, stage_one_args, stage_two_args):
        super().__init__()
        self.stage_one = StageOne_Imputer(**stage_one_args)
        self.stage_two = FlowPredictor(**stage_two_args)
        
    def forward(self, batch, node_mask_indices, edge_mask_indices, training_mode: str = "stage_one"):
        """
        Controls the training flow based on the mode.
        
        Args:
            batch: The input data batch.
            node_mask_indices: Indices of nodes to mask.
            edge_mask_indices: Indices of edges to mask.
            training_mode (str): One of ["stage_one", "stage_two", "inference"].
                               - "stage_one": Trains only the imputer.
                               - "stage_two": Trains only the predictor, using frozen imputer.
                               - "inference": Runs the full pipeline for evaluation.
        """
        if training_mode == "stage_one":
            # In stage one, we only need the output of the imputer for loss calculation.
            pred_pressures, pred_flows_s1 = self.stage_one(batch, node_mask_indices, edge_mask_indices)
            return pred_pressures, pred_flows_s1
        
        elif training_mode in ["stage_two", "inference"]:
            # --- Get imputed data from Stage 1 ---
            # For Stage 2 training or inference, Stage 1 should be in eval mode and gradients disabled.
            with torch.no_grad():
                imputed_pressures, _ = self.stage_one(batch, node_mask_indices, edge_mask_indices)
            
            # --- Prepare input for Stage 2 ---
            # Use observed (true) pressures where available, and imputed where they were masked.
            full_pressures = batch.x_dynamic.clone()
            full_pressures[node_mask_indices] = imputed_pressures[node_mask_indices]
            
            # For inference mode, we must detach, as there's no true value to Teacher Force
            if training_mode == "inference":
                full_pressures = full_pressures.detach()

            stage2_node_input = torch.cat([batch.x_static, full_pressures], dim=-1)
            
            # --- Get final flow prediction from Stage 2 ---
            pred_flows_s2 = self.stage_two(stage2_node_input, batch.edge_attr_static, batch.edge_index)
            
            # During inference, we return both imputed pressures and final flow predictions
            if training_mode == "inference":
                return imputed_pressures, pred_flows_s2
            # During Stage 2 training, we only need the flow predictions for the loss
            else: # training_mode == "stage_two"
                return pred_flows_s2
        else:
            raise ValueError(f"Invalid training_mode: {training_mode}")