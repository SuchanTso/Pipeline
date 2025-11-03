import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, TopKPooling, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch, unbatch_edge_index
import math

# --- Helper Modules ---
class SinusoidalTimeEmbedding(nn.Module):
    # (Same as your previous implementation)
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
    def forward(self, t):
        t = t.flatten()
        device = t.device
        half = self.d_model // 2
        emb = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        out = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if out.size(1) < self.d_model:
            out = F.pad(out, (0, self.d_model - out.size(1)))
        return out

class AdaGN(nn.Module):
    """Adaptive Group Normalization"""
    def __init__(self, d_model, d_cond, num_groups=4):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, d_model, affine=False)
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(d_cond, 2 * d_model))
    def forward(self, x, cond):
        x_norm = self.group_norm(x)
        style, shift = self.cond_proj(cond).chunk(2, dim=1)
        return x_norm * (1 + style) + shift

class GINEBlock(nn.Module):
    """A block containing two GINEConv layers with conditioning"""
    def __init__(self, d_model, d_cond):
        super().__init__()
        self.conv1 = GINEConv(nn.Sequential(nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)))
        self.conv2 = GINEConv(nn.Sequential(nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)))
        self.norm1 = AdaGN(d_model, d_cond)
        self.norm2 = AdaGN(d_model, d_cond)
    def forward(self, x, edge_index, edge_attr, cond):
        h = self.conv1(self.norm1(x, cond), edge_index, edge_attr)
        h = F.silu(h)
        h = self.conv2(self.norm2(h, cond), edge_index, edge_attr)
        h = F.silu(h)
        return x + h # Residual connection

# --- Main G-UNet Model ---
class WaterGUNet(nn.Module):
    def __init__(self,
                 d_node_in=5,      # pressure(1) + mask(1) + elevation(1) + demand(1) + type(1)
                 d_edge_in=3,      # diameter, length, roughness
                 d_model=64,
                 d_time_emb=64,
                 pool_ratios=[0.8, 0.8]): # Two levels of pooling
        super().__init__()
        self.d_model = d_model

        # --- 1. Embeddings ---
        self.time_emb = SinusoidalTimeEmbedding(d_time_emb)
        self.node_encoder = nn.Linear(d_node_in, d_model)
        self.edge_encoder = nn.Linear(d_edge_in, d_model)
        
        d_cond = d_time_emb + d_model # time_emb + global_graph_emb
        
        # --- 2. Encoder ---
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        for ratio in pool_ratios:
            self.encoder_blocks.append(GINEBlock(d_model, d_cond))
            self.pools.append(TopKPooling(d_model, ratio=ratio))

        # --- 3. Bottleneck ---
        self.bottleneck = GINEBlock(d_model, d_cond)
        
        # --- 4. Decoder ---
        self.decoder_blocks = nn.ModuleList()
        for _ in pool_ratios:
            # The input dimension is 2*d_model due to concatenation with skip connection
            self.decoder_blocks.append(GINEBlock(2 * d_model, d_cond))

        # --- 5. Output Projection ---
        self.output_proj = nn.Linear(2 * d_model, 1) # Final skip connection

    def forward(self, x_in, edge_index, edge_attr_in, batch, t):
        # x_in: [N, d_node_in], stacked node features (pressure, mask, static)
        # edge_attr_in: [E, d_edge_in], static edge features
        # batch: [N], batch index for each node
        # t: [B], time/noise level for each graph in the batch

        # --- 1. Initial Embeddings ---
        x = self.node_encoder(x_in)
        edge_attr = self.edge_encoder(edge_attr_in)
        t_emb = self.time_emb(t)

        # --- 2. Encoder Path ---
        skip_connections = []
        encoder_states = []

        h = x
        for i, (block, pool) in enumerate(zip(self.encoder_blocks, self.pools)):
            # Global condition for this level
            global_h = global_mean_pool(h, batch)
            cond = torch.cat([t_emb, global_h], dim=1)[batch]

            h = block(h, edge_index, edge_attr, cond)
            skip_connections.append(h)
            
            h, edge_index, edge_attr, batch, perm, _ = pool(h, edge_index, edge_attr, batch)
            encoder_states.append({'perm': perm, 'edge_index': edge_index, 'edge_attr': edge_attr, 'batch': batch})

        # --- 3. Bottleneck ---
        global_h_bottle = global_mean_pool(h, batch)
        cond_bottle = torch.cat([t_emb, global_h_bottle], dim=1)[batch]
        h = self.bottleneck(h, edge_index, edge_attr, cond_bottle)
        
        # --- 4. Decoder Path ---
        for i, block in enumerate(self.decoder_blocks):
            idx = len(self.decoder_blocks) - 1 - i
            
            # Unpooling
            state = encoder_states[idx]
            skip_h = skip_connections[idx]
            
            # Create an upsampled tensor of the correct size
            upsampled_h = torch.zeros_like(skip_h)
            upsampled_h[state['perm']] = h # Place features back using saved indices
            h = upsampled_h

            # Concatenate with skip connection
            h = torch.cat([h, skip_h], dim=1)
            
            # Get back the topology from before pooling
            edge_index = state['edge_index']
            edge_attr = state['edge_attr']
            batch = state['batch']

            # Global condition for this level
            global_h = global_mean_pool(h, batch)
            cond = torch.cat([t_emb, global_h], dim=1)[batch]
            
            h = block(h, edge_index, edge_attr, cond)

        # --- 5. Final Output ---
        # Final concatenation with initial input embedding
        h = torch.cat([h, x], dim=1)
        out = self.output_proj(h)

        return out