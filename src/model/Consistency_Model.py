import math
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, TopKPooling, global_mean_pool
import numpy as np


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__(); self.d_model = d_model
    def forward(self, t):
        t = t.flatten(); device = t.device; half = self.d_model // 2
        emb = math.log(10000) / max(half - 1, 1); emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0); out = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if out.size(1) < self.d_model: out = F.pad(out, (0, self.d_model - out.size(1)))
        return out
class AdaGN(nn.Module):
    def __init__(self, d_model, d_cond, num_groups=4):
        super().__init__(); self.group_norm = nn.GroupNorm(num_groups, d_model, affine=False)
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(d_cond, 2 * d_model))
    def forward(self, x, cond):
        x_norm = self.group_norm(x); style, shift = self.cond_proj(cond).chunk(2, dim=1); return x_norm * (1 + style) + shift
class GINEBlock(nn.Module):
    """A block containing two GINEConv layers with conditioning"""
    def __init__(self, d_in, d_model, d_cond, d_out=None):
        super().__init__()
        d_out = d_out or d_in
        
        # Linear projection to handle input dimension changes (e.g., in decoder)
        self.proj_in = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()

        self.conv1 = GINEConv(nn.Sequential(nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)))
        self.conv2 = GINEConv(nn.Sequential(nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)))
        
        # AdaGN now always operates on d_model dimension
        self.norm1 = AdaGN(d_model, d_cond)
        self.norm2 = AdaGN(d_model, d_cond)

        # Residual projection to match output dimension
        self.res_proj = nn.Linear(d_model, d_out)
        # Input projection for the residual connection
        self.input_res_proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x, edge_index, edge_attr, cond):
        # Save original input for residual connection
        x_res = self.input_res_proj(x)
        
        # Project input to internal model dimension
        h = self.proj_in(x)
        
        # First GINE layer
        h_norm1 = self.norm1(h, cond)
        h = self.conv1(h_norm1, edge_index, edge_attr)
        h = F.silu(h)
        
        # Second GINE layer
        h_norm2 = self.norm2(h, cond)
        h = self.conv2(h_norm2, edge_index, edge_attr)
        h = F.silu(h)
        
        # Project output and add residual
        h_out = self.res_proj(h)
        return x_res + h_out

class WaterGUNet(nn.Module):
    def __init__(self,d_node_in=5,d_edge_in=3,d_model=64,d_time_emb=64,pool_ratios=[0.8, 0.8]):
        super().__init__()
        self.d_model = d_model
        self.time_emb = SinusoidalTimeEmbedding(d_time_emb)
        self.node_encoder = nn.Linear(d_node_in, d_model)
        self.edge_encoder = nn.Linear(d_edge_in, d_model)
        
        d_cond = d_time_emb + d_model
        
        # Encoder blocks: input and model dimensions are the same (d_model)
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        for ratio in pool_ratios:
            self.encoder_blocks.append(GINEBlock(d_model, d_model, d_cond))
            self.pools.append(TopKPooling(d_model, ratio=ratio))

        # Bottleneck block: input and model dimensions are the same (d_model)
        self.bottleneck = GINEBlock(d_model, d_model, d_cond)
        
        # Decoder blocks: input is 2 * d_model, internal is d_model, output is d_model
        self.decoder_blocks = nn.ModuleList()
        for _ in pool_ratios:
            self.decoder_blocks.append(GINEBlock(2 * d_model, d_model, d_cond, d_out=d_model))
            
        self.output_proj = nn.Sequential(nn.Linear(d_model + d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1))

    # ... forward method remains the same as the last version ...
    def forward(self, x_in, edge_index, edge_attr_in, batch_indices_in, t_per_graph):
        # ... (no changes needed here) ...
        # --- 1. Initial Embeddings ---
        x_init = self.node_encoder(x_in)
        edge_attr_init = self.edge_encoder(edge_attr_in)
        t_emb = self.time_emb(t_per_graph)
        # --- 2. Encoder Path ---
        pre_pool_states = []
        h = x_init
        edge_index_current, edge_attr_current, batch_indices_current = edge_index, edge_attr_init, batch_indices_in
        for block, pool in zip(self.encoder_blocks, self.pools):
            global_h = global_mean_pool(h, batch_indices_current)
            cond = torch.cat([t_emb, global_h], dim=1)[batch_indices_current]
            h = block(h, edge_index_current, edge_attr_current, cond)
            pre_pool_states.append({'skip_h': h, 'edge_index': edge_index_current, 'edge_attr': edge_attr_current, 'batch': batch_indices_current})
            h, edge_index_current, edge_attr_current, batch_indices_current, perm, _ = \
                pool(h, edge_index_current, edge_attr_current, batch_indices_current)
            pre_pool_states[-1]['perm'] = perm
        # --- 3. Bottleneck ---
        global_h_bottle = global_mean_pool(h, batch_indices_current)
        cond_bottle = torch.cat([t_emb, global_h_bottle], dim=1)[batch_indices_current]
        h = self.bottleneck(h, edge_index_current, edge_attr_current, cond_bottle)
        # --- 4. Decoder Path ---
        for i, block in enumerate(self.decoder_blocks):
            idx = len(self.decoder_blocks) - 1 - i
            state = pre_pool_states[idx]
            upsampled_h = torch.zeros_like(state['skip_h']); upsampled_h[state['perm']] = h
            edge_index_current, edge_attr_current, batch_indices_current = state['edge_index'], state['edge_attr'], state['batch']
            global_h = global_mean_pool(state['skip_h'], batch_indices_current)
            cond = torch.cat([t_emb, global_h], dim=1)[batch_indices_current]
            h = torch.cat([upsampled_h, state['skip_h']], dim=1)
            h = block(h, edge_index_current, edge_attr_current, cond)
        # --- 5. Final Output ---
        h = torch.cat([h, x_init], dim=1)
        out = self.output_proj(h)
        return out
    

# --- 1.2 流量预测模型 [核心修改] ---
class WaterGUNet_Flow(nn.Module):
    def __init__(self, d_node_in, d_edge_in, d_model=64, d_time_emb=64, pool_ratios=[0.8, 0.8]):
        super().__init__()
        self.d_model = d_model
        self.time_emb = SinusoidalTimeEmbedding(d_time_emb)
        self.node_encoder = nn.Linear(d_node_in, d_model)
        self.edge_encoder = nn.Linear(d_edge_in, d_model)
        d_cond = d_time_emb + d_model
        
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        for ratio in pool_ratios:
            self.encoder_blocks.append(GINEBlock(d_model, d_model, d_cond))
            self.pools.append(TopKPooling(d_model, ratio=ratio))
        self.bottleneck = GINEBlock(d_model, d_model, d_cond)
        self.decoder_blocks = nn.ModuleList()
        for _ in pool_ratios:
            self.decoder_blocks.append(GINEBlock(2 * d_model, d_model, d_cond, d_out=d_model))
            
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * 2 + d_model, d_model), # h_u || h_v || edge_attr_init
            nn.SiLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x_node_in, edge_attr_in, edge_index, batch_indices_in, t_per_graph):
        h_node_init = self.node_encoder(x_node_in) 
        edge_attr_init = self.edge_encoder(edge_attr_in)
        t_emb = self.time_emb(t_per_graph)
        
        # U-Net Body
        h = h_node_init
        edge_index_current, edge_attr_current, batch_indices_current = edge_index, edge_attr_init, batch_indices_in
        pre_pool_states = []
        for block, pool in zip(self.encoder_blocks, self.pools):
            global_h = global_mean_pool(h, batch_indices_current)
            cond = torch.cat([t_emb, global_h], dim=1)[batch_indices_current]
            h = block(h, edge_index_current, edge_attr_current, cond)
            pre_pool_states.append({'skip_h': h, 'edge_index': edge_index_current, 'edge_attr': edge_attr_current, 'batch': batch_indices_current})
            h, edge_index_current, edge_attr_current, batch_indices_current, perm, _ = pool(h, edge_index_current, edge_attr_current, batch_indices_current)
            pre_pool_states[-1]['perm'] = perm
        
        global_h_bottle = global_mean_pool(h, batch_indices_current)
        cond_bottle = torch.cat([t_emb, global_h_bottle], dim=1)[batch_indices_current]
        h = self.bottleneck(h, edge_index_current, edge_attr_current, cond_bottle)
        
        for i, block in enumerate(self.decoder_blocks):
            idx = len(self.decoder_blocks) - 1 - i
            state = pre_pool_states[idx]
            upsampled_h = torch.zeros_like(state['skip_h']); upsampled_h[state['perm']] = h
            edge_index_current, edge_attr_current, batch_indices_current = state['edge_index'], state['edge_attr'], state['batch']
            global_h = global_mean_pool(state['skip_h'], batch_indices_current)
            cond = torch.cat([t_emb, global_h], dim=1)[batch_indices_current]
            h = torch.cat([upsampled_h, state['skip_h']], dim=1)
            h = block(h, edge_index_current, edge_attr_current, cond)

        # Output Reconstruction
        src, dst = edge_index[0], edge_index[1]
        output_features = torch.cat([h[src], h[dst], edge_attr_init], dim=1)
        out_flow = self.output_proj(output_features)
        return out_flow
    
    
class UnifiedWaterGUNet(nn.Module):
    def __init__(self, d_node_in, d_edge_in, d_model, d_time_emb, pool_ratios):
        super().__init__()
        self.d_model = d_model
        
        # --- 1. 共享的输入编码器 ---
        self.time_emb = SinusoidalTimeEmbedding(d_time_emb)
        self.node_encoder = nn.Linear(d_node_in, d_model)
        self.edge_encoder = nn.Linear(d_edge_in, d_model)
        
        d_cond = d_time_emb + d_model
        
        # --- 2. 共享的U-Net骨干网络 ---
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        for ratio in pool_ratios:
            self.encoder_blocks.append(GINEBlock(d_model, d_model, d_cond))
            self.pools.append(TopKPooling(d_model, ratio=ratio))

        self.bottleneck = GINEBlock(d_model, d_model, d_cond)
        
        self.decoder_blocks = nn.ModuleList()
        for _ in pool_ratios:
            self.decoder_blocks.append(GINEBlock(2 * d_model, d_model, d_cond, d_out=d_model))
            
        # --- 3. 独立的、多任务的输出头 ---
        
        # 压力头 (作用于最终的节点表示)
        self.pressure_head = nn.Sequential(
            nn.Linear(d_model + d_model, d_model), # cat(decoder_output, node_init)
            nn.SiLU(),
            nn.Linear(d_model, 1)
        )
        
        # 流量头 (作用于端点节点表示和初始边表示)
        self.flow_head = nn.Sequential(
            nn.Linear(d_model * 2 + d_model, d_model), # h_u || h_v || edge_attr_init
            nn.SiLU(),
            nn.Linear(d_model, 1)
        )
        
    def forward(self, x_node_in, edge_attr_in, edge_index, batch_indices_in, t_per_graph):
        """
        一次前向传播，同时返回压力和流量的预测。
        """
        # --- 1. 初始嵌入 (共享) ---
        # 对模型的初始节点输入和边输入进行编码
        h_node_init = self.node_encoder(x_node_in) 
        edge_attr_init = self.edge_encoder(edge_attr_in)
        
        # 对时间步进行编码
        t_emb = self.time_emb(t_per_graph)
        
        # --- 2. U-Net 骨干网络前向传播 (共享) ---
        
        # -- 2.1 Encoder Path --
        pre_pool_states = []
        h = h_node_init
        
        # 初始化循环中使用的变量
        edge_index_current = edge_index
        edge_attr_current = edge_attr_init
        batch_indices_current = batch_indices_in
        
        for block, pool in zip(self.encoder_blocks, self.pools):
            # 计算当前图（或子图）的全局上下文表示，用于条件化
            global_h = global_mean_pool(h, batch_indices_current)
            # 将时间嵌入和全局上下文拼接，作为AdaGN的条件
            cond = torch.cat([t_emb, global_h], dim=1)[batch_indices_current]
            
            # 通过GINEBlock更新节点表示
            h = block(h, edge_index_current, edge_attr_current, cond)
            
            # 保存池化前的状态，用于解码器的skip connection
            pre_pool_states.append({
                'skip_h': h, 
                'edge_index': edge_index_current, 
                'edge_attr': edge_attr_current, 
                'batch': batch_indices_current
            })
            
            # 执行TopKPooling进行图的下采样
            h, edge_index_current, edge_attr_current, batch_indices_current, perm, _ = \
                pool(h, edge_index_current, edge_attr_current, batch_indices_current)
            
            # 保存池化操作选择的节点索引，用于上采样
            pre_pool_states[-1]['perm'] = perm
        
        # -- 2.2 Bottleneck --
        # U-Net最深层的计算
        global_h_bottle = global_mean_pool(h, batch_indices_current)
        cond_bottle = torch.cat([t_emb, global_h_bottle], dim=1)[batch_indices_current]
        h = self.bottleneck(h, edge_index_current, edge_attr_current, cond_bottle)
        
        # -- 2.3 Decoder Path --
        # 反向遍历保存的状态，进行上采样和解码
        for i, block in enumerate(self.decoder_blocks):
            # 从列表末尾取出对应的encoder层状态
            idx = len(self.decoder_blocks) - 1 - i
            state = pre_pool_states[idx]
            
            # 上采样：创建一个与池化前大小相同的零张量，然后用perm索引填充
            upsampled_h = torch.zeros_like(state['skip_h'])
            upsampled_h[state['perm']] = h
            
            # 恢复池化前的拓扑和batch索引
            edge_index_current, edge_attr_current, batch_indices_current = \
                state['edge_index'], state['edge_attr'], state['batch']
            
            # 准备解码器GINEBlock的条件
            global_h = global_mean_pool(state['skip_h'], batch_indices_current)
            cond = torch.cat([t_emb, global_h], dim=1)[batch_indices_current]
            
            # Skip Connection: 将上采样结果和encoder的输出拼接
            h = torch.cat([upsampled_h, state['skip_h']], dim=1)
            
            # 通过解码器的GINEBlock
            h = block(h, edge_index_current, edge_attr_current, cond)
            
        # 经过完整的U-Net后，h 是最终的、富含信息的节点表示
        # h 的形状为 [num_nodes, d_model]
        h_decoded = h

        # --- 3. 多任务输出头 ---
        
        # -- 3.1 压力预测 --
        # 将解码器输出与初始节点嵌入拼接，以保留原始信息
        h_for_pressure_head = torch.cat([h_decoded, h_node_init], dim=1)
        p_pred = self.pressure_head(h_for_pressure_head)
        
        # -- 3.2 流量预测 --
        # 从最终的节点表示 h_decoded 重构边属性
        src, dst = edge_index
        # 将边的两个端点节点的最终表示，以及初始的边特征拼接
        flow_head_input = torch.cat([h_decoded[src], h_decoded[dst], edge_attr_init], dim=1)
        q_pred = self.flow_head(flow_head_input)
        
        return p_pred, q_pred