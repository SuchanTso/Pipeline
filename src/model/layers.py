import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeFeatureGATLayer(nn.Module):
    """
    一个GAT层，它能将多维的边特征融入到注意力机制的计算中。
    """
    def __init__(self, node_in_features, node_out_features, edge_in_features, edge_out_features, dropout, alpha, concat=True):
        """
        Args:
            node_in_features (int): 输入节点特征的维度。
            node_out_features (int): 输出节点特征的维度。
            edge_in_features (int): 输入边特征的维度 (例如，你的4个特征)。
            edge_out_features (int): 边特征经过线性变换后的维度。
            dropout (float): Dropout比率。
            alpha (float): LeakyReLU的负斜率。
            concat (bool): 如果为True, 多头注意力结果是拼接；否则是平均。
        """
        super(EdgeFeatureGATLayer, self).__init__()
        self.dropout = dropout
        self.node_in_features = node_in_features
        self.node_out_features = node_out_features
        self.edge_in_features = edge_in_features
        self.edge_out_features = edge_out_features
        self.alpha = alpha
        self.concat = concat

        # 权重矩阵 W, 用于对节点特征进行线性变换
        self.W_node = nn.Parameter(torch.empty(size=(node_in_features, node_out_features)))
        nn.init.xavier_uniform_(self.W_node.data, gain=1.414)

        # 权重矩阵 W_e, 用于对边特征进行线性变换 (新增)
        self.W_edge = nn.Parameter(torch.empty(size=(edge_in_features, edge_out_features)))
        nn.init.xavier_uniform_(self.W_edge.data, gain=1.414)
        
        # 注意力机制的参数向量 a
        # 维度是 2 * node_out + edge_out，因为它要处理拼接后的 [Wh_i || Wh_j || We*edge_attr_ij]
        self.a = nn.Parameter(torch.empty(size=(2 * node_out_features + edge_out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, node_features, adj_mask, edge_features_tensor):
        """
        前向传播。

        Args:
            node_features (torch.Tensor): 节点特征张量, shape [N, node_in_features]。
            adj_mask (torch.Tensor): 邻接矩阵掩码, shape [N, N]。值为1表示有边，0表示无边。
            edge_features_tensor (torch.Tensor): 边特征张量, shape [N, N, edge_in_features]。
                                                 edge_features_tensor[i, j, :] 是边(i,j)的特征。
                                                 如果(i,j)没边, 特征可以是全零。

        Returns:
            torch.Tensor: 更新后的节点嵌入。
            torch.Tensor: 计算出的注意力权重矩阵。
        """
        N = node_features.shape[0]

        # Step 1: 对节点和边特征进行线性变换
        Wh = torch.mm(node_features, self.W_node)  # [N, node_out_features]
        # W_edge是[F_in, F_out], edge_features_tensor是[N,N,F_in], matmul不适用
        # 我们需要用einsum或者reshape+bmm
        We_edge_features = torch.einsum("ijf, fk -> ijk", edge_features_tensor, self.W_edge) # [N, N, edge_out_features]

        # Step 2: 计算注意力系数 e_ij
        # Wh_i 部分: Wh @ a[:node_out] -> [N, 1] -> [N, N] (broadcast)
        Wh1 = torch.matmul(Wh, self.a[:self.node_out_features, :])
        # Wh_j 部分: Wh @ a[node_out:2*node_out] -> [N, 1] -> [N, N] (broadcast)
        Wh2 = torch.matmul(Wh, self.a[self.node_out_features:2*self.node_out_features, :])
        # We*edge_attr_ij 部分: We_edge_features @ a[2*node_out:] -> [N, N, 1] -> [N, N]
        Wh3 = torch.matmul(We_edge_features, self.a[2*self.node_out_features:, :]).squeeze(-1)

        # 广播相加得到注意力系数
        e = self.leakyrelu(Wh1 + Wh2.T + Wh3) # [N, N]

        # Step 3: 应用邻接掩码并进行Softmax
        # 在没有边的地方，将注意力系数置为负无穷，这样softmax后权重为0
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_mask > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        
        # 应用dropout
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Step 4: 加权聚合邻居特征
        h_prime = torch.matmul(attention, Wh) # [N, N] @ [N, node_out_features] -> [N, node_out_features]

        if self.concat:
            return F.elu(h_prime), attention
        else:
            return h_prime, attention

    def __repr__(self):
        return (f"{self.__class__.__name__} ("
                f"node: {self.node_in_features} -> {self.node_out_features}, "
                f"edge: {self.edge_in_features} -> {self.edge_out_features})")
        
        
class MultiLayerGATEncoder(nn.Module):
    def __init__(self, n_layers, n_heads, node_in_feats, node_hid_feats, node_out_feats, 
                 edge_in_feats, edge_hid_feats, dropout, alpha):
        super(MultiLayerGATEncoder, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        
        # --- 输入层 ---
        # n_heads[0]个独立的注意力头
        self.layers.append(nn.ModuleList([
            EdgeFeatureGATLayer(
                node_in_features=node_in_feats, node_out_features=node_hid_feats,
                edge_in_features=edge_in_feats, edge_out_features=edge_hid_feats,
                dropout=dropout, alpha=alpha
            ) for _ in range(n_heads[0])
        ]))

        # --- 隐藏层 (可以堆叠多层) ---
        for i in range(1, n_layers):
            in_dim = node_hid_feats * n_heads[i-1]
            self.layers.append(nn.ModuleList([
                EdgeFeatureGATLayer(
                    node_in_features=in_dim, node_out_features=node_hid_feats,
                    edge_in_features=edge_in_feats, edge_out_features=edge_hid_feats,
                    dropout=dropout, alpha=alpha
                ) for _ in range(n_heads[i])
            ]))
            
        # --- 输出层 (通常头数为1，进行平均) ---
        in_dim_out = node_hid_feats * n_heads[-1]
        self.out_att = EdgeFeatureGATLayer(
            node_in_features=in_dim_out, node_out_features=node_out_feats,
            edge_in_features=edge_in_feats, edge_out_features=edge_hid_feats,
            dropout=dropout, alpha=alpha
        )

    def forward(self, node_features, adj_mask, edge_features_tensor):
        x = node_features
        
        # 遍历除了输出层之外的每一层
        for layer_heads in self.layers:
            x = F.dropout(x, self.dropout, training=self.training)
            # 拼接多个头的输出
            x = torch.cat([att(x, adj_mask, edge_features_tensor) for att in layer_heads], dim=1)
            x = F.elu(x)
            
        # 输出层
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj_mask, edge_features_tensor)
        # 输出层不拼接，而是隐式地在GAT层内部聚合了（如果concat=False）
        # 但我们这里的实现是单头输出，所以直接返回
        
        return F.elu(x)
    
class EdgeFeatureGATLayer(nn.Module):
    """
    一个GAT层，它能将多维的边特征融入到注意力机制的计算中。
    """
    def __init__(self, node_in_features, node_out_features, edge_in_features, edge_out_features, dropout, alpha):
        super(EdgeFeatureGATLayer, self).__init__()
        self.dropout = dropout
        self.node_in_features = node_in_features
        self.node_out_features = node_out_features
        self.edge_in_features = edge_in_features
        self.edge_out_features = edge_out_features
        self.alpha = alpha

        self.W_node = nn.Parameter(torch.empty(size=(node_in_features, node_out_features)))
        nn.init.xavier_uniform_(self.W_node.data, gain=1.414)

        self.W_edge = nn.Parameter(torch.empty(size=(edge_in_features, edge_out_features)))
        nn.init.xavier_uniform_(self.W_edge.data, gain=1.414)
        
        self.a = nn.Parameter(torch.empty(size=(2 * node_out_features + edge_out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, node_features, adj_mask, edge_features_tensor):
        N = node_features.shape[0]
        Wh = torch.mm(node_features, self.W_node)
        We_edge_features = torch.einsum("ijf, fk -> ijk", edge_features_tensor, self.W_edge)

        Wh1 = torch.matmul(Wh, self.a[:self.node_out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.node_out_features:2*self.node_out_features, :])
        Wh3 = torch.matmul(We_edge_features, self.a[2*self.node_out_features:, :]).squeeze(-1)

        e = self.leakyrelu(Wh1 + Wh2.T + Wh3)
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_mask > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)
    

# file: gma_model.py (or wherever GraphTransformerLayer is)

class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, max_spd=10):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # --- 核心修改 ---
        # 1. 移除 __init__ 中的 edge_in_feats 参数
        # 2. 确保 edge_encoder 的输入维度是 d_model
        
        self.degree_encoder = nn.Embedding(20, d_model, padding_idx=0)
        self.spd_encoder = nn.Embedding(max_spd + 2, num_heads)
        self.edge_encoder = nn.Linear(d_model, num_heads) # 输入维度现在固定为 d_model

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

    def forward(self, x_proj, edge_attr_proj, degree_enc, spd_matrix, edge_map, edge_index):
        """
        Args:
            x_proj (torch.Tensor): 投影后的节点特征 [N, d_model]
            edge_attr_proj (torch.Tensor): 投影后的边特征 [E, d_model]
            ... (其他参数不变) ...
        """
        N, _ = x_proj.shape
        device = x_proj.device

        # --- 0. 编码度数并加到节点特征上 ---
        x = x_proj
        if degree_enc is not None:
            x = x + self.degree_encoder(degree_enc.long().squeeze(-1))
        
        # --- 1. 多头注意力计算 ---
        q = self.q_proj(x).view(N, self.num_heads, self.d_head)
        k = self.k_proj(x).view(N, self.num_heads, self.d_head)
        v = self.v_proj(x).view(N, self.num_heads, self.d_head)

        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        content_attn = torch.bmm(q, k.transpose(1, 2)) / (self.d_head ** 0.5)
        
        attn_scores = content_attn
        if spd_matrix is not None:
            # (a) 结构偏置
            # spd_matrix 是 [N, N, max_spd + 2]，我们需要将其转换为 [H, N, N]
            structural_bias = self.spd_encoder(spd_matrix).permute(2, 0, 1)
            attn_scores = attn_scores + structural_bias

        # (c) 边特征偏置
        # --- 核心修改 ---
        # edge_encoder 现在直接处理投影后的 edge_attr_proj
        if edge_attr_proj is not None:
            # edge_attr_proj 是 [E, d_model]，我们需要将其转换为 [H, E]
            edge_bias_proj = self.edge_encoder(edge_attr_proj).permute(1, 0)
            edge_bias = torch.zeros(self.num_heads, N, N, device=device)
            edge_bias[:, edge_index[0], edge_index[1]] = edge_bias_proj
            attn_scores = attn_scores + edge_bias
        # attn_scores = content_attn + structural_bias + edge_bias
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.bmm(attn_probs, v)
        out = out.transpose(0, 1).contiguous().view(N, -1)
        
        # --- 2. Add & Norm, FFN ---
        x = self.layer_norm1(x + self.dropout(self.out_proj(out)))
        x = self.layer_norm2(x + self.dropout(self.ffn(x)))
        
        return x