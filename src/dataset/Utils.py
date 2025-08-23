import random
import math
import numpy as np
import torch
from torch_geometric.utils import to_networkx, degree
from torch_geometric.data import Data, Dataset
import networkx as nx

from torch_geometric.utils import subgraph

def get_visible_subgraph_and_features(batch, visible_node_indices, x_proj, edge_attr_proj):
    """
    一个健壮的函数，它不仅提取子图，还直接提取对应的节点和边特征。
    这避免了在主模型中进行多次和可能出错的索引操作。

    Args:
        batch: 原始的Batch对象。
        visible_node_indices: 可见节点的索引。
        x_proj: 投影后的完整节点特征矩阵 [N, d_model]。
        edge_attr_proj: 投影后的完整边特征矩阵 [E, d_model]。

    Returns:
        (Data, Tensor, Tensor):
            - visible_subgraph: 只包含拓扑的子图对象 (edge_index, num_nodes)。
            - visible_nodes_features_proj: 属于子图的节点特征。
            - visible_edge_features_proj: 属于子图的边特征。
    """
    num_nodes = batch.num_nodes
    edge_index = batch.edge_index

    # --- 1. 处理节点 ---
    visible_nodes_features_proj = x_proj[visible_node_indices]
    
    # --- 2. 处理边 ---
    if visible_node_indices.numel() > 0:
        new_edge_index, edge_mask = subgraph(
            subset=visible_node_indices,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes
        )
        # 确保 edge_mask 是布尔类型
        edge_mask = edge_mask.to(torch.bool) if edge_mask is not None else torch.zeros(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
        
        visible_edge_features_proj = edge_attr_proj[edge_mask]
    else:
        # 如果没有可见节点，子图也没有节点和边
        new_edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        visible_edge_features_proj = torch.empty((0, edge_attr_proj.shape[1]), device=edge_attr_proj.device)

    # --- 3. 创建子图对象 ---
    visible_subgraph = Data(edge_index=new_edge_index, num_nodes=visible_node_indices.size(0))
    
    return visible_subgraph, visible_nodes_features_proj, visible_edge_features_proj

def select_random_indices(index_num, num, exclude, seed=None):
    """
    从 index_range 中随机选择 num 个不在 exclude 中的数字，每次结果相同。
    如果候选数字不足 num 个，则返回所有可用的非排除项。
    
    :param index_range: 索引数量 如 8 表示 0-7
    :param num: 需要随机选择的数字个数
    :param exclude: 排除的数字列表
    :param seed: 随机种子，确保结果可重复
    :return: 随机选中的索引列表
    """
    # 将 exclude 转为集合提高查找效率
    exclude_set = set(exclude)
    
    # 获取可用的候选数字
    candidates = [i for i in range(index_num) if i not in exclude_set]
    
    # 如果候选不足，返回所有可用的
    if len(candidates) <= num:
        return candidates
    
    # 设置随机种子
    if seed is not None:
        random.seed(seed)
    
    # 随机选择 num 个不重复的数字
    selected = random.sample(candidates, num)
    
    return selected

def preprocess_for_graph_transformer(graph_topology, max_spd_cutoff=5):
    """
    为图Transformer模型计算一次性的、与拓扑相关的结构信息。

    Args:
        graph_topology (torch_geometric.data.Data): 
            一个包含图拓扑信息的Data对象 (至少需要 .edge_index 和 .num_nodes)。
        max_spd_cutoff (int): 计算最短路径时的最大路径长度截断，以防大图计算过慢。

    Returns:
        dict: 一个包含 'degree_encoding', 'spd_matrix', 'edge_map' 的字典。
    """
    print("Preprocessing for Graph Transformer...")
    
    num_nodes = graph_topology.num_nodes
    edge_index = graph_topology.edge_index
    num_edges = graph_topology.num_edges

    # 1. 计算节点度数中心性 (Degree Centrality)
    # 度的log变换可以稳定数值
    deg = degree(edge_index[0], num_nodes).float()
    degree_encoding = deg.view(-1, 1)
    print("  - Degree encoding calculated.")

    # 2. 计算最短路径距离 (Shortest Path Distance)
    # to_networkx需要一个包含num_nodes信息的Data对象
    temp_data_for_nx = Data(edge_index=edge_index, num_nodes=num_nodes)
    G = to_networkx(temp_data_for_nx, to_undirected=True)
    
    path_lengths = dict(nx.all_pairs_shortest_path_length(G, cutoff=max_spd_cutoff))
    
    spd_matrix = torch.full((num_nodes, num_nodes), float('inf'))
    for i, paths in path_lengths.items():
        for j, length in paths.items():
            spd_matrix[i, j] = length
            
    # 将inf替换为一个比cutoff大的整数，方便后续embedding
    spd_matrix[spd_matrix == float('inf')] = max_spd_cutoff + 1 
    spd_matrix = spd_matrix.long()
    print("  - Shortest path distance matrix calculated.")

    # 3. 创建边索引到边特征的映射
    # 这步是为了在Transformer层中方便地通过(i, j)找到边的特征
    # 注意：你的图是双向的，这里我们只映射一个方向，或者需要更复杂的处理
    # 简单的处理方式是，假设edge_attr的顺序与edge_index对应
    edge_map = torch.full((num_nodes, num_nodes), -1, dtype=torch.long)
    edge_map[edge_index[0], edge_index[1]] = torch.arange(num_edges)
    edge_map[edge_index[1], edge_index[0]] = torch.arange(num_edges)
    print("  - Edge map created.")

    structural_data = {
        'degree_encoding': degree_encoding,
        'spd_matrix': spd_matrix,
        'edge_map': edge_map
    }
    
    print("Preprocessing finished.")
    return structural_data
