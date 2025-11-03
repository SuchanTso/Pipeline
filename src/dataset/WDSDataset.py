import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_undirected
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csgraph
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian as csgraph_laplacian # 为了避免命名冲突，可以给laplacian改个名
import os
from .normalizer import GraphNormalizer, ZScoreNormalizer, LogZScoreNormalizer
from .epanet_helper import EpytHelper

class WaterNetworkWindowedDataset(Dataset):
    def __init__(self, epanet_path, window_size, stride, transform=None, pre_transform=None, k_eigvecs=16, root='/data/zsc/Pipeline/data/WDSdata'):
        data_root = f"{root}/{epanet_path.split('/')[-1].replace('.inp', '')}"

        self.epanet_path = epanet_path
        self.epanet_helper = EpytHelper(epanet_path)
        self.raw_data = self.epanet_helper.get_raw_data()
        self.epanet_helper.destroy()
        self.window_size = window_size
        self.stride = stride
        self.k_eigvecs = k_eigvecs
        self.static_node_normalizer = ZScoreNormalizer()
        self.static_edge_normalizer = ZScoreNormalizer()
        self.dynamic_node_normalizer = LogZScoreNormalizer()
        super().__init__(data_root, transform, pre_transform)
        
        # 加载处理好的数据模板和完整时间序列
        self.static_data = torch.load(self.processed_paths[0] , weights_only=False)
        self.normalized_pressures = torch.load(self.processed_paths[1] , weights_only=False)
        
        # 计算样本总数
        total_timesteps = self.normalized_pressures.shape[0]
        self.num_samples = (total_timesteps - self.window_size) // self.stride + 1

    @property
    def processed_file_names(self):
        # 我们现在保存两个文件：一个静态数据模板，一个动态时间序列
        return ['static_graph_template.pt', 'normalized_pressures.pt']

    def process(self):
        # --- 1. 静态数据处理 (与之前类似，只做一次) ---
        elevations = torch.from_numpy(self.raw_data['node_elevations']).float().view(-1, 1)
        demands = torch.from_numpy(self.raw_data['node_demands']).float().view(-1, 1)
        node_type = torch.from_numpy(self.raw_data['node_type']).long()
        node_type_one_hot = F.one_hot(node_type, num_classes=node_type.max() + 1).float()
        
        x_static_node = torch.cat([elevations, demands, node_type_one_hot], dim=1)

        diameters = torch.from_numpy(self.raw_data['link_diameters']).float().view(-1, 1)
        lengths = torch.from_numpy(self.raw_data['link_lengths']).float().view(-1, 1)
        roughnesses = torch.from_numpy(self.raw_data['link_roughnesses']).float().view(-1, 1)
        edge_attr_static = torch.cat([diameters, lengths, roughnesses], dim=1)
        
        edge_index = torch.from_numpy(self.raw_data['edge_index_directed']).long()
        edge_index, edge_attr_static = to_undirected(edge_index, edge_attr_static, reduce='mean')

        # --- 2. 结构编码 (预计算) ---
        num_nodes = elevations.shape[0]
        edge_index_np = edge_index.numpy()
        adj = sp.coo_matrix((np.ones(edge_index_np.shape[1]), 
                             (edge_index_np[0], edge_index_np[1])), 
                            shape=(num_nodes, num_nodes))
        L = csgraph_laplacian(adj, normed=True)

        _, laplacian_eigenvectors = eigsh(L, k=self.k_eigvecs + 1, which='SM', tol=1e-5)
        laplacian_eigenvectors = torch.from_numpy(laplacian_eigenvectors[:, 1:]).float()

        # --- 3. 归一化处理 ---
        
        pressures_all_time = torch.from_numpy(self.raw_data['pressures']).float()
        self.static_node_normalizer.fit(x_static_node)
        self.static_edge_normalizer.fit(edge_attr_static)
        self.dynamic_node_normalizer.fit(pressures_all_time)
        
        x_static_node = self.static_node_normalizer.transform(x_static_node)
        edge_attr_static = self.static_edge_normalizer.transform(edge_attr_static)
        
        pressures_normalized = self.dynamic_node_normalizer.transform(pressures_all_time)

        # --- 4. 保存处理结果 ---
        static_data = Data(
            x_static_node=x_static_node,
            edge_index=edge_index,
            edge_attr_static=edge_attr_static,
            laplacian_eigenvectors=laplacian_eigenvectors
        )
        
        torch.save(static_data, self.processed_paths[0])
        torch.save(pressures_normalized, self.processed_paths[1])
        
        # 保存normalizers
        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save({
            'static_node': self.static_node_normalizer,
            'static_edge': self.static_edge_normalizer,
            'dynamic_node': self.dynamic_node_normalizer
        }, os.path.join(self.processed_dir, 'normalizers.pt'))

    def len(self):
        return self.num_samples

    def get(self, idx):
        """
        根据索引idx，动态地从完整时间序列中切分窗口，并与静态数据合并。
        """
        # 计算窗口的起始和结束时间步
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size

        # 从已加载的完整序列中切片
        pressure_window = self.normalized_pressures[start_idx:end_idx, :] # -> [w, N]
        
        # 调整形状以匹配模型输入
        # transpose: [N, w], unsqueeze: [N, w, 1] (F_dyn=1)
        x_dynamic_window = pressure_window.transpose(0, 1).unsqueeze(-1)
        
        # 创建一个新的Data对象，深拷贝静态数据，并添加动态窗口
        sample = self.static_data.clone()
        sample.x_dynamic_window = x_dynamic_window
        
        return sample
    
    def get_orginal_pressure_window(self, idx):
        """
        获取原始（未归一化）压力窗口，便于评估和可视化。
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        pressure_window = self.raw_data['pressures'][start_idx:end_idx, :] # -> [w, N]
        x_dynamic_window = pressure_window.transpose(0, 1)
        
        
        return x_dynamic_window
    
    
if __name__ == '__main__':


    # 2. 创建数据集实例
    # 第一次运行时，它会调用process()来处理和保存数据
    # 后续运行时，它会直接加载已处理的文件
    dataset = WaterNetworkWindowedDataset(
        epanet_path="/data/zsc/Pipeline/data/epaNet/Anytown.inp",
        window_size=6, # e.g., 12 hours
        stride=1        # e.g., stride of 1 hour
    )
    # normalized_pressures = dataset.get(0).x_dynamic_window[:, -1, :] # 最后一个时间步denormalized_pressures = dynamic_normalizer.inverse_transform(normalized_pressures)
    # denormalized_pressures = dataset.dynamic_node_normalizer.inverse_transform(normalized_pressures)
    # print("Original sample (first 5 nodes):", dataset.get_orginal_pressure_window(0)[:5].flatten())
    # print("Normalized sample (first 5 nodes):", normalized_pressures[:5].flatten())
    # print("Denormalized sample (first 5 nodes):", denormalized_pressures[:5].flatten())
