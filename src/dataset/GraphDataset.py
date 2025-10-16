from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
import numpy as np
import torch
from .Utils import preprocess_for_graph_transformer
from .normalizer import ZScoreNormalizer, LogZScoreNormalizer
from .Utils import select_random_indices

class PretrainDataset(Dataset):
    def __init__(self, 
                 raw_data, 
                 fit_ratio=0.8, 
                 fit_node_mask_ratio=0.5, 
                 pressure_norm=None,
                 flow_norm=None,
                 fit_pipe_mask_ratio=0.5):
        super().__init__()
        print("Initializing PretrainDataset...")
        
        # --- 1. 数据转换与拓扑计算 ---
        self._build_data_list(raw_data)
        
        # --- 2. 初始化并Fit归一化器 ---
        self.node_static_norm = ZScoreNormalizer()
        self.edge_static_norm = ZScoreNormalizer()
        self.pressure_norm = pressure_norm
        self.flow_norm = flow_norm
        self._fit_normalizers(fit_ratio, fit_node_mask_ratio, fit_pipe_mask_ratio)

        # --- 3. 对所有数据进行归一化和特征构建 ---
        self._process_all_data()
        print(f"PretrainDataset initialized with {len(self.processed_data_list)} graphs.")

    def _build_data_list(self, raw_data):
        node_static = torch.from_numpy(np.stack([raw_data['node_elevations'], raw_data['node_demands']], axis=1)).float()
        node_type = torch.from_numpy(raw_data['node_type']).long()
        reservoir_index = torch.from_numpy(raw_data['reservoir_indices']).long()
        
        edge_static_attr_directed = np.stack([raw_data['link_diameters'], raw_data['link_lengths'], raw_data['link_roughnesses']], axis=1)
        
        edge_index_directed = torch.from_numpy(raw_data['edge_index_directed']).long()
        row, col = edge_index_directed
        edge_index = torch.cat([torch.stack([row, col]), torch.stack([col, row])], dim=1)
        edge_static_attr = torch.from_numpy(np.concatenate([edge_static_attr_directed, edge_static_attr_directed], axis=0)).float()
        
        num_nodes = len(raw_data['node_elevations'])
        graph_topology = Data(edge_index=edge_index, num_nodes=num_nodes)
        structural_data = preprocess_for_graph_transformer(graph_topology)
        
        self.raw_data_list = []
        for t in range(raw_data['pressures'].shape[0]):
            data = Data()
            data.node_static = node_static
            data.node_type = node_type
            data.reservoir_index = reservoir_index
            data.pressure = torch.from_numpy(raw_data['pressures'][t]).float().view(-1, 1)
            data.edge_index = edge_index
            data.edge_static_attr = edge_static_attr
            flow_directed = torch.from_numpy(raw_data['flows'][t]).float().view(-1, 1)
            data.flow = torch.cat([flow_directed, -flow_directed], dim=0)
            data.degree_encoding = structural_data['degree_encoding']
            data.spd_matrix = structural_data['spd_matrix']
            data.edge_map = structural_data['edge_map']
            data.num_nodes = num_nodes
            self.raw_data_list.append(data)

    def _fit_normalizers(self, fit_ratio, node_mask_ratio, pipe_mask_ratio):
        print("PretrainDataset: Fitting normalizers...")
        fit_len = int(len(self.raw_data_list) * fit_ratio)
        fit_data = self.raw_data_list[:fit_len]
        if not fit_data: return

        template_data = fit_data[0]
        num_nodes = template_data.num_nodes
        num_edges = template_data.num_edges
        reservoir_indices = template_data.reservoir_index.tolist()
        
        self.fit_masked_node_indices = select_random_indices(num_nodes, int(num_nodes * node_mask_ratio), reservoir_indices, seed=42)
        self.fit_masked_edge_indices = select_random_indices(num_edges, int(num_edges * pipe_mask_ratio), [], seed=42)
        
        masked_pressures_for_fit = []
        masked_flows_for_fit = []
        for d in fit_data:
            pressure = d.pressure.clone()
            flow = d.flow.clone()
            if self.fit_masked_node_indices:
                pressure[self.fit_masked_node_indices] = 0.0
            if self.fit_masked_edge_indices: 
                flow[self.fit_masked_edge_indices] = 0.0
            masked_pressures_for_fit.append(pressure)
            masked_flows_for_fit.append(flow)

        all_node_static = torch.cat([d.node_static for d in fit_data], dim=0)
        all_edge_static = torch.cat([d.edge_static_attr for d in fit_data], dim=0)
        all_masked_pressures = torch.cat(masked_pressures_for_fit, dim=0)
        all_masked_flows = torch.cat(masked_flows_for_fit, dim=0)
        
        self.node_static_norm.fit(all_node_static)
        self.edge_static_norm.fit(all_edge_static)
        print("pressure_norm fit")
        self.pressure_norm.fit(all_masked_pressures)
        print("flow_norm fit")
        self.flow_norm.fit(all_masked_flows)

    def _process_all_data(self):
        self.processed_data_list = []
        for data in self.raw_data_list:
            p_data = data.clone()
            
            norm_node_static = self.node_static_norm.transform(p_data.node_static)
            norm_edge_static = self.edge_static_norm.transform(p_data.edge_static_attr)
            norm_pressure = self.pressure_norm.transform(p_data.pressure)
            norm_flow = self.flow_norm.transform(p_data.flow)
            # flow_norm_flag = torch.zeros(norm_flow.shape).long()
            # flow_norm_flag[self.fit_masked_edge_indices] = 1.0
            
            p_data.x = torch.cat([norm_node_static, norm_pressure, p_data.node_type.view(-1, 1).float()], dim=1)
            p_data.x_static = torch.cat([norm_node_static, p_data.node_type.view(-1, 1).float()], dim=1)
            p_data.edge_attr_static = norm_edge_static
            p_data.x_dynamic = norm_pressure
            p_data.edge_attr_dynamic = norm_flow
            p_data.edge_attr = torch.cat([norm_edge_static, norm_flow ], dim=1)
            p_data.y_node = norm_pressure
            p_data.y_edge = norm_flow
            
            self.processed_data_list.append(p_data)

    def __len__(self):
        return len(self.processed_data_list)

    def __getitem__(self, idx):
        return self.processed_data_list[idx]

    def gen_train_loader(self, train_ratio=0.8, val_ratio=0.1, batch_size=32, shuffle=True):
        total_len = len(self)
        train_end = int(total_len * train_ratio)
        val_end = train_end + int(total_len * val_ratio)
        train_dataset = Subset(self, range(0, train_end))
        val_dataset   = Subset(self, range(train_end, val_end))
        test_dataset  = Subset(self, range(val_end, total_len))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    
class PretrainDataset_ET(Dataset):
    def __init__(self, 
                 raw_data, 
                 fit_ratio=0.8, 
                 fit_node_mask_ratio=0.5, 
                 pressure_norm=None,
                 flow_norm=None,
                 fit_pipe_mask_ratio=0.5):
        super().__init__()
        print("Initializing PretrainDataset for preprocessing and normalization fitting...")
        
        # --- 1. 数据转换与拓扑计算 ---
        self._build_data_list(raw_data)
        
        # --- 2. 初始化并Fit归一化器 ---
        self.node_static_norm = ZScoreNormalizer()
        self.edge_static_norm = ZScoreNormalizer()
        self.pressure_norm = pressure_norm if pressure_norm else ZScoreNormalizer()
        self.flow_norm = flow_norm if flow_norm else LogZScoreNormalizer()
        self._fit_normalizers(fit_ratio)

        # --- 3. 对所有数据进行归一化和特征构建 ---
        self._process_all_data()
        print(f"PretrainDataset finished processing {len(self.processed_data_list)} graph snapshots.")

    def _build_data_list(self, raw_data):
        # 提取静态特征 (只执行一次)
        node_static_feat = torch.from_numpy(np.stack([raw_data['node_elevations'], raw_data['node_demands']], axis=1)).float()
        self.node_type = torch.from_numpy(raw_data['node_type']).long()
        # 将 node_type 组合进 node_static_feat
        self.node_static = torch.cat([node_static_feat, self.node_type.view(-1, 1).float()], dim=1)
        
        edge_static_attr_directed = np.stack([raw_data['link_diameters'], raw_data['link_lengths'], raw_data['link_roughnesses']], axis=1)
        
        edge_index_directed = torch.from_numpy(raw_data['edge_index_directed']).long()
        row, col = edge_index_directed
        self.edge_index = torch.cat([torch.stack([row, col]), torch.stack([col, row])], dim=1)
        self.edge_static_attr = torch.from_numpy(np.concatenate([edge_static_attr_directed, edge_static_attr_directed], axis=0)).float()
        
        num_nodes = len(raw_data['node_elevations'])
        graph_topology = Data(edge_index=self.edge_index, num_nodes=num_nodes)
        self.structural_data = preprocess_for_graph_transformer(graph_topology)
        
        # 提取所有时间步的动态特征
        self.pressures = torch.from_numpy(raw_data['pressures']).float().view(-1, num_nodes, 1)
        flows_directed = torch.from_numpy(raw_data['flows']).float().view(-1, self.edge_static_attr.shape[0] // 2, 1)
        self.flows = torch.cat([flows_directed, -flows_directed], dim=1)

    def _fit_normalizers(self, fit_ratio):
        print("PretrainDataset: Fitting normalizers...")
        fit_len = int(self.pressures.shape[0] * fit_ratio)
        if fit_len == 0: return

        self.node_static_norm.fit(self.node_static)
        self.edge_static_norm.fit(self.edge_static_attr)
        self.pressure_norm.fit(self.pressures[:fit_len])
        self.flow_norm.fit(self.flows[:fit_len])

    def _process_all_data(self):
        # 对所有数据进行归一化
        norm_node_static = self.node_static_norm.transform(self.node_static)
        norm_edge_static = self.edge_static_norm.transform(self.edge_static_attr)
        norm_pressures = self.pressure_norm.transform(self.pressures)
        norm_flows = self.flow_norm.transform(self.flows)

        # 构建 PyG Data 对象列表
        self.processed_data_list = []
        for t in range(self.pressures.shape[0]):
            data = Data(
                x_static=norm_node_static,
                x_dynamic=norm_pressures[t],
                edge_attr_static=norm_edge_static,
                edge_attr_dynamic=norm_flows[t],
                edge_index=self.edge_index,
                node_type=self.node_type,
                degree_encoding=self.structural_data['degree_encoding'],
                spd_matrix=self.structural_data['spd_matrix']
            )
            self.processed_data_list.append(data)