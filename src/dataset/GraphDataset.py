from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
import numpy as np
import torch
from .Utils import preprocess_for_graph_transformer
from .normalizer import ZScoreNormalizer, LogZScoreNormalizer
from .Utils import select_random_indices
import random

class PretrainDataset(Dataset):
    def __init__(self, 
                 raw_data, 
                 fit_ratio=0.8, 
                 fit_node_mask_ratio=0.5, 
                 pressure_norm=None,
                 flow_norm=None,
                 node_static_norm=None,
                 edge_static_norm=None,
                 need_fit=True,
                 fit_pipe_mask_ratio=0.5):
        super().__init__()
        print("Initializing PretrainDataset...")
        
        # --- 1. 数据转换与拓扑计算 ---
        self._build_data_list(raw_data)
        
        # --- 2. 初始化并Fit归一化器 ---
        self.node_static_norm = node_static_norm if node_static_norm else ZScoreNormalizer()
        self.edge_static_norm = edge_static_norm if edge_static_norm else ZScoreNormalizer()
        self.pressure_norm = pressure_norm if pressure_norm else ZScoreNormalizer()
        self.flow_norm = flow_norm if flow_norm else LogZScoreNormalizer()
        if need_fit:
            print("PretrainDataset: Need to fit normalizers...")
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
                 node_static_norm=None,
                 edge_static_norm=None,
                 need_fit = True,
                 fit_pipe_mask_ratio=0.5):
        super().__init__()
        print("Initializing PretrainDataset for preprocessing and normalization fitting...")
        
        # --- 1. 数据转换与拓扑计算 ---
        self._build_data_list(raw_data)
        
        # --- 2. 初始化并Fit归一化器 ---
        self.node_static_norm = node_static_norm if node_static_norm else ZScoreNormalizer()
        self.edge_static_norm = edge_static_norm if edge_static_norm else ZScoreNormalizer()
        self.pressure_norm = pressure_norm if pressure_norm else ZScoreNormalizer()
        self.flow_norm = flow_norm if flow_norm else LogZScoreNormalizer()
        if need_fit:
            print("fitting normalizers...")
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
        self.demand_real = raw_data['demand_real']

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
                demands_real = self.demand_real,
                edge_attr_static=norm_edge_static,
                edge_attr_dynamic=norm_flows[t],
                edge_index=self.edge_index,
                node_type=self.node_type,
                degree_encoding=self.structural_data['degree_encoding'],
                spd_matrix=self.structural_data['spd_matrix']
            )
            self.processed_data_list.append(data)
            

class MultiGraphPretrainDataset(Dataset):
    """
    一个可以处理多个时序图数据的PyG数据集类。

    这个类接收一个原始数据字典的列表，其中每个字典代表一个独立的图。
    它会将所有图的所有时间步统一处理、归一化，并整合成一个大的样本列表。
    """
    def __init__(self, 
                 list_of_raw_data,          # <--- 主要变化：输入是一个列表
                 fit_ratio=0.8, 
                 pressure_norm=None,
                 flow_norm=None,
                 fit_node_mask_ratio=0.5, 
                 fit_pipe_mask_ratio=0.5):
        super().__init__()
        print("Initializing MultiGraphPretrainDataset...")
        
        # --- 1. 数据转换与拓扑计算 ---
        # 这个方法现在会遍历列表，处理所有图
        self._build_data_list(list_of_raw_data)
        
        # --- 2. 初始化并Fit共享的归一化器 ---
        # 这些normalizer将被所有图共享，确保特征尺度统一
        self.node_static_norm = ZScoreNormalizer()
        self.edge_static_norm = ZScoreNormalizer()
        self.pressure_norm = pressure_norm if pressure_norm else LogZScoreNormalizer()
        self.flow_norm = flow_norm if flow_norm else LogZScoreNormalizer()
        
        # Fit过程会使用所有图的一部分数据，结果更鲁棒
        self._fit_normalizers(fit_ratio, fit_node_mask_ratio, fit_pipe_mask_ratio)

        # --- 3. 对所有数据进行最终的归一化和特征构建 ---
        self._process_all_data()
        print(f"MultiGraphPretrainDataset initialized with {len(self.processed_data_list)} total samples (time steps).")

    def _build_data_list(self, list_of_raw_data):
        """
        遍历raw_data列表，为每个图的每个时间步创建一个Data对象。
        """
        print(f"Building data list from {len(list_of_raw_data)} graphs...")
        self.raw_data_list = []
        
        # 用来存储每个图独立的、预计算好的结构信息，避免重复计算
        self.structural_data_cache = {}

        for graph_idx, raw_data in enumerate(list_of_raw_data):
            # 从raw_data中提取节点、边等静态信息
            node_static = torch.from_numpy(np.stack([raw_data['node_elevations'], raw_data['node_demands']], axis=1)).float()
            node_type = torch.from_numpy(raw_data['node_type']).long()
            reservoir_index = torch.from_numpy(raw_data['reservoir_indices']).long()
            
            edge_static_attr_directed = np.stack([raw_data['link_diameters'], raw_data['link_lengths'], raw_data['link_roughnesses']], axis=1)
            edge_index_directed = torch.from_numpy(raw_data['edge_index_directed']).long()
            
            # 创建无向图
            row, col = edge_index_directed
            edge_index = torch.cat([torch.stack([row, col]), torch.stack([col, row])], dim=1)
            edge_static_attr = torch.from_numpy(np.concatenate([edge_static_attr_directed, edge_static_attr_directed], axis=0)).float()
            
            num_nodes = len(raw_data['node_elevations'])
            
            # --- 结构信息预计算与缓存 ---
            # 检查是否已经为这个图计算过结构信息
            if graph_idx not in self.structural_data_cache:
                print(f"  - Pre-calculating structural features for graph {graph_idx}...")
                graph_topology = Data(edge_index=edge_index, num_nodes=num_nodes)
                structural_data = preprocess_for_graph_transformer(graph_topology)
                self.structural_data_cache[graph_idx] = structural_data
            
            structural_data = self.structural_data_cache[graph_idx]
            
            # 遍历该图的所有时间步
            for t in range(raw_data['pressures'].shape[0]):
                data = Data()
                
                # 填充静态信息
                data.node_static = node_static
                data.node_type = node_type
                data.reservoir_index = reservoir_index
                data.edge_index = edge_index
                data.edge_static_attr = edge_static_attr
                # print(f"processing data.edge_static_attr: {data.edge_static_attr.shape} , {data.edge_static_attr}")
                
                # 填充动态信息 (该时间步的)
                data.pressure = torch.from_numpy(raw_data['pressures'][t]).float().view(-1, 1)
                flow_directed = torch.from_numpy(raw_data['flows'][t]).float().view(-1, 1)
                data.flow = torch.cat([flow_directed, -flow_directed], dim=0)
                
                # 填充结构信息
                data.degree_encoding = structural_data['degree_encoding']
                data.spd_matrix = structural_data['spd_matrix']
                data.edge_map = structural_data['edge_map']
                data.demands_real = torch.from_numpy(raw_data['demand_real'][t]).float().view(-1, 1)
                
                # 填充元信息
                data.num_nodes = num_nodes
                data.graph_idx = torch.tensor([graph_idx], dtype=torch.long) # <--- [关键] 记录样本来自哪个图
                
                self.raw_data_list.append(data)

    def _fit_normalizers(self, fit_ratio, node_mask_ratio, pipe_mask_ratio):
        """
        在所有图的前fit_ratio比例的数据上拟合归一化器。
        [关键修改]：只使用在模拟中被认为是“已知”的数据点进行拟合。
        """
        print("Fitting shared normalizers on observable data only...")
        fit_len = int(len(self.raw_data_list) * fit_ratio)
        fit_data = self.raw_data_list[:fit_len]
        if not fit_data: return

        all_node_static = []
        all_edge_static = []
        observable_pressures = []  # <--- 修改：只收集可观测的压力值
        observable_flows = []      # <--- 修改：只收集可观测的流量值

        for d in fit_data:
            num_nodes = d.num_nodes
            num_edges = d.num_edges
            reservoir_indices = d.reservoir_index.tolist()

            # 模拟真实世界的观测：随机选择一部分节点/边作为已知的
            # 注意：这里的 node_mask_ratio 现在代表“未知”比例，所以已知比例是 1 - node_mask_ratio
            num_known_nodes = int(num_nodes * (1 - node_mask_ratio))
            known_node_indices = select_random_indices(num_nodes, num_known_nodes, reservoir_indices)
            
            num_known_edges = int(num_edges * (1 - pipe_mask_ratio))
            known_edge_indices = select_random_indices(num_edges, num_known_edges, [])
            
            # --- 收集数据 ---
            # 静态特征通常被认为是全局已知的（如海拔、管径），所以全部收集
            all_node_static.append(d.node_static)
            all_edge_static.append(d.edge_static_attr)
            
            # 动态特征（压力、流量）只收集“已知”部分
            if len(known_node_indices) > 0:
                observable_pressures.append(d.pressure[known_node_indices])
            
            if len(known_edge_indices) > 0:
                observable_flows.append(d.flow[known_edge_indices])
        
        # 确保我们收集到了数据
        if not observable_pressures or not observable_flows:
            raise ValueError("No observable data collected for fitting normalizers. Check mask ratios.")

        # 拼接所有数据进行拟合
        self.node_static_norm.fit(torch.cat(all_node_static, dim=0))
        self.edge_static_norm.fit(torch.cat(all_edge_static, dim=0))
        
        # [关键] 只用可观测数据来fit
        self.pressure_norm.fit(torch.cat(observable_pressures, dim=0))
        self.flow_norm.fit(torch.cat(observable_flows, dim=0))
        
        print("Normalizers fitted using observable data only.")

    def _process_all_data(self):
        """
        使用已经fit好的共享normalizer，处理所有数据点。
        """
        print("Processing all samples with fitted normalizers...")
        self.processed_data_list = []
        for data in self.raw_data_list:
            p_data = data.clone()
            
            # 使用共享的normalizer进行变换
            norm_node_static = self.node_static_norm.transform(p_data.node_static)
            norm_edge_static = self.edge_static_norm.transform(p_data.edge_static_attr)
            norm_pressure = self.pressure_norm.transform(p_data.pressure)
            norm_flow = self.flow_norm.transform(p_data.flow)
            
            # 构建最终的特征 (这部分与你的代码一致)
            p_data.x_static = torch.cat([norm_node_static, p_data.node_type.view(-1, 1).float()], dim=1)
            p_data.edge_attr_static = norm_edge_static
            p_data.x_dynamic = norm_pressure # 动态节点特征是归一化后的压力
            p_data.edge_attr_dynamic = norm_flow # 动态边特征是归一化后的流量
            # p_data.demands_real = self.demands_real
            
            # 如果你的模型需要x和edge_attr，可以这样构建
            # p_data.x = torch.cat([p_data.x_static, p_data.x_dynamic], dim=1)
            # p_data.edge_attr = torch.cat([p_data.edge_attr_static, p_data.edge_attr_dynamic], dim=1)

            # 设定标签
            p_data.y_node = norm_pressure
            p_data.y_edge = norm_flow
            
            # 清理不再需要的原始属性，节省内存
            del p_data.node_static, p_data.edge_static_attr, p_data.pressure, p_data.flow
            
            self.processed_data_list.append(p_data)

    def __len__(self):
        # 返回所有图的所有时间步的总和
        return len(self.processed_data_list)

    def __getitem__(self, idx):
        # 从处理好的列表中返回一个样本
        return self.processed_data_list[idx]

    def gen_train_loader(self, train_ratio=0.8, val_ratio=0.1, batch_size=32, shuffle=True):
        """
        这个方法保持不变，因为它是在整个样本列表上进行操作。
        """
        total_len = len(self)
        indices = list(range(total_len))
        if shuffle:
            random.shuffle(indices)

        train_end = int(total_len * train_ratio)
        val_end = train_end + int(total_len * val_ratio)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train_dataset = Subset(self, train_indices)
        val_dataset   = Subset(self, val_indices)
        test_dataset  = Subset(self, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Subset已经打乱，这里shuffle可以为False
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    
if __name__ == "__main__":
    from epanet_helper import EpytHelper
    data_list = ["/data/zsc/Pipeline/data/epaNet/Anytown.inp",
                 "/data/zsc/Pipeline/data/epaNet/CTOWN.INP"]
    raw_data_list = []
    for data_path in data_list:
        epa = EpytHelper(data_path , 72)
        raw_data = epa.get_raw_data()
        raw_data_list.append(raw_data)
        epa.destroy()
    pressure_norm = LogZScoreNormalizer()
    flow_norm = LogZScoreNormalizer()
    dataset = MultiGraphPretrainDataset(raw_data_list, pressure_norm=pressure_norm, flow_norm=flow_norm)
    train_loader, val_loader, test_loader = dataset.gen_train_loader(batch_size=16)
    # print(f"train_loader[0].size():{train_loader[0].size()}")
    print("Dataset and DataLoaders are ready.")
        