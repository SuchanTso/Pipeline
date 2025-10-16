from .epanet_helper import *
from .Utils import *
from .GraphDataset import *

class WaterEPANetDataset(Dataset):
    def __init__(self, raw_data, fit_ratio=0.8, 
                 fit_node_mask_ratio=0.5, fit_pipe_mask_ratio=0.5,
                 augment_node_mask_ratio_range=(0.2, 0.8), 
                 augment_pipe_mask_ratio_range=(0.2, 0.8), 
                 window_size=5):
        super().__init__()
        print("Initializing WaterEPANetDataset for fine-tuning...")
        self.window_size = window_size
        self.augment_node_mask_ratio_range = augment_node_mask_ratio_range
        self.augment_pipe_mask_ratio_range = augment_pipe_mask_ratio_range

        # --- 1. 数据转换与拓扑计算 ---
        # 重用 PretrainDataset 的逻辑来构建和处理数据
        pretrain_helper = PretrainDataset(raw_data, fit_ratio, fit_node_mask_ratio, fit_pipe_mask_ratio)
        self.processed_data_list = pretrain_helper.processed_data_list
        self.pressure_norm = pretrain_helper.pressure_norm
        self.flow_norm = pretrain_helper.flow_norm
        print(f"WaterEPANetDataset initialized with {len(self.processed_data_list)} graphs.")

    def __len__(self):
        return len(self.processed_data_list) - self.window_size

    def __getitem__(self, idx):
        window_data_list = self.processed_data_list[idx : idx + self.window_size]
        target_data = window_data_list[-1]
        
        x_seq_list = []
        edge_attr_seq_list = []

        for data in window_data_list:
            num_nodes = data.num_nodes
            num_edges = data.num_edges
            
            node_mask_ratio = random.uniform(*self.augment_node_mask_ratio_range)
            pipe_mask_ratio = random.uniform(*self.augment_pipe_mask_ratio_range)
            
            num_masked_nodes = int(num_nodes * node_mask_ratio)
            num_masked_edges = int(num_edges * pipe_mask_ratio)
            
            reservoir_indices = data.reservoir_index.tolist()
            masked_node_indices = select_random_indices(num_nodes, num_masked_nodes, reservoir_indices)
            masked_edge_indices = select_random_indices(num_edges, num_masked_edges, [])

            # --- 动态掩码 + 特征构建 ---
            # 从已处理的data.x和data.edge_attr中提取归一化后的特征
            norm_node_static = data.x[:, 0:2]
            norm_pressure = data.x[:, 2].view(-1, 1)
            node_type = data.x[:, 3].view(-1, 1)
            
            norm_edge_static = data.edge_attr[:, 0:3]
            norm_flow = data.edge_attr[:, 3].view(-1, 1)

            masked_pressure = norm_pressure.clone()
            masked_flow = norm_flow.clone()
            mask_pressure_flag = torch.zeros_like(masked_pressure)
            mask_flow_flag = torch.zeros_like(masked_flow)

            if masked_node_indices: masked_pressure[masked_node_indices] = 0.0
            if masked_edge_indices: masked_flow[masked_edge_indices] = 0.0
            mask_pressure_flag[masked_node_indices] = 1.0
            mask_flow_flag[masked_edge_indices] = 1.0

            x = torch.cat([norm_node_static, masked_pressure, mask_pressure_flag, node_type], dim=1)
            edge_attr = torch.cat([norm_edge_static, masked_flow, mask_flow_flag], dim=1)
            
            x_seq_list.append(x)
            edge_attr_seq_list.append(edge_attr)
            
        x_seq = torch.stack(x_seq_list, dim=0)
        edge_attr_seq = torch.stack(edge_attr_seq_list, dim=0)
        
        # 标签已经是归一化后的，存储在 .y_node 和 .y_edge
        # 在_process_all_data中，我们已经创建了y_node, y_edge
        # y_node = self.pressure_norm.transform(target_data.pressure)
        # y_edge = self.flow_norm.transform(target_data.flow)
        
        return {
            'x_seq': x_seq,
            'edge_attr_seq': edge_attr_seq,
            'y_node': target_data.y_node,
            'y_edge': target_data.y_edge,
            'edge_index': target_data.edge_index,
            'degree_encoding': target_data.degree_encoding,
            'spd_matrix': target_data.spd_matrix,
            'edge_map': target_data.edge_map,
        }
        
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
    
class EST_MAE_Dataset(Dataset):
    def __init__(self, processed_data_list, window_size=12):
        super().__init__()
        self.processed_data_list = processed_data_list
        self.window_size = window_size
        print(f"EST_MAE_Dataset created with {len(self)} windows.")
        self.node_type = self.processed_data_list[0].node_type
        

    def len(self):
        # 减去窗口大小，因为每个样本都需要一个完整的历史窗口
        return len(self.processed_data_list) - self.window_size + 1

    def get(self, idx):
        window_slice = self.processed_data_list[idx : idx + self.window_size]
        
        # --- 静态和拓扑信息来自最后一帧 ---
        target_frame = window_slice[-1]
        
        # --- 时序特征 ---
        x_dynamic_seq = torch.stack([d.x_dynamic for d in window_slice], dim=0)
        edge_attr_dynamic_seq = torch.stack([d.edge_attr_dynamic for d in window_slice], dim=0)

        # --- [CRITICAL FIX] 告诉DataLoader如何批处理 ---
        # 我们创建一个新的Data对象，而不是字典
        data = Data(
            x_static=target_frame.x_static,
            x_dynamic=target_frame.x_dynamic, # y_node
            
            edge_attr_static=target_frame.edge_attr_static,
            edge_attr_dynamic=target_frame.edge_attr_dynamic, # y_edge
            
            edge_index=target_frame.edge_index,
            degree_encoding=target_frame.degree_encoding,
            spd_matrix=target_frame.spd_matrix,
            node_type=self.node_type,
            
            # --- 我们自己添加的自定义属性 ---
            x_dynamic_window=x_dynamic_seq.permute(1, 0, 2), # -> [N, T, D]
            edge_attr_dynamic_window=edge_attr_dynamic_seq.permute(1, 0, 2), # -> [E, T, D]
        )
        
        return data

    # --- [CRITICAL FIX] 实现 follow_batch 属性 ---
    # 这是一个特殊的类属性，PyG DataLoader会读取它
    @property
    def follow_batch(self):
        # 告诉DataLoader，'x_dynamic_window'的第0维（节点维）
        # 应该像'x_static'的第0维一样进行批处理（拼接和偏移）。
        # 'edge_attr_dynamic_window'的第0维（边维）
        # 应该像'edge_attr_static'的第0维一样进行批处理。
        return ['x_static', 'edge_attr_static']

    def __getitem__(self, idx):
        return self.get(idx)
    
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
