import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import os.path as osp
from datetime import datetime
from omegaconf import OmegaConf


class WaterNetworkDataset(InMemoryDataset):
    def __init__(self, root, excel_path ,config_path , sequence_length=12, forecast_length=3, transform=None, pre_transform=None):
        """
        水管网络数据集类
        
        参数:
            root: 数据集存储根目录
            excel_path: Excel文件路径
            sequence_length: 输入序列长度（时间步数）
            forecast_length: 预测序列长度（时间步数）
        """
        self.excel_path = excel_path
        self.sequence_length = sequence_length
        self.forecast_length = forecast_length
        self.config = OmegaConf.load(config_path)
        
        super().__init__(root, transform, pre_transform)
        print(f"Loading dataset from {self.processed_dir}...")
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)
    
    @property
    def raw_file_names(self):
        return [self.excel_path]
    
    @property
    def processed_file_names(self):
        return ['water_network_data.pt']
    
    def download(self):
        pass  # 不需要下载，使用现有Excel文件

    def process(self):
        # 1. 读取Excel数据
        # 节点数据
        node_df = pd.read_excel(self.excel_path, sheet_name=self.config.node_sheet_name)
        # 管道数据
        pipe_df = pd.read_excel(self.excel_path, sheet_name=self.config.pipe_sheet_name)
        
        # 2. 处理静态属性
        # 节点静态属性
        #'X(m)', 'Y(m)', 'node_epa', 'high_elevation', 
        #                    'node_type_industry', 'gis_id', 'node_type'
        node_static_cols = self.config.node_static_cols
        # print(f"node_df.columns: {node_df.columns}")
        node_static = node_df[['ID'] + node_static_cols].copy()
        
        # 管道静态属性
        #'diameter(mm)', 'wall_thickness(mm)', 'inner_diameter(mm)', 
        #                   'geometric_length(m)', 'initial_state', 'roughness', 'material'
        pipe_static_cols = self.config.pipe_static_cols
        pipe_static = pipe_df[['ID', 'upstream_node', 'end_node'] + pipe_static_cols].copy()
        
        # 创建节点ID到索引的映射
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_static['ID'])}
        
        # 3. 处理时间序列属性
        # 查找时间序列列（节点从L列开始，管道从N列开始）
        time_columns = node_df.columns[self.config.node_time_start_point:]  # 从L列（索引11）开始
        
        # 验证两个表的时间序列列是否对齐
        
        if list(pipe_df.columns[self.config.pipe_time_start_point:self.config.pipe_time_start_point+len(time_columns)]) != list(time_columns):
            raise ValueError("节点和管道的时间序列列不匹配!")
        
        # 节点时间序列数据
        node_ts = node_df[time_columns].values
        # 管道时间序列数据
        pipe_ts = pipe_df[time_columns].values
        
        # 4. 预处理静态特征
        # 节点静态特征预处理
        node_static_features, node_encoders = self._preprocess_features(node_static, node_static_cols)
        # 管道静态特征预处理
        pipe_static_features, pipe_encoders = self._preprocess_features(pipe_static, pipe_static_cols)
        
        # 5. 构建图结构
        # 节点索引映射
        pipe_static['source_idx'] = pipe_static['upstream_node'].map(node_id_to_idx)
        pipe_static['target_idx'] = pipe_static['end_node'].map(node_id_to_idx)
        
        # 构建边索引
        edge_index = pipe_static[['source_idx', 'target_idx']].values.T
        
        # 6. 准备时间序列数据片段
        data_list = []
        total_timesteps = len(time_columns)
        
        # 循环生成时间序列片段
        for start_idx in range(0, total_timesteps - self.sequence_length - self.forecast_length + 1):
            # 输入序列结束索引
            end_idx = start_idx + self.sequence_length
            # 预测序列开始索引
            pred_start_idx = end_idx
            pred_end_idx = pred_start_idx + self.forecast_length
            
            # 节点时间序列片段
            node_ts_input = node_ts[:, start_idx:end_idx]
            node_ts_target = node_ts[:, pred_start_idx:pred_end_idx]
            
            # 管道时间序列片段
            pipe_ts_input = pipe_ts[:, start_idx:end_idx]
            pipe_ts_target = pipe_ts[:, pred_start_idx:pred_end_idx]
            
            # 创建图数据对象
            data = Data(
                # 静态属性
                x=torch.tensor(node_static_features, dtype=torch.float),
                edge_attr=torch.tensor(pipe_static_features, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                
                # 时间序列
                node_ts_input=torch.tensor(node_ts_input, dtype=torch.float),
                node_ts_target=torch.tensor(node_ts_target, dtype=torch.float),
                pipe_ts_input=torch.tensor(pipe_ts_input, dtype=torch.float),
                pipe_ts_target=torch.tensor(pipe_ts_target, dtype=torch.float),
                
                # 时间索引
                time_start_index=start_idx,
                time_end_index=end_idx,
                prediction_start_index=pred_start_idx
            )
            
            data_list.append(data)
        
        # 保存处理后的数据
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def _preprocess_features(self, df, feature_cols):
        """预处理节点和管道的静态特征"""
        # 分离数值和分类特征
        numeric_feats = []
        categorical_feats = []
        
        for col in feature_cols:
            if df[col].dtype in [np.float64, np.int64]:
                numeric_feats.append(col)
            else:
                categorical_feats.append(col)
        
        # 处理数值特征
        numeric_scaler = StandardScaler()
        numeric_data = numeric_scaler.fit_transform(df[numeric_feats]) if numeric_feats else np.array([])
        
        # 处理分类特征
        encoded_categorical = []
        label_encoders = {}
        
        for col in categorical_feats:
            le = LabelEncoder()
            encoded = le.fit_transform(df[col].astype(str))
            encoded_categorical.append(encoded.reshape(-1, 1))
            label_encoders[col] = le
        
        # 合并所有特征
        all_features = np.hstack([numeric_data] + encoded_categorical) if numeric_feats or categorical_feats else np.zeros((len(df), 1))
        
        return all_features, label_encoders

def print_dataset_info(dataset):
    """打印数据集信息"""
    print(f"数据集包含 {len(dataset)} 个时间序列样本")
    print(f"每个样本包含:")
    print(f"  - {dataset[0].x.shape[0]} 个节点")
    print(f"  - {dataset[0].edge_index.shape[1]} 条管道")
    print(f"  - 输入序列长度: {dataset[0].node_ts_input.shape[1]} 个时间步")
    print(f"  - 预测序列长度: {dataset[0].node_ts_target.shape[1]} 个时间步")
    
    # 检查一个样本
    sample = dataset[0]
    print("\n样本0的维度:")
    print(f"节点特征: {sample.x.shape}")
    print(f"边索引: {sample.edge_index.shape}")
    print(f"边特征: {sample.edge_attr.shape}")
    print(f"节点输入时间序列: {sample.node_ts_input.shape}")
    print(f"节点目标时间序列: {sample.node_ts_target.shape}")
    print(f"管道输入时间序列: {sample.pipe_ts_input.shape}")
    print(f"管道目标时间序列: {sample.pipe_ts_target.shape}")

# 使用示例
if __name__ == "__main__":
    # 配置参数
    data_dir = 'data'
    excel_path = 'outputs/Shiqi.xlsx'  # 替换为您的Excel文件路径
    config_path = "config/dataset.yaml"  # 替换为您的配置文件路径
    sequence_length = 24  # 输入时间步数 (例如 24小时)
    forecast_length = 6   # 预测时间步数 (例如 6小时)
    
    # 创建数据集
    dataset = WaterNetworkDataset(
        root=data_dir,
        excel_path=excel_path,
        config_path=config_path,
        sequence_length=sequence_length,
        forecast_length=forecast_length
    )
    
    # 打印数据集信息
    print_dataset_info(dataset)
    
    # 提取第一个样本用于测试
    sample = dataset[0]
    print(f"sample:{sample}")
    
    # 可以访问以下属性:
    # sample.x: 节点静态特征 [num_nodes, num_static_features]
    # sample.edge_index: 边连接关系 [2, num_edges]
    # sample.edge_attr: 管道静态特征 [num_edges, num_static_features]
    # sample.node_ts_input: 节点时间序列输入 [num_nodes, sequence_length]
    # sample.node_ts_target: 节点时间序列目标 [num_nodes, forecast_length]
    # sample.pipe_ts_input: 管道时间序列输入 [num_edges, sequence_length]
    # sample.pipe_ts_target: 管道时间序列目标 [num_edges, forecast_length]
    # sample.time_start_index: 时间序列开始索引
    # sample.time_end_index: 时间序列结束索引
    # sample.prediction_start_index: 预测开始索引