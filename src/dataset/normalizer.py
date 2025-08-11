
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
import torch

class PowerLogTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,log_transform=False,power=4,reverse=True):
        if log_transform == True:
            self.log_transform = log_transform
            self.power = None
        else:
            self.power = power
            self.log_transform = None
        self.reverse=reverse
        self.max_ = None
        self.min_ = None
        
    def fit(self,X,y=None):        
        self.max_ = np.max(X)
        self.min_ = np.min(X)        
        return self
    
    def transform(self,X):
        if self.log_transform==True:
            if self.reverse == True:
                return np.log1p(self.max_-X)
            else:
                return np.log1p(X-self.min_)
        else:
            if self.reverse == True:
                return (self.max_-X)**(1/self.power )
            else:
                return (X-self.min_)**(1/self.power )
            
    def inverse_transform(self,X):
        if self.log_transform==True:
            if self.reverse == True:
                return (self.max_ - np.exp(X))
            else:
                return (np.exp(X) + self.min_)
        else:
            if self.reverse == True:
                return (self.max_ - X**self.power )               
            else:
                return (X**self.power + self.min_)               
    
class GraphNormalizer:
    def __init__(self, x_feat_names=['elevation','base_demand'],
                 ea_feat_names=['diameter','length','roughness'], output='pressure'):        
        # store 
        self.x_feat_names = x_feat_names
        self.ea_feat_names = ea_feat_names
        self.output = output
        
        # create separate scaler for each feature (can be improved, e.g., you can fit a scaler for multiple columns)
        self.scalers = {}
        for feat in self.x_feat_names:
            if feat == 'elevation':
                self.scalers[feat] = PowerLogTransformer(log_transform=True,reverse=False)
            else:
                self.scalers[feat] = MinMaxScaler()
        self.scalers[output] = PowerLogTransformer(log_transform=True,reverse=True)
        for feat in self.ea_feat_names:
            if feat == 'length':
                self.scalers[feat] = PowerLogTransformer(log_transform=True,reverse=False)
            else:
                self.scalers[feat] = MinMaxScaler()            
            
    def fit(self, graphs):
        ''' Fit the scalers on an array of x and ea features
        '''
        x , y_pressure , y_flow , ea = from_graphs_to_pandas(graphs)
        for ix, feat in enumerate(self.x_feat_names):
            self.scalers[feat] = self.scalers[feat].fit(x[:,ix].reshape(-1,1))
        self.scalers[self.output] = self.scalers[self.output].fit(y_pressure.reshape(-1,1))
        self.scalers[self.output] = self.scalers[self.output].fit(y_flow.reshape(-1,1))

        for ix, feat in enumerate(self.ea_feat_names):
            self.scalers[feat] = self.scalers[feat].fit(ea[:,ix].reshape(-1,1))        
        return self

    def transform(self, graph):
        ''' Transform graph based on normalizer
        '''
        graph = graph.clone()
        for ix, feat in enumerate(self.x_feat_names):#TODO: do not normalize node_type
            temp = graph.x[:,ix].numpy().reshape(-1,1)
            graph.x[:,ix] = torch.tensor(self.scalers[feat].transform(temp).reshape(-1))
        for ix, feat in enumerate(self.ea_feat_names):
            temp = graph.edge_attr[:,ix].numpy().reshape(-1,1)
            graph.edge_attr[:,ix] = torch.tensor(self.scalers[feat].transform(temp).reshape(-1))
        graph.y_node = torch.tensor(self.scalers[self.output].transform(graph.y_node.numpy().reshape(-1,1)).reshape(-1))
        graph.y_edge = torch.tensor(self.scalers[self.output].transform(graph.y_edge.numpy().reshape(-1,1)).reshape(-1))                                      
        return graph

    def inverse_transform(self, graph):
        ''' Perform inverse transformation to return original features
        '''
        graph = graph.clone()
        for ix, feat in enumerate(self.x_feat_names):
            temp = graph.x[:,ix].numpy().reshape(-1,1)
            graph.x[:,ix] = torch.tensor(self.scalers[feat].inverse_transform(temp).reshape(-1))
        for ix, feat in enumerate(self.ea_feat_names):
            temp = graph.edge_attr[:,ix].numpy().reshape(-1,1)
            graph.edge_attr[:,ix] = torch.tensor(self.scalers[feat].inverse_transform(temp).reshape(-1))
        graph.y_node = torch.tensor(self.scalers[self.output].inverse_transform(graph.y_node.numpy().reshape(-1,1)).reshape(-1))
        graph.y_edge = torch.tensor(self.scalers[self.output].inverse_transform(graph.y_edge.numpy().reshape(-1,1)).reshape(-1))                                      
        return graph
            
    def transform_array(self,z,feat_name):
        '''
            This is for MLP dataset; it can be done better (the entire thing, from raw data to datasets)
        '''
        return torch.tensor(self.scalers[feat_name].transform(z).reshape(-1))
        
    def inverse_transform_array(self,z,feat_name):
        '''
            This is for MLP dataset; it can be done better (the entire thing, from raw data to datasets)
        '''
        return self.scalers[feat_name].inverse_transform(z).reshape(-1).detach().clone()

def from_graphs_to_pandas(graphs, l_x=3, l_ea=3):
    x = []
    y_pressure = []
    y_flow = []
    ea = []
    for i, graph in enumerate(graphs):
        x.append(graph.x.numpy())
        y_pressure.append(graph.y_node.reshape(-1,1).numpy())
        y_flow.append(graph.y_edge.reshape(-1,1).numpy())
        ea.append(graph.edge_attr.numpy())     
    return np.concatenate(x,axis=0),np.concatenate(y_pressure,axis=0) , np.concatenate(y_flow , axis=0),np.concatenate(ea,axis=0)

# class ZScoreNormalizer:
#     def __init__(self):
#         self.mean = None
#         self.std = None

#     def fit(self, data: torch.Tensor , mask_index=None):
#         """
#         统计 mean 和 std（不改变输入）
#         data: [N, F] 或 [N, 1]
#         """
#         if mask_index is None:
#             self.mean = data.mean(dim=0, keepdim=True)
#             self.std = data.std(dim=0, keepdim=True)
#         else:
#             self.mean = data[~mask_index].mean(dim=0, keepdim=True)
#             self.std = data[~mask_index].std(dim=0, keepdim=True)
            
#             pass
#         self.std[self.std == 0] = 1.0  # 防止除0

#     def transform(self, data: torch.Tensor):
#         return (data - self.mean) / self.std

#     def inverse_transform(self, norm_data: torch.Tensor):
#         return norm_data * self.std + self.mean

# 放入你的 normalizer.py
class ZScoreNormalizer:
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data: torch.Tensor):
        """
        统计非零元素的均值和标准差。
        data: 任意形状的张量，0被视作无效/mask值。
        """
        # 找到所有非零元素
        non_zero_elements = data[data != 0]
        if non_zero_elements.numel() > 0:
            self.mean = non_zero_elements.mean()
            self.std = non_zero_elements.std()
        
        # 防止std为0
        if self.std < 1e-8:
            self.std = 1.0
        print(f"ZScoreNormalizer: mean={self.mean}, std={self.std}")

    def transform(self, data: torch.Tensor):
        """
        只对非零元素进行归一化，0值保持为0。
        """
        # 复制数据以避免就地修改
        norm_data = data.clone()
        # 创建非零掩码
        non_zero_mask = (data != 0)
        # 应用归一化
        norm_data[non_zero_mask] = (norm_data[non_zero_mask] - self.mean) / self.std
        return norm_data

    def inverse_transform(self, norm_data: torch.Tensor):
        """
        反归一化。同样，只对非零值操作。
        """
        # 复制数据以避免就地修改
        inv_data = norm_data.clone()
        non_zero_mask = (norm_data != 0)
        inv_data[non_zero_mask] = (inv_data[non_zero_mask] * self.std) + self.mean
        return inv_data
    
    def get_mean(self):
        return self.mean

class LogZScoreNormalizer:
    """
    一个组合了对数变换和Z-Score标准化的归一化器。
    专门用于处理具有长尾分布（数值范围跨越多个数量级）的数据，如流量。
    
    处理流程:
    1. Transform: x -> sign(x) * log(1 + |x|) -> (log_x - mean) / std
    2. Inverse Transform: y -> (y * std + mean) -> sign(y_orig_log) * (exp(|y_orig_log|) - 1)
    """
    def __init__(self):
        self.mean_log = 0.
        self.std_log = 1.

    def fit(self, data: torch.Tensor):
        """
        对数据的非零元素进行对数变换后，统计其均值和标准差。
        
        Args:
            data (torch.Tensor): 任意形状的张量，0被视作无效/mask值。
        """
        # 找到所有非零元素
        non_zero_elements = data[data != 0]
        
        if non_zero_elements.numel() > 0:
            # Step 1: 对数变换
            # 使用 sign(x) * log(1 + |x|) 来处理正负值
            log_transformed_elements = torch.sign(non_zero_elements) * torch.log1p(torch.abs(non_zero_elements))
            
            # Step 2: 计算变换后数据的均值和标准差
            self.mean_log = log_transformed_elements.mean()
            self.std_log = log_transformed_elements.std()
        
        # 防止标准差为0（当所有值都相同时）
        if self.std_log < 1e-8:
            self.std_log = 1.0

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        对非零元素应用“对数变换 + Z-Score标准化”。0值保持为0。
        
        Args:
            data (torch.Tensor): 待归一化的数据。
            
        Returns:
            torch.Tensor: 归一化后的数据。
        """
        # 复制数据以避免就地修改
        norm_data = data.clone()
        # 创建非零掩码
        non_zero_mask = (data != 0)
        
        if non_zero_mask.any():
            # 获取所有非零元素
            non_zero_elements = data[non_zero_mask]
            
            # Step 1: 对数变换
            log_transformed_elements = torch.sign(non_zero_elements) * torch.log1p(torch.abs(non_zero_elements))
            
            # Step 2: Z-Score标准化
            normalized_elements = (log_transformed_elements - self.mean_log) / self.std_log
            
            # 将归一化后的值放回原位置
            norm_data[non_zero_mask] = normalized_elements
            
        return norm_data

    def inverse_transform(self, norm_data: torch.Tensor) -> torch.Tensor:
        """
        对非零元素进行反向操作：“反向Z-Score + 反向对数变换”。0值保持为0。
        
        Args:
            norm_data (torch.Tensor): 待反归一化的数据。
            
        Returns:
            torch.Tensor: 反归一化后的数据，恢复到原始物理尺度。
        """
        # 复制数据以避免就地修改
        inv_data = norm_data.clone()
        # 创建非零掩码
        non_zero_mask = (norm_data != 0)
        
        if non_zero_mask.any():
            # 获取所有非零元素
            non_zero_elements_norm = norm_data[non_zero_mask]
            
            # Step 1: 反向Z-Score，得到对数变换空间的值
            inv_zscore_elements = (non_zero_elements_norm * self.std_log) + self.mean_log
            
            # Step 2: 反向对数变换，恢复到原始尺度
            # 使用 sign(y) * (exp(|y|) - 1)
            original_scale_elements = torch.sign(inv_zscore_elements) * (torch.exp(torch.abs(inv_zscore_elements)) - 1)

            # 将恢复后的值放回原位置
            inv_data[non_zero_mask] = original_scale_elements
            
        return inv_data
    def get_mean(self):
        return self.mean_log

if __name__ == '__main__':
    from epanet_helper import WaterEPANetDataset
    data_path = 'data/epaNet/tt.inp'
    hr = 72
    x_normalizer = ZScoreNormalizer()
    y_node_normalizer = ZScoreNormalizer()
    y_edge_normalizer = ZScoreNormalizer()
    dataset = WaterEPANetDataset(data_path, hr , x_normalizer=x_normalizer,y_node_normalizer=y_node_normalizer,y_edge_normalizer=y_edge_normalizer,window_size=5)
    train_loader , val_loader , test_loader = dataset.gen_train_loader()
    print(f"First graph data: {dataset[0]}")