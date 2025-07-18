
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

class ZScoreNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: torch.Tensor):
        """
        统计 mean 和 std（不改变输入）
        data: [N, F] 或 [N, 1]
        """
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True)
        self.std[self.std == 0] = 1.0  # 防止除0

    def transform(self, data: torch.Tensor):
        return (data - self.mean) / self.std

    def inverse_transform(self, norm_data: torch.Tensor):
        return norm_data * self.std + self.mean


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