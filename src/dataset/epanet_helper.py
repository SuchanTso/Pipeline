from epyt import epanet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

#################################EXAMPLECODE#################################
# epa_net_path = '/data/zsc/Pipeline/data/epaNet/tt.inp'
# # 加载EPANET模型
# G = epanet(epa_net_path)
# hrs = 72
# G.setTimeSimulationDuration(hrs * 3600)
# R = G.getComputedHydraulicTimeSeries()

# # 提取数据
# node_pressures = R.Pressure  # 形状: (时间步, 节点数)
# pipe_flows = R.Flow          # 形状: (时间步, 管道数)


# # Plot link flows and quality
# hrs_time = R.Time / 3600
# count = G.getNodeCount()
# print(f"Node count: {count}")
# node_indices = [1,2,3,4,5,6,7,8] #第i个节点，1~
# c_p_indices = [0, 1, 2, 3, 4, 5, 6, 7] #第i个节点，0~7
# node_names = G.getNodeNameID(node_indices)
# print(f"node_names = {node_names} , node_indices = {node_indices}")
# print(f"R.pressure ={R.Pressure}")
# G.plot_ts(X=hrs_time, Y=R.Pressure[:, c_p_indices], legend_location='best',
#           title=f'Pressure, Node IDs: {node_names}', figure_size=[4, 3],
#           xlabel='Time (hrs)', ylabel=f'Pressure ({G.units.NodePressureUnits})',
#           marker=None, labels=node_names, save_fig=True, filename='figures/paper_pressures')
# 模拟EPANET水网并获取x小时数据
#################################EXAMPLECODE#################################

class EpytHelper:
    # let's all use index when refering to nodes and pipes
    # when required we shall convert to ID by retaining the interface of epanet
    def __init__(self, epa_net_path, hrs=72):
        self.epa_net_path = epa_net_path
        self.hrs = hrs
        self.G = epanet(epa_net_path)
        self.G.setTimeSimulationDuration(hrs * 3600)
        self.R = self.G.getComputedHydraulicTimeSeries()

    def get_node_pressures(self):
        return self.R.Pressure  # 形状: (时间步, 节点数)

    def get_pipe_flows(self):
        return self.R.Flow  # 形状: (时间步, 管道数)


    def create_graph_data(self,timestep):
        # 获取当前时间步的数据
        node_features = self.get_node_pressures()[timestep]  # 向量: [num_nodes]
        edge_features = self.get_pipe_flows()[timestep]       # 向量: [num_edges]
        
        # 构建邻接表: 每个管道连接两个节点
        edge_index = []
        for pipe_id in self.G.getLinkIndex():
            start_node , end_node = self.G.getLinkNodesIndex(pipe_id)# node index start from 1 so out of index
            # print(f"{self.G.getLinkNameID(pipe_id)} :link node = {self.G.getNodeNameID([start_node, end_node])}")
            # print(f"pipe_id = {pipe_id}, start_node = {start_node}, end_node = {end_node}")
            edge_index.append([start_node - 1, end_node - 1])
        edge_index = torch.tensor(edge_index, dtype=torch.long).T  # 形状: [2, num_edges]
        
        # 节点特征、边特征转为Tensor
        x = torch.tensor(node_features, dtype=torch.float).unsqueeze(1)  # [num_nodes, 1]
        edge_attr = torch.tensor(edge_features, dtype=torch.float).unsqueeze(1)  # [num_edges, 1]
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# 创建整个时间序列的数据集
# graph_data_list = [create_graph_data(t) for t in range(x)]
# epytNet.create_graph_data(0)  # 获取第一个时间步的图数据示例


class WaterEPANetDataset(Dataset):
    def __init__(self, epa_net_path, hrs=72, transform=None):
        self.epa_net_path = epa_net_path
        self.hrs = hrs
        self.transform = transform
        super(WaterEPANetDataset, self).__init__()
        self.epyt_helper = EpytHelper(epa_net_path, hrs)
        self.data_list = [self.epyt_helper.create_graph_data(t) for t in range(hrs)]

    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def __len__(self):
        return len(self.data_list)
    
    def gen_train_loader(self, batch_size=32, shuffle=True):
        train_data, val_data, test_data = self.data_list[:int(0.8*self.hrs)], self.data_list[int(0.8*self.hrs):int(0.9*self.hrs)], self.data_list[int(0.9*self.hrs):]
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        return train_loader, val_data, test_data
    
if __name__ == '__main__':
    dataset = WaterEPANetDataset('/data/zsc/Pipeline/data/epaNet/tt.inp', hrs=72)
    print(f"Dataset length: {len(dataset)}")
    print(f"First graph data: {dataset[0]}")