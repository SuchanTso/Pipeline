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
    
    def get_node_head(self):
        return self.R.Head


    def create_graph_data(self,hrs , normalizer=None):
        # TODO: normalize node features and edge features???
        #TODO: is index right?
        # 获取当前时间步的数据
        graph_data_list = []
        print(f"node pressures shape = {self.get_node_pressures().shape}")
        for timestep in range(hrs):
            graph = Data()
            node_features = self.get_node_pressures()[timestep]  # 向量: [num_nodes]
            edge_features = self.get_pipe_flows()[timestep] 
            node_head = self.get_node_head()[timestep] + self.G.getNodeElevations()  # 向量: [num_nodes]
            edge_index = []
            diameters = []
            lengths = []
            roughnesses = []
            for pipe_id in self.G.getLinkIndex():
                # pipe attributes
                start_node , end_node = self.G.getLinkNodesIndex(pipe_id)# node index start from 1 so out of index
                edge_index.append([start_node - 1, end_node - 1])
                diameters.append(float(self.G.getLinkDiameter(pipe_id)))
                lengths.append(float(self.G.getLinkLength(pipe_id)))
                roughnesses.append(float(self.G.getLinkRoughnessCoeff(pipe_id)))
            edge_index = torch.tensor(edge_index, dtype=torch.long).T  # 形状: [2, num_edges]
            
            reservoirIdx =self.G.getNodeReservoirIndex()  # 获取水库节点索引
            # print(f"demands = {self.G.getNodeBaseDemands()[1]}")
            reservoirType = [1 if i in reservoirIdx else 0 for i in range(self.G.getNodeCount())]  # [0,1] 0:非水库节点, 1:水库节点
            graph.x = torch.stack((torch.tensor(node_head) , torch.tensor(self.G.getNodeBaseDemands()[1]) , torch.tensor(reservoirType) ) , dim=1).float()  # [num_nodes, 3]  # 节点特征: [扬程, 基础需求,水库类型] #, torch.tensor(self.G.getNodeCount()*[timestep])
            graph.edge_index = edge_index  # [2, num_edges]
            # print(f"node head = {self.G.getNodeHydraulicHead()}")
            # print(f"node time head = {self.R.Head.shape}")
            graph.edge_attr = torch.stack((torch.tensor(diameters, dtype=torch.float), torch.tensor(lengths, dtype=torch.float), torch.tensor(roughnesses, dtype=torch.float)) , dim=1) # [num_edges, 4]  # 边特征: [直径, 长度, 粗糙度] #, torch.tensor(self.G.getLinkCount()*[timestep]))
            graph.y_node = torch.tensor(node_features, dtype=torch.float).reshape(-1,1)  # 节点压力特征: [num_nodes, 1]
            graph.y_edge = torch.tensor(edge_features, dtype=torch.float).reshape(-1,1)  # 管道流量特征: [num_edges, 1]
            
            if normalizer is not None:
                graph = normalizer.transform(graph)
            graph_data_list.append(graph)
        return graph_data_list
    
    def destroy(self):
        """释放EPANET资源"""
        self.G.unload()

# 创建整个时间序列的数据集
# graph_data_list = [create_graph_data(t) for t in range(x)]
# epytNet.create_graph_data(0)  # 获取第一个时间步的图数据示例


class WaterEPANetDataset(Dataset):
    def __init__(self, epa_net_path, hrs=72, normalizer=None):
        self.epa_net_path = epa_net_path
        self.hrs = hrs
        super(WaterEPANetDataset, self).__init__()
        self.epyt_helper = EpytHelper(epa_net_path, hrs)
        self.data_list = self.epyt_helper.create_graph_data(hrs)
        if normalizer is not None:
            train , val , test = self.gen_train_loader(batch_size=32, shuffle=True)
            normalizer.fit(train)
            self.data_list = self.epyt_helper.create_graph_data(hrs, normalizer)
        

    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def __len__(self):
        return len(self.data_list)
    
    def gen_train_loader(self, batch_size=32, shuffle=True):
        train_data, val_data, test_data = self.data_list[:int(0.8*self.hrs)], self.data_list[int(0.8*self.hrs):int(0.9*self.hrs)], self.data_list[int(0.9*self.hrs):]
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        return train_loader, val_data, test_data
    def __del__(self):
        self.epyt_helper.destroy()
        pass
    
if __name__ == '__main__':
    dataset = WaterEPANetDataset('/data/zsc/Pipeline/data/epaNet/tt.inp', hrs=72)
    print(f"Dataset length: {len(dataset)}")
    print(f"First graph data: {dataset[0]}")