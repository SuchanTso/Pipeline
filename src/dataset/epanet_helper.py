from epyt import epanet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import random
import math
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
    
    def gen_node_static_features(self , mask_num=0):
        node_count = self.G.getNodeCount()
        node_elevation = self.G.getNodeElevations()  # 向量: [num_nodes] # self.get_node_head()[timestep] head indicates the pressure
        reservoirIdx =[index - 1 for index in self.G.getNodeReservoirIndex()]  # 获取水库节点索引
        masked_index = select_random_indices(node_count , mask_num , reservoirIdx)
        reservoirType = [1 if i in reservoirIdx else 0 for i in range(node_count)]  # [0,1] 0:非水库节点, 1:水库节点
        print(f"masked_index: {masked_index}")
        return node_elevation , self.G.getNodeBaseDemands()[1] , reservoirType , masked_index , reservoirIdx
    
    def gen_node_masked_pressure(self , timestep ,reservoir_idx, masked_index=None):
        # print(f"masked_index: {masked_index}")
        node_pressure = self.get_node_pressures()[timestep].copy()
        for i in masked_index:
            node_pressure[i] = 0
        # node_pressure = torch.tensor([self.get_node_pressures()[timestep] if i is not in masked_index else 0 for i in node_count], dtype=torch.float).reshape(-1,1)
        node_pressure = torch.tensor(node_pressure, dtype=torch.float)
        # print(node_pressure)
        return node_pressure
    
    def gen_edge_masked_flow(self, timestep , masked_index):
        link_count = self.G.getLinkCount()
        edge_flow = self.get_pipe_flows()[timestep].copy()
        for i in masked_index:
            edge_flow[i] = 0
        edge_flow = torch.tensor(edge_flow, dtype=torch.float).reshape(-1,1)
        return edge_flow
    
    def gen_edge_features(self , mask_num = 0):
        edge_index = []
        diameters = []
        lengths = []
        roughnesses = []
        link_count = self.G.getLinkCount()
        for pipe_id in self.G.getLinkIndex():
            # pipe attributes
            start_node , end_node = self.G.getLinkNodesIndex(pipe_id)# node index start from 1 so out of index
            edge_index.append([start_node - 1, end_node - 1])
            diameters.append(float(self.G.getLinkDiameter(pipe_id)))
            lengths.append(float(self.G.getLinkLength(pipe_id)))
            roughnesses.append(float(self.G.getLinkRoughnessCoeff(pipe_id)))
        edge_attr = np.stack([diameters, lengths, roughnesses], axis=1)
        edge_index = torch.tensor(edge_index, dtype=torch.long).T  # 形状: [2, num_edges]
        masked_index = select_random_indices(link_count,mask_num,[])
        return edge_index, edge_attr , masked_index

    def create_graph_data(self,hrs , normalizer=None , mask_ratio=0.3 , pipe_mask_ratio=0.3):
        # 3 , 4 , 5 , 6 , 7 , 8 , 1 , 2
        # 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7
        # 获取当前时间步的数据
        graph_data_list = []
        mask_num = math.floor(mask_ratio * self.G.getNodeCount())
        pipe_mask_num = math.floor(pipe_mask_ratio * self.G.getLinkCount())
        print(f"Masking {mask_num} nodes out of {self.G.getNodeCount()} nodes")
        print(f"Masking {pipe_mask_num} pipes out of {self.G.getLinkCount()} pipes")
        edge_index, edge_attr , pipe_masked_index = self.gen_edge_features(mask_num=pipe_mask_num)  # 获取管道特征
        node_elevation , node_demands , reservoir_type , masked_index , reservoir_index = self.gen_node_static_features(mask_num)
        for timestep in range(hrs):
            graph = Data()
            masked_pressure = self.gen_node_masked_pressure(timestep ,reservoir_index, masked_index)
            masked_flow = self.gen_edge_masked_flow(timestep ,pipe_masked_index)
            graph.x = torch.stack((torch.tensor(node_elevation) , torch.tensor(node_demands) , masked_pressure.detach().clone() , torch.tensor(reservoir_type)) , dim=1).float()  # 节点特征: [num_nodes, 4]  # [扬程, 基础需求,水库类型]
            graph.y_node = torch.tensor(self.get_node_pressures()[timestep], dtype=torch.float).reshape(-1,1)  # 节点压力特征: [num_nodes, 1]
            graph.y_edge = torch.tensor(self.get_pipe_flows()[timestep] , dtype=torch.float).reshape(-1,1)  # 管道流量特征: [num_edges, 1]
            graph.edge_index = edge_index  # 边索引: [2, num_edges]
            graph.edge_attr = torch.cat((torch.tensor(edge_attr, dtype=torch.float),masked_flow.detach().clone()),dim=1).float()  # 边特
            # print(f"edge_attr.shape:{graph.edge_attr.shape}")
            # if normalizer is not None:
            #     graph = normalizer.transform(graph)
            graph_data_list.append(graph)
        return graph_data_list
    
    def destroy(self):
        """释放EPANET资源"""
        self.G.unload()

# 创建整个时间序列的数据集
# graph_data_list = [create_graph_data(t) for t in range(x)]
# epytNet.create_graph_data(0)  # 获取第一个时间步的图数据示例
def select_random_indices(index_num, num, exclude, seed=42):
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
    random.seed(seed)
    
    # 随机选择 num 个不重复的数字
    selected = random.sample(candidates, num)
    
    return selected


class WaterEPANetDataset(Dataset):
    def __init__(self, epa_net_path, hrs=72, x_normalizer=None , y_node_normalizer = None , y_edge_normalizer = None , fit_ratio = 0.7,masked_ratio =0.0 , window_size=5):
        self.epa_net_path = epa_net_path
        self.hrs = hrs
        self.window_size = window_size
        super(WaterEPANetDataset, self).__init__()
        self.epyt_helper = EpytHelper(epa_net_path, hrs)
        self.data_list = self.epyt_helper.create_graph_data(hrs,mask_ratio=masked_ratio ,pipe_mask_ratio=masked_ratio)
        
        # 内部归一化器
        self.x_norm = x_normalizer
        self.y_node_norm = y_node_normalizer
        self.y_edge_norm = y_edge_normalizer

        # 自动归一化
        self._fit_and_normalize(fit_ratio)

    def _fit_and_normalize(self, fit_ratio):
        if self.x_norm is not None and self.y_edge_norm is not None and self.y_node_norm is not None:
            fit_len = int(len(self.data_list) * fit_ratio)
            fit_data = self.data_list[:fit_len]
            # 统计归一化参数
            all_x = torch.cat([d.x for d in fit_data], dim=0)
            all_y_node = torch.cat([d.y_node for d in fit_data], dim=0)
            all_y_edge = torch.cat([d.y_edge for d in fit_data], dim=0)

            self.x_norm.fit(all_x)
            self.y_node_norm.fit(all_y_node)
            self.y_edge_norm.fit(all_y_edge)
            print(f"normalizer fit done")
            # 所有数据归一化（就地修改）
            for d in self.data_list:
                d.x = self.x_norm.transform(d.x)
                d.y_node = self.y_node_norm.transform(d.y_node)
                d.y_edge = self.y_edge_norm.transform(d.y_edge)

    def __getitem__(self, idx):
        x_seq = []
        for i in range(self.window_size):
            x_seq.append(self.data_list[idx + i].x)  # [N, F]
        x_seq = torch.stack(x_seq, dim=0)  # [T, N, F]

        # 当前时刻对应的标签
        target = self.data_list[idx + self.window_size]
        y_node = target.y_node  # [N, 1]
        y_edge = target.y_edge  # [E, 1]
        edge_index = target.edge_index  # [2, E]
        edge_attr = target.edge_attr    # [E, 3]

        return x_seq, edge_index, edge_attr, y_node, y_edge
    
    def __len__(self):
        return len(self.data_list) - self.window_size
    
    def gen_train_loader(
        self,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size=16,
        shuffle=True,
        num_workers=0
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "划分比例必须加起来为1"

        total_len = len(self)
        train_end = int(total_len * train_ratio)
        val_end = train_end + int(total_len * val_ratio)

        train_dataset = Subset(self, range(0, train_end))
        val_dataset   = Subset(self, range(train_end, val_end))
        test_dataset  = Subset(self, range(val_end, total_len))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, test_loader
    def __del__(self):
        self.epyt_helper.destroy()
        pass
    
if __name__ == '__main__':
    dataset = WaterEPANetDataset('/data/zsc/Pipeline/data/epaNet/tt.inp', hrs=72,masked_ratio=0.5)
    print(f"Dataset length: {len(dataset)}")
    # print(f"First graph data: {dataset[0]}")
    # dataset.gen_train_loader()