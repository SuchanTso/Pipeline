from epyt import epanet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import random
import math
from .normalizer import ZScoreNormalizer ,LogZScoreNormalizer
from .Utils import * 

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
    def __init__(self, epa_net_path, hrs=72):
        print(f"Initializing EPANET model from: {epa_net_path}")
        self.epa_net_path = epa_net_path
        self.hrs = hrs
        self.G = epanet(epa_net_path)
        self.G.setTimeSimulationDuration(hrs * 3600)
        self.R = self.G.getComputedHydraulicTimeSeries()
        self.time_ranges = self.R.Flow.shape[0]
        print(f"EPANET simulation complete for {self.time_ranges} timesteps.")

    def get_raw_data(self):
        """
        返回所有原始数据的字典，格式为numpy数组或列表。
        """
        node_count = self.G.getNodeCount()
        reservoir_indices = [index - 1 for index in self.G.getNodeReservoirIndex()]
        node_type = [1 if i in reservoir_indices else 0 for i in range(node_count)]
        
        edge_index_list = []
        for pipe_id in self.G.getLinkIndex():
            start_node, end_node = self.G.getLinkNodesIndex(pipe_id)
            edge_index_list.append([start_node - 1, end_node - 1])
        
        raw_data = {
            'pressures': self.R.Pressure,  # [T, N]
            'flows': self.R.Flow,          # [T, E]
            'node_elevations': np.array(self.G.getNodeElevations()), # [N]
            'node_demands': np.array(self.G.getNodeBaseDemands()[1]), # [N]
            'node_type': np.array(node_type), # [N]
            'reservoir_indices': np.array(reservoir_indices), # [num_reservoirs]
            'link_diameters': np.array(self.G.getLinkDiameter()), # [E]
            'link_lengths': np.array(self.G.getLinkLength()),   # [E]
            'link_roughnesses': np.array(self.G.getLinkRoughnessCoeff()), # [E]
            'edge_index_directed': np.array(edge_index_list).T # [2, E]
        }
        return raw_data

    def destroy(self):
        self.G.unload()

# class EpytHelper:
#     # let's all use index when refering to nodes and pipes
#     # when required we shall convert to ID by retaining the interface of epanet
#     def __init__(self, epa_net_path, hrs=72):
#         self.epa_net_path = epa_net_path
#         self.hrs = hrs
#         self.G = epanet(epa_net_path)
#         self.G.setTimeSimulationDuration(hrs * 3600)
#         self.R = self.G.getComputedHydraulicTimeSeries()
#         self.time_ranges = self.R.Flow.shape[0]

#     def get_node_pressures(self):
#         return self.R.Pressure  # 形状: (时间步, 节点数)

#     def get_pipe_flows(self):
#         return self.R.Flow  # 形状: (时间步, 管道数)
    
#     def get_node_head(self):
#         return self.R.Head
    
#     def gen_node_static_features(self , mask_num=0):
#         node_count = self.G.getNodeCount()
#         node_elevation = self.G.getNodeElevations()  # 向量: [num_nodes] # self.get_node_head()[timestep] head indicates the pressure
#         reservoirIdx =[index - 1 for index in self.G.getNodeReservoirIndex()]  # 获取水库节点索引
#         masked_index = select_random_indices(node_count , mask_num , reservoirIdx)
#         node_type = [1 if i in reservoirIdx else 0 for i in range(node_count)]  # [0,1] 0:非水库节点, 1:水库节点
#         # print(f"masked_index: {masked_index}")
#         return node_elevation , self.G.getNodeBaseDemands()[1] , node_type , masked_index , reservoirIdx
    
#     def gen_node_masked_pressure(self , timestep ,node_count, masked_index=None):
#         # print(f"masked_index: {masked_index}")
#         node_pressure = self.get_node_pressures()[timestep].copy()
#         mask_flag = torch.zeros(node_count, dtype=torch.float)
#         for i in masked_index:
#             node_pressure[i] = 0
#             mask_flag[i] = 1
#         # node_pressure = torch.tensor([self.get_node_pressures()[timestep] if i is not in masked_index else 0 for i in node_count], dtype=torch.float).reshape(-1,1)
#         node_pressure = torch.tensor(node_pressure, dtype=torch.float)
#         # print(node_pressure)
#         return node_pressure.detach().clone() , mask_flag.reshape(-1,1)  # 返回节点压力和掩码标志，形状为 [num_nodes, 1]
    
#     def gen_edge_masked_flow(self, timestep , linkCount , masked_index):
#         edge_flow = self.get_pipe_flows()[timestep].copy()
#         mask_flag = torch.zeros(linkCount, dtype=torch.float)
#         for i in masked_index:
#             edge_flow[i] = 0
#             mask_flag[i] = 1
#         edge_flow = torch.tensor(edge_flow, dtype=torch.float).reshape(-1,1)
#         return edge_flow.detach().clone() , mask_flag.reshape(-1,1)
    
#     def gen_edge_features(self , mask_num = 0):
#         edge_index = []
#         diameters = []
#         lengths = []
#         roughnesses = []
#         link_count = self.G.getLinkCount()
#         for pipe_id in self.G.getLinkIndex():
#             # pipe attributes
#             start_node , end_node = self.G.getLinkNodesIndex(pipe_id)# node index start from 1 so out of index
#             edge_index.append([start_node - 1, end_node - 1])
#             diameters.append(float(self.G.getLinkDiameter(pipe_id)))
#             lengths.append(float(self.G.getLinkLength(pipe_id)))
#             roughnesses.append(float(self.G.getLinkRoughnessCoeff(pipe_id)))
#         edge_attr = np.stack([diameters, lengths, roughnesses], axis=1)
#         edge_index = torch.tensor(edge_index, dtype=torch.long).T  # 形状: [2, num_edges]
#         masked_index = select_random_indices(link_count,mask_num,[])
#         return edge_index, edge_attr , masked_index

#     # def create_graph_data(self,hrs , normalizer=None , mask_ratio=0.3 , pipe_mask_ratio=0.3):
#     #     # 3 , 4 , 5 , 6 , 7 , 8 , 1 , 2
#     #     # 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7
#     #     # 获取当前时间步的数据
#     #     graph_data_list = []
#     #     mask_num = math.floor(mask_ratio * self.G.getNodeCount())
#     #     pipe_mask_num = math.floor(pipe_mask_ratio * self.G.getLinkCount())
#     #     print(f"Masking {mask_num} nodes out of {self.G.getNodeCount()} nodes")
#     #     print(f"Masking {pipe_mask_num} pipes out of {self.G.getLinkCount()} pipes")
#     #     edge_index, edge_attr , pipe_masked_index = self.gen_edge_features(mask_num=pipe_mask_num)  # 获取管道特征
#     #     node_elevation , node_demands , reservoir_type , masked_index , reservoir_index = self.gen_node_static_features(mask_num)
#     #     print(f"mask_index:{masked_index} , pipe_masked_index:{pipe_masked_index}")
#     #     for timestep in range(self.time_ranges):
#     #         graph = Data()
#     #         graph.node_type = torch.tensor(reservoir_type, dtype=torch.long) # 节点类型: [num_nodes]，0:非水库节点, 1:水库节点
#     #         graph.node_static = torch.stack((torch.tensor(node_elevation) , torch.tensor(node_demands)) , dim=1).float()
#     #         graph.masked_pressure , graph.mask_pressure_flag = self.gen_node_masked_pressure(timestep ,self.G.getNodeCount(), masked_index)  # 节点压力特征: [num_nodes, 1]，masked_pressure
#     #         # graph.x = torch.stack((torch.tensor(node_elevation) , torch.tensor(node_demands) , masked_pressure.detach().clone() , torch.tensor(reservoir_type)) , dim=1).float()  # 节点特征: [num_nodes, 4]  # [扬程, 基础需求,水库类型]
            
#     #         graph.edge_index = edge_index  # 边索引: [2, num_edges]
#     #         graph.edge_static_attr = torch.tensor(edge_attr, dtype=torch.float)  # 边静态特征: [num_edges, 3] (直径, 长度, 粗糙度)
#     #         graph.masked_flow , graph.mask_flow_flag = self.gen_edge_masked_flow(timestep , self.G.getLinkCount() ,pipe_masked_index)  # 管道流量特征: [num_edges, 1]，masked_flow
#     #         # graph.edge_attr = torch.cat((torch.tensor(edge_attr, dtype=torch.float),masked_flow.detach().clone()),dim=1).float()  # 边特
#     #         graph.masked_node_index = torch.tensor(masked_index, dtype=torch.long)  # 节点掩码索引
#     #         graph.masked_pipe_index = torch.tensor(pipe_masked_index, dtype=torch.long)
            
#     #         graph.y_node = torch.tensor(self.get_node_pressures()[timestep], dtype=torch.float).reshape(-1,1)  # 节点压力特征: [num_nodes, 1]
#     #         graph.y_edge = torch.tensor(self.get_pipe_flows()[timestep] , dtype=torch.float).reshape(-1,1)  # 管道流量特征: [num_edges, 1]
            
#     #         graph_data_list.append(graph)
#     #     return graph_data_list
    
#     def create_graph_data(self, hrs, mask_ratio=0.3, pipe_mask_ratio=0.3):
#         # ... (前面的代码保持不变) ...
#         graph_data_list = []
#         mask_num = math.floor(mask_ratio * self.G.getNodeCount())
#         pipe_mask_num = math.floor(pipe_mask_ratio * self.G.getLinkCount())
#         print(f"Masking {mask_num} nodes out of {self.G.getNodeCount()} nodes")
#         print(f"Masking {pipe_mask_num} pipes out of {self.G.getLinkCount()} pipes")
#         self.mask_ratio = mask_ratio
#         self.pipe_mask_ratio = pipe_mask_ratio
#         node_num = self.G.getNodeCount()
#         link_num = self.G.getLinkCount()

#         # --- 核心修改部分开始 ---
#         # 1. 获取基础拓扑信息
#         # 注意：你的gen_edge_features已经不是无向的了，你需要自己把它变成无向图
#         # 或者，我们在这里手动创建无向图的edge_index
#         edge_index_directed, edge_attr, pipe_masked_index = self.gen_edge_features(mask_num=pipe_mask_num)
        
#         # 创建双向边
#         row, col = edge_index_directed
#         edge_index_undirected = torch.cat([torch.stack([row, col]), torch.stack([col, row])], dim=1)
#         # 对应地，边特征也需要复制
#         edge_attr_undirected = np.concatenate([edge_attr, edge_attr], axis=0)
        
#         # 创建一个临时的拓扑Data对象用于预处理
#         graph_topology = Data(
#             edge_index=edge_index_undirected,
#             num_nodes=self.G.getNodeCount(),
#             num_edges=edge_index_undirected.shape[1]
#         )
        
#         # 2. 一次性计算结构信息
#         structural_data = preprocess_for_graph_transformer(graph_topology)
#         # --- 核心修改部分结束 ---

#         node_elevation, node_demands, reservoir_type, masked_index, reservoir_index = self.gen_node_static_features(mask_num)
#         print(f"mask_index:{masked_index} , pipe_masked_index:{pipe_masked_index}")
        
#         graph_data_list = []
#         for timestep in range(self.time_ranges):
#             graph = Data()
#             graph.num_nodes = node_num
#             graph.num_edges = link_num
#             # --- 赋值节点和边的动态、静态特征 (这部分逻辑不变) ---
#             graph.node_type = torch.tensor(reservoir_type, dtype=torch.long)
#             graph.node_static = torch.stack((torch.tensor(node_elevation), torch.tensor(node_demands)), dim=1).float()
#             # graph.masked_pressure, graph.mask_pressure_flag = self.gen_node_masked_pressure(timestep, self.G.getNodeCount(), masked_index)
            
#             # 使用双向的edge_index和edge_attr
#             graph.edge_index = edge_index_undirected
#             graph.edge_static_attr = torch.tensor(edge_attr_undirected, dtype=torch.float)
            
#             # 注意：masked_flow的处理需要匹配双向边。
#             # 一个简单的处理方式是假设反向流等于正向流的相反数。
#             # 这里为了简化，我们先假设正反向流特征相同（比如只用绝对值），具体物理意义可再调整。
#             # masked_flow_directed, mask_flow_flag_directed = self.gen_edge_masked_flow(timestep, self.G.getLinkCount(), pipe_masked_index)
#             # graph.masked_flow = torch.cat([masked_flow_directed, masked_flow_directed], dim=0) # 复制特征
#             # graph.mask_flow_flag = torch.cat([mask_flow_flag_directed, mask_flow_flag_directed], dim=0) # 复制flag
#             graph.node_pressure = torch.tensor(self.get_node_pressures()[timestep], dtype=torch.float).reshape(-1, 1)
#             graph.pipe_flow = torch.tensor(self.get_pipe_flows()[timestep], dtype=torch.float).reshape(-1, 1)
            
#             graph.y_node = torch.tensor(self.get_node_pressures()[timestep], dtype=torch.float).reshape(-1, 1)
            
#             y_edge_directed = torch.tensor(self.get_pipe_flows()[timestep], dtype=torch.float).reshape(-1, 1)
#             graph.y_edge = torch.cat([y_edge_directed, -y_edge_directed], dim=0) # 假设反向流量为负

#             # --- 将计算好的结构信息添加到每个Data对象中 ---
#             graph.degree_encoding = structural_data['degree_encoding']
#             graph.spd_matrix = structural_data['spd_matrix']
#             graph.edge_map = structural_data['edge_map']
            
#             # 掩码索引也需要保留，但注意它们是针对原始有向图的
#             # graph.masked_node_index = torch.tensor(masked_index, dtype=torch.long)
#             # graph.masked_pipe_index = torch.tensor(pipe_masked_index, dtype=torch.long)
                
#             graph_data_list.append(graph)
            
#         return graph_data_list
    
#     def create_raw_graph_data_list(self):
#         """
#         新方法：创建包含所有原始物理值的Data对象列表，并进行一次性的拓扑预计算。
#         """
#         print("Creating raw graph data list...")
        
#         # --- 1. 获取一次性的静态和拓扑信息 ---
#         node_count = self.G.getNodeCount()
#         link_count = self.G.getLinkCount()
        
#         node_elevation, node_demands, node_type, _, reservoir_indices = self.gen_node_static_features()
#         node_static = torch.stack((torch.tensor(node_elevation), torch.tensor(node_demands)), dim=1).float()
#         node_type = torch.tensor(node_type, dtype=torch.long)
        
#         edge_index_directed, edge_attr_static_directed, _ = self.gen_edge_features()
#         row, col = edge_index_directed
#         edge_index = torch.cat([torch.stack([row, col]), torch.stack([col, row])], dim=1)
#         edge_static_attr = torch.tensor(np.concatenate([edge_attr_static_directed, edge_attr_static_directed], axis=0), dtype=torch.float)

#         # --- 2. 一次性计算图Transformer所需的结构信息 ---
#         graph_topology = Data(edge_index=edge_index, num_nodes=node_count)
#         structural_data = preprocess_for_graph_transformer(graph_topology)

#         # --- 3. 循环创建每个时间步的Data对象 ---
#         graph_data_list = []
#         for t in range(self.time_ranges):
#             data = Data()
            
#             # 静态节点特征
#             data.node_static = node_static
#             data.node_type = node_type
#             data.reservoir_index = torch.tensor(reservoir_indices, dtype=torch.long)
            
#             # 动态节点特征 (原始物理值)
#             data.pressure = torch.tensor(self.get_node_pressures()[t], dtype=torch.float).view(-1, 1)

#             # 静态边特征和拓扑
#             data.edge_index = edge_index
#             data.edge_static_attr = edge_static_attr
            
#             # 动态边特征 (原始物理值)
#             flow_directed = torch.tensor(self.get_pipe_flows()[t], dtype=torch.float).view(-1, 1)
#             data.flow = torch.cat([flow_directed, -flow_directed], dim=0) # 假设反向流为负
            
#             # 附加结构信息
#             data.degree_encoding = structural_data['degree_encoding']
#             data.spd_matrix = structural_data['spd_matrix']
#             data.edge_map = structural_data['edge_map']
            
#             # 确保num_nodes等属性存在
#             data.num_nodes = node_count

#             graph_data_list.append(data)
            
#         print("Raw graph data list created successfully.")
#         return graph_data_list
    
#     def destroy(self):
#         """释放EPANET资源"""
#         self.G.unload()

# 创建整个时间序列的数据集
# graph_data_list = [create_graph_data(t) for t in range(x)]
# epytNet.create_graph_data(0)  # 获取第一个时间步的图数据示例



    
