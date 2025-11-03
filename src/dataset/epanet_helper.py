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
# class EpytHelper:
#     def __init__(self, epa_net_path, hrs=72):
#         print(f"Initializing EPANET model from: {epa_net_path}")
#         self.epa_net_path = epa_net_path
#         self.hrs = hrs
#         self.G = epanet(epa_net_path)
#         self.G.setTimeSimulationDuration(hrs * 3600)
#         self.R = self.G.getComputedHydraulicTimeSeries()
#         self.time_ranges = self.R.Flow.shape[0]
#         print(f"EPANET simulation complete for {self.time_ranges} timesteps.")

#     def get_raw_data(self):
#         """
#         返回所有原始数据的字典，格式为numpy数组或列表。
#         """
#         node_count = self.G.getNodeCount()
#         reservoir_indices = [index - 1 for index in self.G.getNodeReservoirIndex()]
#         node_type = [1 if i in reservoir_indices else 0 for i in range(node_count)]
        
#         edge_index_list = []
#         for pipe_id in self.G.getLinkIndex():
#             start_node, end_node = self.G.getLinkNodesIndex(pipe_id)
#             edge_index_list.append([start_node - 1, end_node - 1])
        
#         raw_data = {
#             'pressures': self.R.Pressure,  # [T, N]
#             'flows': self.R.Flow,          # [T, E]
#             'node_elevations': np.array(self.G.getNodeElevations()), # [N]
#             'node_demands': np.array(self.G.getNodeBaseDemands()[1]), # [N]
#             'demand_real': np.array(self.R.Demand), # [T, N]
#             'node_type': np.array(node_type), # [N]
#             'reservoir_indices': np.array(reservoir_indices), # [num_reservoirs]
#             'link_diameters': np.array(self.G.getLinkDiameter()), # [E]
#             'link_lengths': np.array(self.G.getLinkLength()),   # [E]
#             'link_roughnesses': np.array(self.G.getLinkRoughnessCoeff()), # [E]
#             'edge_index_directed': np.array(edge_index_list).T # [2, E]
#         }
#         return raw_data

#     def destroy(self):
#         self.G.unload()

class EpytHelper:
    def __init__(self, epa_net_path, hrs=72):
        print(f"Initializing EPANET model from: {epa_net_path}")
        self.epa_net_path = epa_net_path
        self.hrs = hrs
        try:
            self.G = epanet(epa_net_path)
        except Exception as e:
            print(f"Error loading EPANET file {epa_net_path}: {e}")
            raise

        # --- 单位检测与转换系数设定 ---
        self._detect_and_set_conversion_factors()
        
        # 运行仿真
        self.G.setTimeSimulationDuration(hrs * 3600)
        self.R = self.G.getComputedHydraulicTimeSeries()
        self.time_ranges = self.R.Flow.shape[0]
        print(f"EPANET simulation complete for {self.time_ranges} timesteps.")

    def _detect_and_set_conversion_factors(self):
        """
        检测 .inp 文件中的单位制，并设置相应的SI转换系数。
        目标单位制: 流量(m³/s), 压力(m), 长度/直径(m)。
        """
        # 获取流量单位字符串，并转换为大写以进行不区分大小写的比较
        flow_units = self.G.getFlowUnits().upper()
        print(f"Detected flow units: {flow_units}")

        # 美制单位 (US Customary)
        us_units = ['CFS', 'GPM', 'MGD', 'IMGD', 'AFD']
        
        self.conversion = {}
        if flow_units in us_units:
            print("System identified as US Customary. Applying US to SI conversion factors.")
            
            # 流量转换
            if flow_units == 'GPM':
                self.conversion['flow'] = 6.30902e-5  # GPM to m³/s
            elif flow_units == 'CFS':
                self.conversion['flow'] = 0.0283168   # CFS to m³/s
            # ...可以为 MGD, IMGD, AFD 添加更多转换 ...
            else:
                raise ValueError(f"Unsupported US Customary flow unit: {flow_units}")

            # 长度、压力等单位
            self.conversion['pressure_to_head'] = 0.70325 # psi to m H₂O
            self.conversion['length'] = 0.3048           # ft to m
            self.conversion['diameter'] = 0.0254         # in to m
            
        else: # 假设其他都是SI单位或接近SI单位
            print("System identified as SI-based. Applying SI-based to standard SI conversion factors.")

            # 流量转换
            if flow_units in ['LPS']:
                self.conversion['flow'] = 0.001       # LPS to m³/s
            elif flow_units in ['LPM']:
                self.conversion['flow'] = 1.66667e-5  # LPM to m³/s
            elif flow_units in ['CMH', 'M3/HR']:
                self.conversion['flow'] = 1 / 3600    # m³/h to m³/s
            elif flow_units in ['CMD']:
                self.conversion['flow'] = 1 / (3600 * 24) # m³/d to m³/s
            elif flow_units in ['MLD']:
                self.conversion['flow'] = 1000 / (3600 * 24) # MLD to m³/s
            else:
                 raise ValueError(f"Unsupported SI-based flow unit: {flow_units}")
            
            # 长度、压力等单位
            # 在SI制下，压力通常直接是米水头 (m)，但要检查[OPTIONS]中的Pressure设置
            # 我们假设 epynet 总是返回米水头
            self.conversion['pressure_to_head'] = 1.0  
            self.conversion['length'] = 1.0               # m to m
            self.conversion['diameter'] = 0.001           # mm to m
            
        # Hazen-Williams C系数无量纲，无需转换
        self.conversion['hazen_williams_c'] = 1.0
        
        print("Conversion factors to standard SI (m, s, m³, m H₂O) set:", self.conversion)

    def get_raw_data(self):
        """
        返回所有原始数据的字典，所有值都已转换为国际单位制 (SI)。
        """
        node_count = self.G.getNodeCount()
        reservoir_indices = [index - 1 for index in self.G.getNodeReservoirIndex()]
        node_type = [1 if i in reservoir_indices else 0 for i in range(node_count)]
        
        edge_index_list = []
        link_indices = self.G.getLinkIndex()
        for link_id in link_indices:
            start_node, end_node = self.G.getLinkNodesIndex(link_id)
            edge_index_list.append([start_node - 1, end_node - 1])
        
        # --- 在提取数据后立即应用转换 ---
        pressures = self.R.Pressure * self.conversion['pressure_to_head']
        flows = self.R.Flow * self.conversion['flow']
        demands = self.R.Demand * self.conversion['flow']
        
        elevations = np.array(self.G.getNodeElevations()) * self.conversion['length']
        # getNodeBaseDemands() 返回一个元组，我们需要第二个元素
        base_demands = np.array(self.G.getNodeBaseDemands()[1]) * self.conversion['flow']
        
        diameters = np.array([self.G.getLinkDiameter(i) for i in link_indices]) * self.conversion['diameter']
        lengths = np.array([self.G.getLinkLength(i) for i in link_indices]) * self.conversion['length']
        roughnesses = np.array([self.G.getLinkRoughnessCoeff(i) for i in link_indices]) # 无需转换

        raw_data = {
            # 动态数据 (SI)
            'pressures': pressures,      # [T, N], 单位: m
            'flows': flows,              # [T, E], 单位: m³/s
            'demand_real': demands,      # [T, N], 单位: m³/s
            
            # 静态节点数据 (SI)
            'node_elevations': elevations, # [N], 单位: m
            'node_demands': base_demands,    # [N], 单位: m³/s
            
            # 拓扑和元数据
            'node_type': np.array(node_type),
            'reservoir_indices': np.array(reservoir_indices),
            'edge_index_directed': np.array(edge_index_list).T,
            
            # 静态边数据 (SI)
            'link_diameters': diameters,     # [E], 单位: m
            'link_lengths': lengths,         # [E], 单位: m
            'link_roughnesses': roughnesses  # [E], 无量纲
        }
        return raw_data

    def destroy(self):
        self.G.unload()



    
