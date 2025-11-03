from epyt import epanet
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--epa_net_path",
        type=str,
        help="path of the epanet file",
    )
    
    return parser.parse_args()

def display_epanet_info(G):
    """
    Display information about the EPANET model.
    """
    # G = epanet(epa_net_path)
    
    # Display node and pipe counts
    node_count = G.getNodeCount()
    pipe_count = G.getLinkCount()
    
    print(f"Node count: {node_count}")
    print(f"Pipe count: {pipe_count}")
    
    ReservoirIdx = G.getNodeReservoirIndex()
    print(f"Reservoir node count : {len(ReservoirIdx)}")
    
    TankIdx = G.getNodeTankIndex()
    print(f"Tank node count : {len(TankIdx)}")
    
    PumpCnt = G.getLinkPumpCount()
    print(f"Pump count: {PumpCnt}")
    
    # Display node names and IDs
    # node_names = G.getNodeNameID(range(node_count))
    # print(f"Node names and IDs: {node_names}")
    # G.unload()
    
def display_node_demand(G):
    """
    Display demand information for each node in the EPANET model.
    """
    node_count = G.getNodeCount()
    
    for node_idx in range(node_count):
        demand = G.getNodeDemand(node_idx)
        print(f"Node {node_idx} Demand: {demand}")
        
def plot_epanet_results(d, node_index: int = 5, link_index: int = 5, duration_hours: int = 72):
    """
    加载EPANET模型，模拟指定时长，并绘制特定节点压力和管道流量的时间序列曲线。

    Args:
        inp_file (str): EPANET模型文件 (.inp) 的路径。
        node_index (int, optional): 要绘制压力曲线的节点索引 (从1开始). 默认为 5.
        link_index (int, optional): 要绘制流量曲线的管道索引 (从1开始). 默认为 5.
        duration_hours (int, optional): 总仿真时长 (小时). 默认为 72.
    """

    print(f"正在加载模型")
    
    # 使用 with 语句确保 epanet 对象被正确关闭
    
    # --- 1. 设置仿真时长 ---
    duration_seconds = duration_hours * 3600
    d.setTimeSimulationDuration(duration_seconds)
    print(f"仿真时长设置为: {duration_hours} 小时 ({duration_seconds} 秒)")

    # --- 2. 获取节点和管道的ID，用于绘图标签 ---
    try:
        node_id = d.getNodeNameID(node_index)
        link_id = d.getLinkNameID(link_index)
    except Exception as e:
        print(f"错误: 无法获取节点/管道ID。请检查索引是否有效。")
        print(f"epyt 错误信息: {e}")
        return
        
    # 获取单位，让图表更清晰
    head_units = d.units.NodePressureUnits
    flow_units = d.units.LinkFlowUnits

    # --- 3. 运行仿真并收集数据 ---
    print("开始步进式液压仿真...")
    
    # 初始化用于存储结果的列表
    time_steps_seconds = []
    node_pressures = []
    link_flows = []

    d.openHydraulicAnalysis()
    d.initializeHydraulicAnalysis()
    
    R = d.getComputedHydraulicTimeSeries()
    node_pressures = R.Pressure.transpose()[node_index].transpose()
    link_flows = R.Flow.transpose()[link_index].transpose()
    node_dmand = R.Demand.transpose()[node_index].transpose()
    time_steps_seconds = (np.arange(R.Flow.shape[0]))
    
    print(f"node_pressures: {R.Pressure.shape},time_steps_seconds: {time_steps_seconds.shape}")
    print("仿真完成。")

    # --- 4. 数据后处理和绘图 ---

    # 将时间从秒转换为小时，方便绘图
    time_in_hours = time_steps_seconds
    print(f"共收集到 {time_in_hours} 个flow。")

    print(f"共收集到 {len(node_pressures)} 个pressure。")
    
    
    print(f"node demands shape: {R.Demand.shape}")

    # 创建两个子图，共享X轴
    fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'EPANET simulation', fontsize=16)

    # 绘制节点压力曲线
    ax1.plot(time_in_hours, node_pressures, 'b-o', markersize=4, label=f'node: {node_id} (index: {node_index})')
    ax1.set_title(f'node pressure curve')
    ax1.set_ylabel(f'pressure ({head_units})')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # 绘制管道流量曲线
    ax2.plot(time_in_hours, link_flows, 'r-s', markersize=4, label=f'link {link_id} (index {link_index})')
    ax2.set_title(f'flow curve')
    ax2.set_ylabel(f'flow ({flow_units})')
    ax2.set_xlabel('time (hours)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    # 绘制需水量曲线
    ax3.plot(time_in_hours, node_dmand, 'r-s', markersize=4, label=f'node {node_id} (index {node_index})')
    ax3.set_title(f'demand curve')
    ax3.set_ylabel(f'demand (L/s)')
    ax3.set_xlabel('time (hour)')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend()
    
    # 优化布局并显示图形
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局以适应主标题
    plt.savefig(f"figures/epanet_results.png")
    
    
if __name__ == "__main__":
    args = parse_args()
    epa_net_path = args.epa_net_path
    
    if not epa_net_path or not os.path.exists(epa_net_path):
        print("Please provide the path to the EPANET file using the -p or --epa_net_path argument.")
    else:
        print(f"Processing EPANET file: {epa_net_path}")
        epa = epanet(epa_net_path)
        display_epanet_info(epa)
        # display_node_demand(epa_net_path)
        plot_epanet_results(epa)
        epa.unload()