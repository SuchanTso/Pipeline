from epyt import epanet
import argparse
import os

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

def display_epanet_info(epa_net_path):
    """
    Display information about the EPANET model.
    """
    G = epanet(epa_net_path)
    
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
    G.unload()
    
if __name__ == "__main__":
    args = parse_args()
    epa_net_path = args.epa_net_path
    
    if not epa_net_path or not os.path.exists(epa_net_path):
        print("Please provide the path to the EPANET file using the -p or --epa_net_path argument.")
    else:
        print(f"Processing EPANET file: {epa_net_path}")
        display_epanet_info(epa_net_path)