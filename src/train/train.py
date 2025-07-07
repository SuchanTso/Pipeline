import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import GNN_ChebConv , TGCN_PyG , TGCN_MessageCoupling
from loss import physics_loss
from dataset import WaterEPANetDataset , GraphNormalizer , ZScoreNormalizer
import os
import argparse
from tqdm import tqdm
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        nargs="?",
        help="path to model",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        nargs="?",
        help="path to epanet data",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train",
    )
    parser.add_argument(
        "-l",
        "--log_every_epoch",
        type=int,
        default=10,
        help="log every n epochs",
    )
    parser.add_argument(
        "--hours_analysis",
        type=int,
        default=72,
        help="hours to analysis",
    )
    
    
    return parser
def prepare_training_env(ckpt_path , data_path , hr):
    model = TGCN_MessageCoupling(in_feats=4, gcn_hidden=32, gru_hidden=32,edge_hidden=32, out_node_feats=1,out_edge_feats=1)
    # model = TGCN_PyG(in_feats=4, gcn_hidden=32, gru_hidden=32, out_feats=1)
    # model = GNN_ChebConv(hid_channels=32, edge_features=3, node_features=3, edge_channels=32, dropout_rate=0.2, CC_K=2,
    #                      emb_aggr='max', depth=2, normalize=True)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # normalizer = GraphNormalizer()
    x_norm = ZScoreNormalizer()
    y_node_normalizer = ZScoreNormalizer()
    y_edge_normalizer = ZScoreNormalizer()
    dataset = WaterEPANetDataset(data_path, hr ,x_normalizer=x_norm , y_node_normalizer=y_node_normalizer , y_edge_normalizer=y_edge_normalizer,masked_ratio=0.5 , window_size=8)
    # print(f"dataset load:{dataset[0]}")
    train_loader , val_loader , test_loader = dataset.gen_train_loader(batch_size=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, optimizer ,train_loader , val_loader , test_loader , y_node_normalizer , y_edge_normalizer , device
def training(model , optimizer, train_loader , epochs , device ,ckpt_path, log_every_epoch=10):
    model.train()
    total_loss = 0
    iterator = tqdm(range(epochs), desc=f'train_loss:{total_loss}', total= epochs)
    for i , epoch in enumerate(iterator):
        for batch in train_loader:
            
            x_seq, edge_index, edge_attr, y_node, y_edge = batch  # x_seq: [B, T, N, F]
            x_seq = x_seq[0].to(device)          # [T, N, F]
            edge_index = edge_index[0].to(device)  # [2, E]
            y_node = y_node[0].to(device)        # [N, 1]
            y_edge = y_edge[0].to(device)        # [E, 1]
            edge_attr = edge_attr[0].to(device)

            # 调用模型
            pred_nodes , pred_edges = model(x_seq, edge_index , edge_attr)  # pred_node: [B, N, 1]
            # pred_nodes = model(batch)
            # 常规重建损失
            loss_node = torch.nn.MSELoss()(pred_nodes, y_node) + torch.nn.MSELoss()(pred_edges, y_edge)
            # loss_edge = torch.nn.MSELoss()(pred_edges, batch.edge_attr)
            # 物理约束损失
            # loss_physics = physics_loss(pred_nodes, pred_edges, batch)
            # 总损失
            total_loss = loss_node
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if i % log_every_epoch == 0:
                # print(f"pred: {pred_nodes[:5]}, true: {batch.y_node[:5]}")
                iterator.set_description(f'train_loss:{total_loss.item():.2f}')
                torch.save(model.state_dict(), ckpt_path)
                
def eval(model , test_loader , pressure_norm , flow_norm , device):
    model.eval()
    total_loss = 0
    np.set_printoptions(precision=10, suppress=True)
    with torch.no_grad():
        for batch in test_loader:
            x_seq, edge_index, edge_attr, y_node, y_edge = batch  # x_seq: [B, T, N, F]
            x_seq = x_seq[0].to(device)          # [T, N, F]
            edge_index = edge_index[0].to(device)  # [2, E]
            y_node = y_node[0].to(device)        # [N, 1]
            y_edge = y_edge[0].to(device)        # [E, 1]
            edge_attr = edge_attr[0].to(device)

            # 调用模型
            pred_nodes , pred_edges = model(x_seq, edge_index , edge_attr)  # pred_node: [B, N, 1]
            inverse_pred_nodes = pressure_norm.inverse_transform(pred_nodes.cpu())
            inverse_real_nodes = pressure_norm.inverse_transform(y_node.cpu())
            print(f"pred_nodes: {inverse_pred_nodes.numpy()}")
            print(f"real_nodes: {inverse_real_nodes.numpy()}")
            inverse_pred_edge = flow_norm.inverse_transform(pred_edges.cpu())
            inverse_real_edge = flow_norm.inverse_transform(y_edge.cpu())
            print(f"pred_edge: {inverse_pred_edge.numpy()}")
            print(f"real_edge: {inverse_real_edge.numpy()}")
            loss_node = torch.nn.MSELoss()(pred_nodes, y_node) + torch.nn.MSELoss()(pred_edges, y_edge)
            total_loss += loss_node.item()
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss
            
            
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    model, optimizer , train_loader , val_loader , test_loader , pressure_norm , flow_norm , device = prepare_training_env(args.model_path, args.data, args.hours_analysis)
    training(model, optimizer, train_loader, args.epochs , device ,args.model_path , args.log_every_epoch )
    eval(model, val_loader , pressure_norm , flow_norm ,device)