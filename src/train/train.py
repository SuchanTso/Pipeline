import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import HydraulicGNN, GNN_ChebConv
from loss import physics_loss
from dataset import WaterEPANetDataset , GraphNormalizer
import os
import argparse
from tqdm import tqdm


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
    model = GNN_ChebConv(hid_channels=32, edge_features=3, node_features=3, edge_channels=32, dropout_rate=0.2, CC_K=2,
                         emb_aggr='max', depth=2, normalize=True)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    normalizer = GraphNormalizer()
    dataset = WaterEPANetDataset(data_path, hr , normalizer=normalizer)
    print(f"dataset load:{dataset[0]}")
    train_loader , val_loader , test_loader = dataset.gen_train_loader(batch_size=8)
    return model, optimizer ,train_loader , val_loader , test_loader , normalizer
def training(model , optimizer, train_loader , epochs ,ckpt_path, log_every_epoch=10):
    model.train()
    total_loss = 0
    iterator = tqdm(range(epochs), desc=f'train_loss:{total_loss}', total= epochs)
    for i , epoch in enumerate(iterator):
        for batch in train_loader:
            pred_nodes = model(batch)
            # 常规重建损失
            loss_node = torch.nn.MSELoss()(pred_nodes, batch.y_node)
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
                
def eval(model , test_loader , normalizer):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            pred_nodes = model(batch)
            loss_node = torch.nn.MSELoss()(pred_nodes, batch.y_node)
            pred_pressure = normalizer.inverse_transform_array(pred_nodes, 'pressure')
            label_pressure = normalizer.inverse_transform_array(batch.y_node, 'pressure')
            print(f"pred: {pred_pressure}, true: {label_pressure}")
            total_loss += loss_node.item()
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss
            
            
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    model, optimizer , train_loader , val_loader , test_loader , normalizer = prepare_training_env(args.model_path, args.data, args.hours_analysis)
    training(model, optimizer, train_loader, args.epochs,args.model_path , args.log_every_epoch )
    eval(model, val_loader , normalizer)