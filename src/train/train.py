import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import HydraulicGNN
from loss import physics_loss
from dataset import WaterEPANetDataset
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
    model = HydraulicGNN()
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataset = WaterEPANetDataset(data_path, hr)
    train_loader , val_loader , test_loader = dataset.gen_train_loader()
    return model, optimizer ,train_loader
def training(model , optimizer, train_loader , epochs ,ckpt_path, log_every_epoch=10):
    model.train()
    total_loss = 0
    iterator = tqdm(range(epochs), desc=f'train_loss:{total_loss}', total= epochs)
    for i , epoch in enumerate(iterator):
        for batch in train_loader:
            pred_nodes, pred_edges = model(batch)
            # 常规重建损失
            loss_node = torch.nn.MSELoss()(pred_nodes, batch.x)
            loss_edge = torch.nn.MSELoss()(pred_edges, batch.edge_attr)
            # 物理约束损失
            loss_physics = physics_loss(pred_nodes, pred_edges, batch)
            # 总损失
            total_loss = loss_node + loss_edge + 0.3 * loss_physics
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            iterator.set_description(f'train_loss:{total_loss.item()}')
            
            if i % log_every_epoch == 0:
                # print(f"Epoch: {epoch}, Loss: {total_loss.item()}")
                torch.save(model.state_dict(), ckpt_path)
            
            
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    model, optimizer , train_loader = prepare_training_env(args.model_path, args.data, args.hours_analysis)
    training(model, optimizer, train_loader, args.epochs,args.model_path, args.log_every_epoch)