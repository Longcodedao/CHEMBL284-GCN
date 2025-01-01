import torch
import torch.optim as optim
from data import GraphData
from utils import *
from model import ChemGCN
from torch.utils.data import Dataset, DataLoader
import os 
import datetime
import torch.nn as nn


def main():
    dataset_path = 'CHEMBL284_final.csv'
    node_vec_len = 60
    max_atoms = 200
    batch_size = 32

    graph_data = GraphData(dataset_path, node_vec_len, max_atoms)

    train_data, val_data, test_data = split_dataset(graph_data.graphs)

    # Create DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a directory with a timestamp to save the best weights
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = f"weights/weights_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    model = ChemGCN(
        node_vec_len = node_vec_len,
        node_fea_len = 60,
        hidden_fea_len = 128,
        n_conv = 4,
        n_hidden = 2,
        n_outputs = 1,
        p_dropout = 0.3
    ).to(device)
    
    learning_rate = 0.01
    num_epochs = 10
    clip_value = 1.0
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10,
                                                            eta_min=1e-6)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')  # Initialize with a high value

    print("Training GCN for CHEMBL Dataset")
    for epoch in range(num_epochs):
        train_loss = train_batch(train_loader, model, optimizer, 
                                 loss_fn, scheduler, device = device, clip_value = clip_value)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss}')

        eval_loss = evaluate_batch(val_loader, model, loss_fn, device)

        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_weights.pth'))
            print(f'Epoch {epoch+1}, Validation Loss: {eval_loss} (New Best)\n')
        else:
            print(f'Epoch {epoch+1}, Validation Loss: {eval_loss}\n')



if __name__ == '__main__':
    main()