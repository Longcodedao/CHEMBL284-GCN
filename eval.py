import torch
from data import GraphData
from utils import *
from model import ChemGCN
from torch.utils.data import  DataLoader
import torch.nn as nn
import pandas as pd


def evaluate_model(weights_path, test_loader):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = ChemGCN(
        node_vec_len = 60,
        node_fea_len = 60,
        hidden_fea_len = 128,
        n_conv = 4,
        n_hidden = 2,
        n_outputs = 1,
        p_dropout = 0.3
    ).to(device)

    model.load_state_dict(torch.load(weights_path))
    loss_fn = nn.MSELoss()
    predictions = []
    targets = []

    running_loss = 0.0
    with torch.no_grad():
        model.eval()

        for batch in test_loader:
            node_mat = batch['node_mat'].to(device)
            adj_mat = batch['adj_mat'].to(device)
            output = batch['output'].to(device)

            pred = model(node_mat, adj_mat)

            # Append predictions and targets to lists
            predictions.extend(pred.cpu().numpy().flatten())  # Convert to numpy and flatten
            targets.extend(output.cpu().numpy().flatten())    # Convert to numpy and flatten

            loss = loss_fn(pred, output)
            running_loss += loss.item()

    print(f'Test Loss: {running_loss / len(test_loader)}')

    results_df = pd.DataFrame({
        "Predicted": predictions,
        "Target": targets
    })

    return results_df

if __name__ == '__main__':
    weights_path = 'weights/weights_2025-01-01_07-35-28/best_weights.pth'
    output_csv_path = 'prediction_result.csv'
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

    results_df = evaluate_model(weights_path, test_loader)
    # Write the DataFrame to a CSV file
    results_df.to_csv(output_csv_path, index=False)

    print(f"Results saved to {output_csv_path}")

