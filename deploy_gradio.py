import torch
from data import Graph
from utils import *
from model import ChemGCN
import torch.nn as nn
import pandas as pd
from chembl_webresource_client.new_client import new_client
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import gradio as gr


# Fetch SMILES from ChEMBL ID
def fetch_smiles_from_chembl(chembl_id):
    molecule = new_client.molecule
    result = molecule.get(chembl_id)
    return result['molecule_structures']['canonical_smiles']

# Visualize Molecule
def plot_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=(300, 300))
    return img

# Prediction Function
def predict_pIC50(chembl_id, model, node_vec_len, max_atoms, device):
    # Fetch SMILES
    smiles = fetch_smiles_from_chembl(chembl_id)

    # Convert to graph
    graph = Graph(smiles, node_vec_len, max_atoms)
    node_mat = torch.tensor(graph.node_mat, dtype = torch.float32).unsqueeze(0)  # Add batch dimension
    adj_mat = torch.tensor(graph.adj_mat, dtype = torch.float32).unsqueeze(0)

    node_mat = node_mat.to(device)
    adj_mat = adj_mat.to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        pred = model(node_mat, adj_mat)
    
    return smiles, pred.item()

# Gradio Interface Function
def gradio_interface(chembl_id):
    smiles, pIC50 = predict_pIC50(chembl_id, model, node_vec_len, max_atoms, device)
    molecule_img = plot_molecule(smiles)

    # Determine pIC50 range label
    if pIC50 >= 7:
        potency_label = "Highly Potent"
    elif 5 <= pIC50 < 7:
        potency_label = "Moderately Potent"
    else:
        potency_label = "Weakly Potent"


    return pIC50, potency_label, molecule_img


# Model Setup
node_vec_len = 60
max_atoms = 200
weights_path = 'weights/weights_2025-01-01_07-35-28/best_weights.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChemGCN(
    node_vec_len=node_vec_len,
    node_fea_len=60,
    hidden_fea_len=128,
    n_conv=4,
    n_hidden=2,
    n_outputs=1,
    p_dropout=0.3
).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))


# Gradio App
inputs = gr.Textbox(label="ChEMBL ID", placeholder="Enter a ChEMBL ID (e.g., CHEMBL515387)")
outputs = [
    gr.Number(label="Predicted pIC50"),
    gr.Textbox(label="Potency Classification"),
    gr.Image(label="Molecule Visualization")
]

gr.Interface(
    fn=gradio_interface,
    inputs=inputs,
    outputs=outputs,
    title="Compounds Tested for Inhibition or Interaction with DPP-IV",
    description="Enter a ChEMBL ID to predict its pIC50 and view the molecular structure.",
).launch()