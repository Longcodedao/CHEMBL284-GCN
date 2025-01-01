from rdkit import Chem
from rdkit.Chem import Draw, rdmolops
from rdkit.Chem import rdDistGeom as molDG

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch

class Graph:
    def __init__(
            self, molecule_smiles,
            node_vec_len,
            max_atoms = None
    ):
        self.smiles = molecule_smiles
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms

        # Call helper function to convert SMILES to RDKit mol
        self.smiles_to_mol()

        if self.mol is not None:
            self.smiles_to_graph()

    def smiles_to_mol(self):
        mol = Chem.MolFromSmiles(self.smiles)

        if mol is None:
            self.mol = None
            return

        self.mol = Chem.AddHs(mol)


    def smiles_to_graph(self):
        atoms = self.mol.GetAtoms()

        # If max_atoms is not provided, max_atoms is equal to maximum number
        # of atoms in this molecule.
        if self.max_atoms is None:
            n_atoms = len(list(atoms))
        else:
            n_atoms = self.max_atoms

        node_mat = np.zeros((n_atoms, self.node_vec_len))
        for atom in atoms:
            atom_index = atom.GetIdx()
            atom_no = atom.GetAtomicNum()

            # Assign to node matrix (Use One-Hot Encoding)
            node_mat[atom_index, atom_no] = 1

        # Get adjacency matrix using RDKit
        adj_mat = rdmolops.GetAdjacencyMatrix(self.mol)
        self.std_adj_mat = np.copy(adj_mat)

        # Get distance matrix using RDKit
        dist_mat = molDG.GetMoleculeBoundsMatrix(self.mol)
        dist_mat[dist_mat == 0.] = 1

        # Get modified adjacency matrix by the Convalent Bonds
        adj_mat = adj_mat * (1 / dist_mat)

        # Pad the adjacency matrix with 0s
        dim_add = n_atoms - adj_mat.shape[0]
        adj_mat = np.pad(
            adj_mat, pad_width=((0, dim_add), (0, dim_add)), mode="constant"
        )

        # Add an identity matrix to adjacency matrix
        # This will make an atom its own neighbor
        adj_mat = adj_mat + np.eye(n_atoms)

        # Save both matrices
        self.node_mat = node_mat
        self.adj_mat = adj_mat


# GraphData class inheriting from the Dataset class in PyTorch.
class GraphData(Dataset):
    def __init__(self, dataset_path, node_vec_len, max_atoms):

        # Save attributes
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms

        # Open dataset file
        df = pd.read_csv(dataset_path)

        # Create lists
        self.indices = df.index.to_list()
        self.smiles = df["smiles"].to_list()
        self.outputs = df["pIC50"].to_list()

        # Load the dataset
        self.data = pd.read_csv(dataset_path)

        # Create a list of Graph objects
        self.graphs = []
        for index, row in self.data.iterrows():
            smiles = row['smiles']
            graph = Graph(molecule_smiles=smiles, node_vec_len=node_vec_len, max_atoms=max_atoms)

            # Get matrices
            node_mat = torch.Tensor(graph.node_mat)
            adj_mat = torch.Tensor(graph.adj_mat)

            # Get output
            output = torch.Tensor([self.outputs[index]])

            self.graphs.append({
                "node_mat": node_mat,
                "adj_mat": adj_mat,
                "output": output  # Replace 'target' with the actual label column name
            })

    # Length of dataset
    def __len__(self):
        return len(self.indices)

    # Get item
    def __getitem__(self, idx):
        return self.graphs[idx]

if __name__ == '__main__':
    dataset_file = pd.read_csv('CHEMBL284_final.csv')

    mols = [Chem.MolFromSmiles(smiles) for smiles in dataset_file['smiles']]

    # Select a single molecule (e.g., the first one)
    single_smiles = dataset_file['smiles'][0]
    single_molecule = Chem.MolFromSmiles(single_smiles)

    # Draw the molecule
    single_img = Draw.MolToImage(single_molecule, size=(300, 300),
                                legend=dataset_file['molecule_id'][0])

    # Display the image
    plt.imshow(single_img)
    plt.axis('off')
    plt.show()