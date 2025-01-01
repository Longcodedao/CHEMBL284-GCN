import torch
import torch.nn as nn


#### CLASSES
class ConvolutionLayer(nn.Module):
    def __init__(self, node_in_len, node_out_len):
        super(ConvolutionLayer, self).__init__()
        self.linear = nn.Linear(node_in_len, node_out_len)

        # Create activation function
        self.conv_activation = nn.LeakyReLU()

    def forward(self, node_mat, adj_mat):
        n_neighbors = adj_mat.sum(dim = -1, keepdims = True)

        # Cteate identity
        self.idx_mat = torch.eye(
            adj_mat.shape[-2], adj_mat.shape[-1], device = n_neighbors.device
        )

        idx_mat = self.idx_mat.unsqueeze(0).expand(*adj_mat.shape)
        inv_degree_mat = torch.mul(idx_mat, 1 / n_neighbors)

        # Perform matrix multiplication D^(-1)AN
        node_fea = torch.bmm(inv_degree_mat, adj_mat)
        node_fea = torch.bmm(node_fea, node_mat)

        node_fea = self.linear(node_fea)
        node_fea = self.conv_activation(node_fea)
        return node_fea


# Pooling Layer in Graph Convolution 
class PoolingLayer(nn.Module):
    """
    Create a pooling layer to average node-level properties into graph-level properties
    """

    def __init__(self):
        # Call constructor of base class
        super().__init__()

    def forward(self, node_fea):
        # Pool the node matrix
        pooled_node_fea = node_fea.mean(dim=1)
        return pooled_node_fea


# Graph Convolution Network for CHEMBL Dataset 
class ChemGCN(nn.Module):
    def __init__(
            self,
            node_vec_len,
            node_fea_len,
            hidden_fea_len,
            n_conv: int,
            n_hidden: int,
            n_outputs: int,
            p_dropout: float = 0.0,
    ):
        super().__init__()

        # Initial transformation from node matrix to node features
        self.init_transform = nn.Linear(node_vec_len, node_fea_len)
        self.conv_layers = nn.ModuleList(
            [ConvolutionLayer(
                node_fea_len, node_fea_len
            ) for i in range(n_conv)]
        )

        self.pooling = PoolingLayer()
        pooled_node_fea_len = node_fea_len

        self.pooled_to_hidden = nn.Linear(pooled_node_fea_len, hidden_fea_len)

        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p_dropout)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_fea_len, hidden_fea_len)
             for i in range(n_hidden)]
        )

        self.output_layer = nn.Linear(hidden_fea_len, n_outputs)


    def forward(self, node_mat, adj_mat):
        node_fea = self.init_transform(node_mat)

        for conv_layer in self.conv_layers:
            node_fea = conv_layer(node_fea, adj_mat)

        pooled_node_fea = self.pooling(node_fea)

        hidden_node_fea = self.pooled_to_hidden(pooled_node_fea)
        hidden_node_fea = self.leakyrelu(hidden_node_fea)
        hidden_node_fea = self.dropout(hidden_node_fea)

        for hidden_layer in self.hidden_layers:
            hidden_node_fea = hidden_layer(hidden_node_fea)
            hidden_node_fea = self.leakyrelu(hidden_node_fea)
            hidden_node_fea = self.dropout(hidden_node_fea)

        output = self.output_layer(hidden_node_fea)

        return output
