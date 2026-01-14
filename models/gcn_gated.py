# models/gcn_gated.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp


# GCN based model with gated synergy (no weight sharing)
class GCNNetGated(nn.Module):
    def __init__(self, n_output=2, n_filters=32, embed_dim=128, num_features_xd=78, num_features_xt=954, output_dim=128, dropout=0.2):
        super(GCNNetGated, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # SMILES1 graph branch (drug1)
        self.n_output = n_output
        self.drug1_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.drug1_conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.drug1_conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.drug1_fc_g1 = nn.Linear(num_features_xd*4, num_features_xd*2)
        self.drug1_fc_g2 = nn.Linear(num_features_xd*2, output_dim)

        # SMILES2 graph branch (drug2, no sharing)
        self.drug2_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.drug2_conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.drug2_conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.drug2_fc_g1 = nn.Linear(num_features_xd * 4, num_features_xd*2)
        self.drug2_fc_g2 = nn.Linear(num_features_xd*2, output_dim)

        # Cell line feature reduction
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        # Added: Lightweight gated synergy module
        self.synergy_gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid()
        )

        # Prediction head (input now 4*output_dim due to synergy feat)
        self.fc1 = nn.Linear(4 * output_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, self.n_output)

    def forward(self, data1, data2):
        x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        # Drug1 encoding
        x1 = self.relu(self.drug1_conv1(x1, edge_index1))
        x1 = self.relu(self.drug1_conv2(x1, edge_index1))
        x1 = self.relu(self.drug1_conv3(x1, edge_index1))
        x1 = gmp(x1, batch1)
        x1 = self.relu(self.drug1_fc_g1(x1))
        x1 = self.dropout(x1)
        x1 = self.drug1_fc_g2(x1)
        x1 = self.dropout(x1)

        # Drug2 encoding
        x2 = self.relu(self.drug2_conv1(x2, edge_index2))
        x2 = self.relu(self.drug2_conv2(x2, edge_index2))
        x2 = self.relu(self.drug2_conv3(x2, edge_index2))
        x2 = gmp(x2, batch2)
        x2 = self.relu(self.drug2_fc_g1(x2))
        x2 = self.dropout(x2)
        x2 = self.drug2_fc_g2(x2)
        x2 = self.dropout(x2)

        # Cell feature
        cell_vector = F.normalize(cell, 2, 1)
        cell_vector = self.reduction(cell_vector)

        # Added: Gated synergy interaction
        interaction = x1 * x2
        gate = self.synergy_gate(torch.cat((x1, x2), 1))
        synergy = gate * interaction

        # Concat with synergy
        xc = torch.cat((x1, x2, synergy, cell_vector), 1)

        # Dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out