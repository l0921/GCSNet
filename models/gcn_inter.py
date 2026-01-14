import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp


class GCSNetInteraction(nn.Module):
    """
    Ablation Model: Shared Encoder + Explicit Interaction (No Gating)

    - Shared drug encoder
    - Explicit symmetric drug–drug interaction (product + absolute difference)
    - Cell features are concatenated only (no modulation)
    """

    def __init__(
        self,
        n_output=2,
        num_features_xd=78,
        num_features_xt=954,
        output_dim=128,
        dropout=0.2
    ):
        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # ========= Shared Drug GCN Encoder =========
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)

        self.fc_g1 = nn.Linear(num_features_xd * 4, num_features_xd * 2)
        self.fc_g2 = nn.Linear(num_features_xd * 2, output_dim)

        # ========= Cell Line Encoder =========
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        # ========= Prediction Head =========
        # drug1 + drug2 + interaction + difference + cell
        self.fc1 = nn.Linear(5 * output_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, n_output)

    # ========= Shared Drug Encoder =========
    def drug_encoder(self, x, edge_index, batch):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))

        x = gmp(x, batch)

        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        return x

    # ========= Forward =========
    def forward(self, data1, data2):
        # unpack
        x1, edge_index1, batch1, cell = (
            data1.x,
            data1.edge_index,
            data1.batch,
            data1.cell
        )
        x2, edge_index2, batch2 = (
            data2.x,
            data2.edge_index,
            data2.batch
        )

        # shared drug encoding
        drug1_emb = self.drug_encoder(x1, edge_index1, batch1)
        drug2_emb = self.drug_encoder(x2, edge_index2, batch2)

        # cell encoding
        cell = F.normalize(cell, p=2, dim=1)
        cell_vec = self.reduction(cell)

        # ========= Explicit Symmetric Interaction =========
        interaction = drug1_emb * drug2_emb
        difference = torch.abs(drug1_emb - drug2_emb)

        # ========= Feature Fusion =========
        xc = torch.cat(
            (drug1_emb, drug2_emb, interaction, difference, cell_vec),
            dim=1
        )

        # prediction head
        xc = self.relu(self.fc1(xc))
        xc = self.dropout(xc)
        xc = self.relu(self.fc2(xc))
        xc = self.dropout(xc)

        out = self.out(xc)
        return out
