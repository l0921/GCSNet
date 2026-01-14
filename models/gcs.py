import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp


class GCSNet(nn.Module):
    """
    GCSNet: Gated Co-Synergy Shared Network (Enhanced Version)

    Enhancements over the original:
    - Symmetric interaction modeling (product + absolute difference)
    - Cell-aware FiLM-style gated synergy module
    - Residual-protected synergy signal for stable training
    """

    def __init__(
        self,
        n_output=2,
        num_features_xd=78,
        num_features_xt=954,
        output_dim=128,
        dropout=0.2
    ):
        super(GCSNet, self).__init__()

        self.n_output = n_output
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # ========= Shared Drug GCN Encoder =========
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)

        self.fc_g1 = nn.Linear(num_features_xd * 4, num_features_xd * 2)
        self.fc_g2 = nn.Linear(num_features_xd * 2, output_dim)

        # ========= Cell Line Feature Encoder =========
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        # ========= Enhanced Gated Synergy Module =========
        # Input: interaction ⊙ + |d1-d2| + cell_vec  => 3 * output_dim
        # Output: scale and bias (FiLM)
        #生成门控参数，控制输出scale和bias，被控制的对象是interaction
        self.synergy_gate = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim * 2)
        )

        # ========= Prediction Head =========
        # drug1 + drug2 + synergy + cell = 4 * output_dim
        self.fc1 = nn.Linear(4 * output_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, n_output)

    # ========= Shared Drug Encoder =========
    def drug_encoder(self, x, edge_index, batch):
        x = su(self.conv2(x, edge_index))
        x = self.relu(selelf.relu(self.conv1(x, edge_index))
        x = self.relf.conv3(x, edge_index))

        x = gmp(x, batch)

        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        return x

    # ========= Forward =========
    def forward(self, data1, data2):
        # Unpack
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

        # Shared drug encoding
        drug1_emb = self.drug_encoder(x1, edge_index1, batch1)
        drug2_emb = self.drug_encoder(x2, edge_index2, batch2)

        # Cell encoding
        cell = F.normalize(cell, p=2, dim=1)
        cell_vec = self.reduction(cell)

        # ========= Enhanced Gated Synergy =========
        # Symmetric interaction features
        interaction = drug1_emb * drug2_emb#哪些地方一样
        difference = torch.abs(drug1_emb - drug2_emb)#哪些地方差别很大

        # Gate input (cell-aware)合并成一个向量，决定在这个细胞种药物该如何相互作用
        gate_input = torch.cat(
            (interaction, difference, cell_vec),
            dim=1
        )

        # FiLM-style gating
        scale, bias = self.synergy_gate(gate_input).chunk(2, dim=1)#生成两个调节值，分别调节interaction的scale和bias
        scale = torch.sigmoid(scale)#不允许模型生成过大的scale

        # Residual-protected synergy feature
        synergy_feat = interaction * (1.0 + scale) + bias#gate_input 告诉模型 “该怎么调”，interaction 是 “被调的东西”

        # ========= Feature Fusion =========
        xc = torch.cat(
            (drug1_emb, drug2_emb, synergy_feat, cell_vec),
            dim=1
        )

        # Prediction head
        xc = self.relu(self.fc1(xc))
        xc = self.dropout(xc)

        xc = self.relu(self.fc2(xc))
        xc = self.dropout(xc)

        out = self.out(xc)
        return out
