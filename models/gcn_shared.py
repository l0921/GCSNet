# models/gcn_shared.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp


class GCNNetShared(nn.Module):
    """
    GCNNet with Weight Sharing
    两个药物使用完全相同的GCN编码器参数（权重共享），显著提升性能并减少参数量
    """
    def __init__(self, n_output=2, num_features_xd=78, num_features_xt=954, output_dim=128, dropout=0.2):
        super(GCNNetShared, self).__init__()

        self.n_output = n_output
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # ========= 共享的药物 GCN 编码器 =========
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)

        self.fc_g1 = nn.Linear(num_features_xd * 4, num_features_xd * 2)
        self.fc_g2 = nn.Linear(num_features_xd * 2, output_dim)

        # ========= 细胞系特征降维 =========
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        # ========= 预测头 =========
        self.fc1 = nn.Linear(3 * output_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, n_output)

    def drug_encoder(self, x, edge_index, batch):
        """共享的药物图编码器"""
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)

        x = gmp(x, batch)  # global max pooling

        x = self.fc_g1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc_g2(x)
        x = self.dropout(x)

        return x

    def forward(self, data1, data2):
        # 解包两个药物图和细胞系特征
        x1, edge_index1, batch1, cell = (
            data1.x, data1.edge_index, data1.batch, data1.cell
        )
        x2, edge_index2, batch2 = (
            data2.x, data2.edge_index, data2.batch
        )

        # 使用同一套参数编码两个药物
        drug1_emb = self.drug_encoder(x1, edge_index1, batch1)
        drug2_emb = self.drug_encoder(x2, edge_index2, batch2)

        # 细胞系特征处理
        cell = F.normalize(cell, p=2, dim=1)
        cell_vec = self.reduction(cell)

        # 融合：drug1 + drug2 + cell
        xc = torch.cat([drug1_emb, drug2_emb, cell_vec], dim=1)

        # 分类头
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        out = self.out(xc)
        return out