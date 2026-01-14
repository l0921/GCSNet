import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp


# SiameseGCN + Gate (No Interaction) 
# 消融变体：权重共享 + 只加 Gate 模块，但不使用 d1*d2 交互项
# 目的：证明 gate 本身单独作用几乎无效，只有与交互结合才有意义（负面对照）
class GCGNet(torch.nn.Module):
    def __init__(self, n_output=2, n_filters=32, embed_dim=128, num_features_xd=78, num_features_xt=954, output_dim=128, dropout=0.2):
        super(GCGNet, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.n_output = n_output

        # ========= 共享的药物GCN编码器（权重共享）=========
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd * 4, num_features_xd * 2)
        self.fc_g2 = torch.nn.Linear(num_features_xd * 2, output_dim)

        # ========= 细胞系特征处理（保持不变）=========
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        # ========= 只保留 Gate 模块（但不乘以交互）=========
        # 这里 gate 仍然基于 [d1, d2] 生成一个 output_dim 维的向量，用于占位融合
        self.synergy_gate = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.Sigmoid()
        )

        # ========= 融合层：输入维度为 4*output_dim (d1 + d2 + gate_output + cell) =========
        self.fc1 = nn.Linear(4 * output_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, self.n_output)

    def drug_encoder(self, x, edge_index, batch):
        """共享的药物图编码器"""
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)

        x = gmp(x, batch)

        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        return x

    def forward(self, data1, data2):
        x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        # 共享权重编码两个药物
        d1 = self.drug_encoder(x1, edge_index1, batch1)
        d2 = self.drug_encoder(x2, edge_index2, batch2)

        # 细胞系编码
        cell_vector = F.normalize(cell, p=2, dim=1)
        cell_vector = self.reduction(cell_vector)

        # 只生成 gate 输出，不计算 d1*d2 交互项
        gate_output = self.synergy_gate(torch.cat([d1, d2], dim=1))  # shape: [batch, output_dim]

        # 注意：这里故意不乘以任何交互，直接使用 gate_output 作为第四个特征
        # （相当于一个基于两药物的额外“注意力权重向量”，但无实际交互语义）

        # 融合：d1 + d2 + gate_output + cell_vector  → 4 * output_dim
        xc = torch.cat((d1, d2, gate_output, cell_vector), dim=1)

        # 预测头（与 GCSNet 一致）
        xc = self.relu(self.fc1(xc))
        xc = self.dropout(xc)
        xc = self.relu(self.fc2(xc))
        xc = self.dropout(xc)
        out = self.out(xc)

        return out