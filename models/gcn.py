import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp


# GCN based model
class GCNNet(torch.nn.Module):#继承自torch的nn.Module类
    def __init__(self, n_output=2, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=954, output_dim=128, dropout=0.2):
    #构造函数，设置网络结构和超参数
    #n_output最终的输出类别是俩种；药物分子图的节点特征维度是78；细胞系特征维度是954；
        super(GCNNet, self).__init__()#调用父类的构造函数
        self.relu = nn.ReLU()#定义激活函数
        self.dropout = nn.Dropout(dropout)

        # SMILES1 graph branch药物1分支
        self.n_output = n_output
        self.drug1_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.drug1_conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.drug1_conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)#三层GCN卷积，扩展药物维度到312
        self.drug1_fc_g1 = torch.nn.Linear(num_features_xd*4, num_features_xd*2)
        self.drug1_fc_g2 = torch.nn.Linear(num_features_xd*2, output_dim)#俩层全连接降维

        # SMILES2 graph branch药物2分支
        self.drug2_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.drug2_conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.drug2_conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.drug2_fc_g1 = torch.nn.Linear(num_features_xd * 4, num_features_xd*2)
        self.drug2_fc_g2 = torch.nn.Linear(num_features_xd*2, output_dim)


        # DL cell featrues细胞系特征处理（全连接降维）
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        # combined layers融合层：三个128的向量融合
        self.fc1 = nn.Linear(3*output_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, self.n_output)#384 → 512 → 128 → 2（logits）

    def forward(self, data1, data2):#data1, data2是输入的SMILES1和SMILES2的图结构数据
        x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        # deal drug1
        x1 = self.drug1_conv1(x1, edge_index1)
        x1 = self.relu(x1)

        x1 = self.drug1_conv2(x1, edge_index1)
        x1 = self.relu(x1)

        x1 = self.drug1_conv3(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = gmp(x1, batch1)       # global max pooling把每个图的所有节点特征取最大值，将变长图压缩为固定长度向量

        # flatten
        x1 = self.relu(self.drug1_fc_g1(x1))
        x1 = self.dropout(x1)
        x1 = self.drug1_fc_g2(x1)
        x1 = self.dropout(x1)
        # print('x1.shape', x1.shape)
        # print('x1', x1[0])


        # deal drug2
        x2 = self.drug2_conv1(x2, edge_index2)
        x2 = self.relu(x2)

        x2 = self.drug2_conv2(x2, edge_index2)
        x2 = self.relu(x2)

        x2 = self.drug2_conv3(x2, edge_index2)
        x2 = self.relu(x2)
        x2 = gmp(x2, batch2)  # global max pooling

        # flatten
        x2 = self.relu(self.drug2_fc_g1(x2))
        x2 = self.dropout(x2)
        x2 = self.drug2_fc_g2(x2)
        x2 = self.dropout(x2)
        # print('x2.shape', x2.shape)
        # print('x', x2[0])

        # deal cell
        cell_vector = F.normalize(cell, 2, 1)
        cell_vector = self.reduction(cell_vector)#线性层，降维

        # concat融合三个特征
        xc = torch.cat((x1, x2, cell_vector), 1)#128+128+128=384维度的向量
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
