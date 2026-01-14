import csv
from itertools import islice

import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
# from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils_test import *


#挨个获取cell的特征（这个函数没被用上）
def get_cell_feature(cellId, cell_features):
    for row in islice(cell_features, 0, None):
        if row[0] == cellId:
            return row[1: ]

#给药物的每一个原子整合48维特征
#getsymbol获取符号、getdegree获取度数、gettotalnumhs获取氢原子数、getimplicitvalence获取隐式价态、getisaromatic获取是否芳香族
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

#计算这个原子的度数，一般的原子度数很少超过6，这里取10作为一个安全的经验上限
#one_of_k_encoding函数用于将输入x映射到一个one-hot编码向量，如果x不在允许集合中，则抛出异常。
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


#计算别的特征，只不过这个是可以有多个1，如果数据不存在就会选择unkonwn
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


#将药物的smile转化为图
def smile_to_graph(smile):
    #mol是一个“分子对象”
    mol = Chem.MolFromSmiles(smile)

    #获取原子数量
    c_size = mol.GetNumAtoms()

    #存放每个原子的特征
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        #将一个原子的单独特征，除以这个所有特征的综合，来进行归一化，防止其它原子的特征长度不同时造成训练不稳定
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    #将边的索引转换为有向图，ns是处理图的工具。这里转换成双向有向图是为了让GNN可以双向传递信息
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index
    #返回原子数量、原子特征、边的索引



def creat_data(datafile, cellfile):

#细胞系的数据处理
    file2 = cellfile
    cell_features = []
    #读取cell特征
    with open(file2) as csvfile:#with..as..自动打开文件，用完之后自动关闭
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件，一行一行、按逗号正确地读取csv文件
        for row in csv_reader:
            cell_features.append(row)#每读一行，row就是一个列表
    cell_features = np.array(cell_features)#转化为numpy数组，处理数值时更方便
    print('cell_features', cell_features)


#药物smlies转换为图
    compound_iso_smiles = []#存放药物的smile数据
    df = pd.read_csv('data/smiles.csv')#df像一个Excel表格
    compound_iso_smiles += list(df['smile'])#是一个列表形式
    compound_iso_smiles = set(compound_iso_smiles)#自动去重
    smile_graph = {}#空字典，SMILES字符串作为键，图数据作为值
    print('compound_iso_smiles', compound_iso_smiles)
    for smile in compound_iso_smiles:
        print('smiles', smile)
        g = smile_to_graph(smile)
        smile_graph[smile] = g


    datasets = datafile
    # convert to PyTorch data format，用pt形式是因为图数据得用这个形式来存放
    processed_data_file_train = 'data/processed/' + datasets + '_train.pt'#只是个字符串变量

    if ((not os.path.isfile(processed_data_file_train))):#这个是判断之前有没有生成相同的数据集的处理过后的数据，如果有就不再处理了
        df = pd.read_csv('data/' + datasets + '.csv')
        drug1, drug2, cell, label = list(df['drug1']), list(df['drug2']), list(df['cell']), list(df['label'])#转换成python列表
        drug1, drug2, cell, label = np.asarray(drug1), np.asarray(drug2), np.asarray(cell), np.asarray(label)#再转换成numpy数组，因为处理起来更快
        # make data PyTorch Geometric ready

        print('开始创建数据')
        TestbedDataset(root='data', dataset=datafile + '_drug1', xd=drug1, xt=cell, xt_featrue=cell_features, y=label,smile_graph=smile_graph)
        TestbedDataset(root='data', dataset=datafile + '_drug2', xd=drug2, xt=cell, xt_featrue=cell_features, y=label,smile_graph=smile_graph)
        print('创建数据成功')
        print('preparing ', datasets + '_.pt in pytorch format!')
    #
    #     print(processed_data_file_train, ' have been created')
    #
    # else:
    #     print(processed_data_file_train, ' are already created')

if __name__ == "__main__":
    # datafile = 'prostate'
    cellfile = 'data/new_cell_features_954.csv'
    da = ['new_labels_0_10']
    for datafile in da:
        creat_data(datafile, cellfile)
