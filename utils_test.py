import os
from itertools import islice
import sys
import pickle
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
from creat_data_DC import creat_data

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='_drug1',
                 xd=None, xt=None, y=None, xt_featrue=None, transform=None,
                 pre_transform=None, smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        #调用父类的初始化方法
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'，把传进来的数据集名字保存类的属性
        self.dataset = dataset

        
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            # Minimal compatible loader: try normal load; if UnpicklingError occurs
            # (e.g., PyTorch 2.6 weights-only behavior), fall back to weights_only=False.
            try:
                self.data, self.slices = torch.load(self.processed_paths[0])
            except pickle.UnpicklingError:
                print('Warning: standard torch.load failed with UnpicklingError; trying safe_globals fallback then weights_only=False')
                # Try safe_globals with fully-qualified name if available (preferred)
                if hasattr(torch.serialization, 'safe_globals'):
                    try:
                        with torch.serialization.safe_globals(["torch_geometric.data.data.DataEdgeAttr"]):
                            self.data, self.slices = torch.load(self.processed_paths[0])
                    except Exception:
                        print('safe_globals fallback failed; retrying with weights_only=False')
                        try:
                            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
                        except TypeError:
                            raise
                else:
                    try:
                        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
                    except TypeError:
                        raise
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, xt_featrue, y, smile_graph)
            # 新增 weights_only=False，兼容旧版本保存的含自定义类的.pt文件
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)   


    @property
    def raw_file_names(self):
        pass#这个函数是用来下载数据集的，但是我们的数据集全在本地，就直接pass不用管这个函数
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

# 获取cell的特征
    def get_cell_feature(self, cellId, cell_features):
        for row in islice(cell_features, 0, None):
            if cellId in row[0]:
                return row[1:]
        return False

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, xt_featrue, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"#要一样的长度是指他们各自数量的要相同对应的上
        data_list = []
        data_len = len(xd)
        print('number of data', data_len)
        for i in range(data_len):
            # print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.Tensor([labels]))
            cell = self.get_cell_feature(target, xt_featrue)

            if cell is False:  # 如果读取cell失败则中断程序
                print('cell', cell)
                sys.exit()

            new_cell = []
            # print('cell_feature', cell_feature)
            for n in cell:
                new_cell.append(float(n))
            GCNData.cell = torch.FloatTensor([new_cell])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

#回归任务的指标，根均方误差 RMSE
def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

#保存AUC值
def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')

#均方误差MSE，是RMSE的平方
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse

#皮尔逊相关系数，衡量预测和真实值的线性相关性
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp

#斯皮尔曼相关系数，衡量预测和真实的排名相关性
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs

#一致性指数，只看这个真实的比a高的组合是否和预测的比a高的组合一致
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci
