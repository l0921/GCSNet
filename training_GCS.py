import random
import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch.utils.data as Data
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset
from models.gat import GATNet
from models.gat_gcn_test import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from models.gcs import GCSNet
from utils_test import *
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import pandas as pd

#这个完整代码在5折交叉验证的过程中，每折都会：
# 保存这一折表现最好的模型权重（基于 AUC 最高的那一轮）
# 保存每一轮（epoch）的所有分类指标（AUC、AUPR、ACC 等）
# 保存这一折最佳模型在测试集上的详细预测结果（索引、真实标签、预测标签、预测分数）


# training function at each epoch
def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch):
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))
    model.train()
    # train_loader = np.array(train_loader)
    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        data1 = data[0]
        data2 = data[1]
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = data[0].y.view(-1, 1).long().to(device)
        y = y.squeeze(1)
        optimizer.zero_grad()#梯度清零
        output = model(data1, data2)#前向传播，得到预测值。会结合drug1和drug2的数据以及细胞特征
        loss = loss_fn(output, y)#计算损失值
        # print('loss', loss)
        loss.backward()#反向传播计算梯度
        optimizer.step()#更新模型参数
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data1.x),
                                                                           len(drug1_loader_train.dataset),
                                                                           100. * batch_idx / len(drug1_loader_train),
                                                                           loss.item()))


def predicting(model, device, drug1_loader_test, drug2_loader_test):
    model.eval()#切换到评估模式
    total_preds = torch.Tensor()#预测的正类概率
    total_labels = torch.Tensor()#真实的标签
    total_prelabels = torch.Tensor()#预测的标签
    print('Make prediction for {} samples...'.format(len(drug1_loader_test.dataset)))#打印训练的样本数
    with torch.no_grad():#关闭梯度计算，节省内存
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            output = model(data1, data2)
            ys = F.softmax(output, 1).to('cpu').data.numpy()#将输出转换为概率分布
            predicted_labels = list(map(lambda x: np.argmax(x), ys))#选择概率最大的类别作为预测标签
            predicted_scores = list(map(lambda x: x[1], ys))#选择正率概率
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)#累加正类概率
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)#累加预测标签
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)#累加真实标签
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()
    # 返回真实标签，预测的正类概率，预测标签

# 数据集打乱
def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

#按指定比例将数据集划分为俩个子集（ratio划分比例）
def split_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


# 保存最佳模型预测结果函数
def save_best_result(test_num, T, Y, S, result_file_name):
    """
    保存每折最佳模型预测结果为：
    | sample_idx | y_true | y_pred | y_score |
    """
    # 构建 DataFrame
    df = pd.DataFrame({
        'sample_idx': test_num,
        'y_true': T,
        'y_pred': Y,
        'y_score': S
    })
    df.to_csv(result_file_name, index=False)



# modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
# datasets = [['davis', 'kiba'][int(sys.argv[1])]]
# model_st = modeling.__name__
modeling = GCSNet
#modeling = GCINet

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
datafile = 'new_labels_0_10'

# CPU or GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

drug1_data = TestbedDataset(root='data', dataset=datafile + '_drug1')
drug2_data = TestbedDataset(root='data', dataset=datafile + '_drug2')

lenth = len(drug1_data)
pot = int(lenth/5)#用来做数据集划分
print('lenth', lenth)
print('pot', pot)



#summary_flie = 'data/result/GCI/GCINet_best_per_fold_summary.txt'
summary_file = 'data/result/GCS/GCS_best_per_fold_summary.txt'
with open(summary_file, 'w') as f:
    f.write('Fold\tEpoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL\n')

#5折交叉验证
random_num = random.sample(range(0, lenth), lenth)#生成一个从0到lenth-1的随机排列索引列表
for i in range(5):
    test_num = random_num[pot*i:pot*(i+1)]
    train_num = random_num[:pot*i] + random_num[pot*(i+1):]

    drug1_data_train = drug1_data[train_num]
    drug1_data_test = drug1_data[test_num]
    # print('type(drug1_data_train)', type(drug1_data_train))
    # print('drug1_data_train[0]', drug1_data_train[0])
    # print('len(drug1_data_train)', len(drug1_data_train))
    drug1_loader_train = DataLoader(drug1_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    drug1_loader_test = DataLoader(drug1_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None)


    drug2_data_test = drug2_data[test_num]
    drug2_data_train = drug2_data[train_num]

    drug2_loader_train = DataLoader(drug2_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None)

    model = modeling().to(device)#每一折都要重新初始化模型
    loss_fn = nn.CrossEntropyLoss()#使用交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)#用Adam优化器

    #保存模型
    model_file_name = 'data/result/GCS/GCSNet(DrugA_DrugB)' + str(i) + '--model_' + datafile +  '.model'#每一折训练好的模型
    result_file_name = 'data/result/GCS/GCSNet(DrugA_DrugB)' + str(i) + '--result_' + datafile +  '.csv'#每一折的预测结果
    file_AUCs = 'data/result/GCS/GCSNet(DrugA_DrugB)' + str(i) + '--AUCs--' + datafile + '.txt'#每轮训练的评估指标
    AUCs = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')


    # model_file_name = 'data/result/GCS/GCSNet(DrugA_DrugB)' + str(i) + '--model_' + datafile +  '.model'
    # result_file_name = 'data/result/GCS/GCSNet(DrugA_DrugB)' + str(i) + '--result_' + datafile +  '.csv'
    # result = ('test_num\correct_label\predict_label\predict_score')
    # with open(result_file_name, 'w') as f:
    #     f.write(result + '\n')
    # file_AUCs = 'data/result/GCS/GCSNet(DrugA_DrugB)' + str(i) + '--AUCs--' + datafile + '.txt'
    # AUCs = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL')
    # with open(file_AUCs, 'w') as f:
    #     f.write(AUCs + '\n')

    # model_file_name = 'data/result/GCI/GCINet(DrugA_DrugB)' + str(i) + '--model_' + datafile +  '.model'
    # result_file_name = 'data/result/GCI/GCINet(DrugA_DrugB)' + str(i) + '--result_' + datafile +  '.csv'
    # result = ('test_num\correct_label\predict_label\predict_score')
    # with open(result_file_name, 'w') as f:
    #     f.write(result + '\n')
    # file_AUCs = 'data/result/GCI/GCINet(DrugA_DrugB)' + str(i) + '--AUCs--' + datafile + '.txt'
    # AUCs = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL')
    # with open(file_AUCs, 'w') as f:
    #     f.write(AUCs + '\n')
    


    #训练完一轮后在测试集上评估模型性能
    best_auc = 0#初始化变量

    best_metrics = None  # 新增：用来记录最佳那一行的所有指标（列表）

    for epoch in range(NUM_EPOCHS):
        train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1)
        T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test)#调用之前的预测函数
        # T is correct label
        # S is predict score
        # Y is predict label

        # compute preformence
        AUC = roc_auc_score(T, S)#计算AUC
        precision, recall, threshold = metrics.precision_recall_curve(T, S)#计算PR曲线
        PR_AUC = metrics.auc(recall, precision)#计算PRAUC
        BACC = balanced_accuracy_score(T, Y)#计算BACC，平衡准确率
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()#计算混淆矩阵
        TPR = tp / (tp + fn)#计算TPR，真阳性率
        PREC = precision_score(T, Y)#模型预测为正的所有样本中，真正正确的比例
        ACC = accuracy_score(T, Y)#所有样本中预测正确的比例
        KAPPA = cohen_kappa_score(T, Y)#衡量模型比“随机猜”好多少
        recall = recall_score(T, Y)#回归率

        # save data
        AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall]
        save_AUCs(AUCs, file_AUCs)#保存AUCs
        ret = [rmse(T, S), mse(T, S), pearson(T, S), spearman(T, S), ci(T, S)]#五个回归指标
        if best_auc < AUC:
            best_auc = AUC # 更新最佳 AUC
            best_metrics = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall]  # 新增：记录整行
            print(f'New best AUC: {best_auc:.4f} at epoch {epoch}')
            torch.save(model.state_dict(), model_file_name)
            save_best_result(test_num, T, Y, S, result_file_name)

    
    if best_metrics is not None:
        with open(summary_file, 'a') as f:  # 'a' 表示追加模式
            f.write(f'{i}\t' + '\t'.join(map(str, best_metrics)) + '\n')
        print(f'Fold {i} completed. Best AUC: {best_auc:.4f} (recorded to summary)')
    else:
        print(f'Fold {i} warning: no improvement recorded.')



# ===== 所有 5 折训练完全结束后：计算平均值并追加写入 summary 文件 =====
print('\n===== Final 5-Fold Cross Validation Results =====')

# 确保 summary_file 已经存在（前面已经创建并写了5折数据）
df_sum = pd.read_csv(summary_file, sep='\t')

# 计算主要指标的平均值 ± 标准差
auc_mean = df_sum['AUC_dev'].mean()
auc_std  = df_sum['AUC_dev'].std()
aupr_mean = df_sum['PR_AUC'].mean()
aupr_std  = df_sum['PR_AUC'].std()
acc_mean = df_sum['ACC'].mean()
acc_std  = df_sum['ACC'].std()
bacc_mean = df_sum['BACC'].mean()
bacc_std  = df_sum['BACC'].std()
prec_mean = df_sum['PREC'].mean()
prec_std  = df_sum['PREC'].std()
tpr_mean = df_sum['TPR'].mean()
tpr_std  = df_sum['TPR'].std()
kappa_mean = df_sum['KAPPA'].mean()
kappa_std  = df_sum['KAPPA'].std()
recall_mean = df_sum['RECALL'].mean()
recall_std  = df_sum['RECALL'].std()

# 打印到控制台
print(f"AUC   : {auc_mean:.4f} ± {auc_std:.4f}")
print(f"AUPR  : {aupr_mean:.4f} ± {aupr_std:.4f}")
print(f"ACC   : {acc_mean:.4f} ± {acc_std:.4f}")
print(f"BACC  : {bacc_mean:.4f} ± {bacc_std:.4f}")
print(f"PREC  : {prec_mean:.4f} ± {prec_std:.4f}")
print(f"TPR   : {tpr_mean:.4f} ± {tpr_std:.4f}")
print(f"KAPPA : {kappa_mean:.4f} ± {kappa_std:.4f}")
print(f"RECALL: {recall_mean:.4f} ± {recall_std:.4f}")
print('===============================================\n')

# ===== 关键：追加写入 summary 文件 =====
with open(summary_file, 'a') as f:  # 'a' 表示追加模式
    f.write('\n===== Average over 5 folds =====\n')
    f.write(f"AUC   : {auc_mean:.4f} ± {auc_std:.4f}\n")
    f.write(f"AUPR  : {aupr_mean:.4f} ± {aupr_std:.4f}\n")
    f.write(f"ACC   : {acc_mean:.4f} ± {acc_std:.4f}\n")
    f.write(f"BACC  : {bacc_mean:.4f} ± {bacc_std:.4f}\n")
    f.write(f"PREC  : {prec_mean:.4f} ± {prec_std:.4f}\n")
    f.write(f"TPR   : {tpr_mean:.4f} ± {tpr_std:.4f}\n")
    f.write(f"KAPPA : {kappa_mean:.4f} ± {kappa_std:.4f}\n")
    f.write(f"RECALL: {recall_mean:.4f} ± {recall_std:.4f}\n")
# ===========================================================