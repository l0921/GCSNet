GCSNet: A Gated Co-Synergy Network for drug synergy prediction using graph neural networks.GCSNet：一种使用图神经网络进行药物协同预测的门控协同网络。

tips   提示

Step1: create dataset for input步骤1：创建输入数据集
creat_data_DC.py could deal with the file like data/new_labels_0_10.csvcreat_data_DC.py可以处理像data
ew_labels_0_10.csv这样的文件

Step2: input the train data to the modle Step2：将列车数据输入到模型中

  training_GCS.py is a train model by GCS training_GCS.py是GCS提供的训练模型

  training_validation.py is a validation model for predicting novel drug pairsTraining_validation.py是一个预测新药对的验证模型

<img width   宽度="931" height="519" alt="屏幕截图 2026-03-10 160916" src="https://github.com/user-attachments/assets/9177c2a6-cd06-4da2-874b-f1e58ba60fee" />
