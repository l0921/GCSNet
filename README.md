Step1: create dataset for inputStep1：创建输入数据集
creat_data_DC.py could deal with the file like data/new_labels_0_10.csvcreat_data_DC.py可以像datacreat_data_DC.py可以处理datacreat_data_DC.py可以像data 
ew_labels_0_10.csvcreat_data_DC.py可以像datacreat_data_DC.py可以处理data那样处理文件
ew_labels_0_10.csvcreat_data_DC.py可以像data这样的文件
ew_labels_0_10.csv那样处理文件

Step2: input the train data to the modle Step2：将列车数据输入模型
training_GCS.py is a train model by GCS training_GCS.py是GCS的一个火车模型
training_validation.py is a validation model for predicting novel drug pairs py是一个用于预测新药配对的验证模型

<img width   宽度="1152" height="666" alt="屏幕截图 2026-01-14 154649" src="https://github.com/user-attachments/assets/063d615f-2709-4632-a874-73c09f3e15b9" />
这个模型架构图展示了一个药物协同预测网络：两个药物分子图（Drug A 和 Drug B）通过共享的 GCN 编码器提取特征，细胞系基因表达数据经 MLP 降维处理，三者随后进入增强型门控协同模块（Enhanced Gated Synergy Module）进行交互融合，最终通过全连接层输出协同预测结果。
