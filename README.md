<img width   宽度="1167" height="653" alt="image" src="https://github.com/user-attachments/assets/76d6a050-a1d3-4a36-b665-bff3cb19e67c" /># GCSNet
# GCSNet: A Gated Co-Synergy Network for drug synergy prediction using graph neural networks.# GCSNet：使用图神经网络进行药物协同预测的门控协同网络。
tips   提示
Step1: create dataset for inputStep1：创建输入数据集
creat_data_DC.py could deal with the file like data/new_labels_0_10.csvcreat_data_DC.py可以像data
ew_labels_0_10.csv那样处理文件

Step2: input the train data to the modle Step2：将列车数据输入模型
training_GCS.py is a train model by GCS training_GCS.py是GCS的一个火车模型
training_validation.py is a validation model for predicting novel drug pairs py是一个用于预测新药配对的验证模型
![Uploading image.png…   上传image.png…]()
