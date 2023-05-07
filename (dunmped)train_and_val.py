# import librosa
# import numpy as np
# import os
# import torch
# from sklearn.model_selection import train_test_split
# import torch.utils.data as data
# from torch import nn, optim
#
# from TCNN import TCNN
#
# """
# # 数据预处理
# """
#
#
# def preprocess_data(audio_dir):
#     # 设置参数
#     sr = 16000
#     duration = 2
#
#     # 初始化变量
#     data = []  # 存放音频数据
#     labels = []  # 存放标签
#
#     # 得到音频文件列表
#     files = os.listdir(audio_dir)
#
#     # 处理每个音频文件
#     for file in files:
#         # 加载音频文件
#         filepath = os.path.join(audio_dir, file)
#         y, sr = librosa.load(filepath, sr=sr, duration=duration)
#
#         # 转化为单声道
#         if len(y.shape) > 1:
#             y = librosa.to_mono(y)
#
#         # 修改为16kHz采样率
#         y = librosa.resample(y, orig_sr=sr, target_sr=16000)
#
#         # 归一化
#         y = librosa.util.normalize(y)  # 归一化
#
#         # 剪切或补 0
#         if len(y) < 32000:
#             y = np.pad(y, (32000 - len(y), 0), mode='constant')
#         else:
#             y = y[0:32000]
#
#         # 将数据添加到列表中，包含标签和数据
#         data.append(y.reshape(1, -1))  # 将数据转化为 1 行，-1 列的形式
#         if 'negative' in audio_dir:
#             labels.append(0)
#         else:
#             labels.append(1)
#
#     # 转化为 numpy 数组
#     data = np.array(data)
#     labels = np.array(labels)
#
#     return data, labels
#
#
# neg_data, neg_labels = preprocess_data('D:/大三下文件夹/大创/DataSet/2sData/negative')
# pos_data, pos_labels = preprocess_data('D:/大三下文件夹/大创/DataSet/2sData/positive')
#
# """
# # 数据装填
# """
# # 划分数据集
# neg_train_data, neg_test_data, neg_train_label, neg_test_label = train_test_split(neg_data, neg_labels, test_size=0.2,
#                                                                                   random_state=42)
# pos_train_data, pos_test_data, pos_train_label, pos_test_label = train_test_split(pos_data, pos_labels, test_size=0.2,
#                                                                                   random_state=42)
#
# neg_train_data, neg_val_data, neg_train_label, neg_val_label = train_test_split(neg_train_data, neg_train_label,
#                                                                                 test_size=0.25, random_state=42)
# pos_train_data, pos_val_data, pos_train_label, pos_val_label = train_test_split(pos_train_data, pos_train_label,
#                                                                                 test_size=0.25, random_state=42)
#
# # 合并训练集、测试集、验证集数据和标签
# train_data = np.concatenate((neg_train_data, pos_train_data))
# val_data = np.concatenate((neg_val_data, pos_val_data))
# test_data = np.concatenate((neg_test_data, pos_test_data))
#
# train_label = np.concatenate((neg_train_label, pos_train_label))
# val_label = np.concatenate((neg_val_label, pos_val_label))
# test_label = np.concatenate((neg_test_label, pos_test_label))
#
# # 转换成Tensor
# train_data = torch.from_numpy(train_data).float()
# val_data = torch.from_numpy(val_data).float()
# test_data = torch.from_numpy(test_data).float()
#
# train_label = torch.from_numpy(train_label).long()
# val_label = torch.from_numpy(val_label).long()
# test_label = torch.from_numpy(test_label).long()
#
# # 生成 DataLoader
# batch_size = 1
#
# train_dataset = data.TensorDataset(train_data, train_label)
# train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#
# val_dataset = data.TensorDataset(val_data, val_label)
# val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#
# test_dataset = data.TensorDataset(test_data, test_label)
# test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
# print('训练集大小：', train_data.shape)
# print('验证集大小：', val_data.shape)
# print('测试集大小：', test_data.shape)
#
# """
# # 训练与验证函数的定义
# """
#
#
# def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
#     # 设置GPU
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     criterion.to(device)
#
#     # 训练循环
#     for epoch in range(num_epochs):
#         train_loss = 0.0
#         train_acc = 0.0
#         val_loss = 0.0
#         val_acc = 0.0
#         model.train()
#         for i, (data, label) in enumerate(train_loader):
#             data = data.to(device)
#             label = label.to(device)
#             optimizer.zero_grad()
#             outputs = model(data)
#             loss = criterion(outputs, label)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * data.size(0)
#             _, preds = torch.max(outputs, 1)
#             train_acc += torch.sum(preds == label.data)
#
#         train_loss /= len(train_loader.dataset)
#         train_acc /= len(train_loader.dataset)
#
#         model.eval()
#         for i, (data, label) in enumerate(val_loader):
#             data = data.to(device)
#             label = label.to(device)
#             outputs = model(data)
#             loss = criterion(outputs, label)
#             val_loss += loss.item() * data.size(0)
#             _, preds = torch.max(outputs, 1)
#             val_acc += torch.sum(preds == label.data)
#
#         val_loss /= len(val_loader.dataset)
#         val_acc /= len(val_loader.dataset)
#
#         print('Epoch [{}/{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'
#               .format(epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc))
#
#
# def test(model, test_loader, criterion):
#     # 设置GPU
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     criterion.to(device)
#
#     model.eval()
#     test_loss = 0.0
#     test_acc = 0.0
#     with torch.no_grad():
#         for i, (data, label) in enumerate(test_loader):
#             data = data.to(device)
#             label = label.to(device)
#             outputs = model(data)
#             loss = criterion(outputs, label)
#             test_loss += loss.item() * data.size(0)
#             _, preds = torch.max(outputs, 1)
#             test_acc += torch.sum(preds == label.data)
#
#         test_loss /= len(test_loader.dataset)
#         test_acc /= len(test_loader.dataset)
#
#         print('Test Loss: {:.4f}, Test Acc: {:.4f}'.format(test_loss, test_acc))
#
#
# """
# # 定义超参数
# """
# # 加载模型
# model = TCNN()
#
# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 训练
# train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
#
# # 验证
# test(model, test_loader, criterion)
