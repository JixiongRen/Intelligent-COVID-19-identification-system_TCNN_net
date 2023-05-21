import librosa
import soundfile as sf
import numpy as np
import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from torch import optim
import torch.optim.lr_scheduler as lr_scheduler

# TCNN 模型
from model.TCNN2 import TCNN2

# 绘图
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

# 损失函数
from loss_function import focalloss


def preprocess_data(audio_file_path):
    '''将音频数据转化为numpy数组'''
    # 设置参数
    sr = 16000
    duration = 2
    # 初始化变量
    data = []  # 存放音频数据
    # 加载音频文件
    filepath = audio_file_path
    y, sr = sf.read(filepath)
    # 转化为单声道
    if len(y.shape) > 1:
        y = librosa.to_mono(y)
    # 修改为16kHz采样率
    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    # 归一化
    y = librosa.util.normalize(y)  # 归一化
    # 剪切或补 0
    if len(y) < 32000:
        y = np.pad(y, (32000 - len(y), 0), mode='constant')
    else:
        y = y[0:32000]
    # 将数据添加到列表中，包含标签和数据
    data.append(y.reshape(1, -1))  # 将数据转化为 1 行，-1 列的形式
    # 转化为 numpy 数组
    data = np.array(data)
    return data


def path2target(audio_dir):
    '''建立音频路径-标签之关系'''
    # 初始化变量
    filepath_array = []  # 存放音频数据
    labels = []
    # 得到音频文件列表
    files = os.listdir(audio_dir)
    # 处理每个音频文件
    for file in files:
        # 加载音频文件
        filepath = os.path.join(audio_dir, file)
        audios = os.listdir(filepath)
        for audio in audios:
            audiofilepath = os.path.join(filepath, audio)
            filepath_array.append(audiofilepath)
            if 'negative' in audiofilepath:
                labels.append(0)
            else:
                labels.append(1)
    return filepath_array, labels


'''建立文件路径与标签的对应关系'''

# windows 下训练时取消下面一行的注释
# filepath_array, labels = path2target(r'D:\大三下文件夹\大创\DataSet\2sData_bak')
# ubuntu 下训练时取消下面一行的注释
# filepath_array, labels = path2target(r'/media/renjixiong/Data/大三下文件夹/大创/DataSet/2sData_bak')
# WSL 下训练时取消下面一行注释
filepath_array, labels = path2target(r'/home/renjixiong/Model_Data/DataSet/2sData_bak')


'''
# 划分测试集、训练集、验证集
'''

import random

# 将数据路径和对应标签打包成元组列表
data_pairs = list(zip(filepath_array, labels))
# 随机打乱元组列表
random.shuffle(data_pairs)
# 计算每个字符串列表应包含的元素数量
total_count = len(data_pairs)
list1_count = int(total_count * 0.6)
list2_count = int(total_count * 0.2)
# 使用切片操作将元组列表分割成3个比例为6:2:2的元组列表
lists = [data_pairs[:list1_count], data_pairs[list1_count:list1_count + list2_count],
         data_pairs[list1_count + list2_count:]]
# 分别提取字符串列表和数字列表
path_lists = [[pair[0] for pair in sublist] for sublist in lists]
label_lists = [[pair[1] for pair in sublist] for sublist in lists]
# 定义训练集、验证集、测试集数据容器
train_data = []
train_label = []
val_data = []
val_label = []
test_data = []
test_label = []
# 保存结果
for i, path_list in enumerate(path_lists):
    if i == 0:
        train_data = path_lists[i]
        # print(train_data)
        train_label = label_lists[i]
    if i == 1:
        val_data = path_lists[i]
        val_label = label_lists[i]
    if i == 2:
        test_data = path_lists[i]
        test_label = label_lists[i]

'''
# 生成DataLoader
'''

batch_size = 128
class MyDataset(Dataset):
    def __init__(self, data_path, data_label):
        self.data_path = data_path
        self.data_label = data_label

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        # 加载数据
        data = self.data_path[idx]
        # print(data)
        # 进行预处理
        data = preprocess_data(data)
        # 获取标签
        label = self.data_label[idx]
        # 转换为 tensor
        data = torch.tensor(data).float()
        label = torch.tensor(label).float()
        return data, label


# 训练集
train_dataset = MyDataset(train_data, train_label)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 验证集
val_dataset = MyDataset(val_data, val_label)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# 测试集
test_dataset = MyDataset(test_data, test_label)
test_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


def train(best_val_loss, patience, no_improvement_count, scheduler, model, train_loader, val_loader, criterion, optimizer, num_epochs):
    '''训练与验证函数的定义'''
    # 定义一些列表用于存储训练过程中各个epochs的loss和acc
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []


# 设置GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion.to(device)

    # 训练循环
    for epoch in range(num_epochs):
        loss_min = float('inf')
        train_loss = 0.0
        train_loss_per_batch = 0.0
        train_acc = 0.0
        train_acc_per_batch = 0.0
        val_loss = 0.0
        val_acc = 0.0
        model.train()
        for i, (data, label) in tqdm(enumerate(train_loader), total=len(train_loader), desc='训练模型', ncols=80, colour='red'):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            label2calculoss = label.to(torch.int64)
            loss = criterion(outputs, label2calculoss)
            # loss = criterion(outputs, label)

            '''flood机制'''
            # if loss.item() < loss_min:
            #     loss_min = loss
            # flood = (loss - loss_min).abs() + loss_min
            # flood.backward()
            loss.backward()
            

            optimizer.step()
            train_loss += loss.item() * data.size(0)
            train_loss_per_batch += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds == label.data)
            train_acc_per_batch += torch.sum(preds == label.data)

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)

        model.eval()
        for i, (data, label) in tqdm(enumerate(val_loader), total=len(val_loader), desc='验证模型', ncols=80, colour='green'):
            data = data.to(device)
            label = label.to(device)
            outputs = model(data)

            label_loss = label.to(torch.int64)
            loss = criterion(outputs, label_loss)
            val_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            val_acc += torch.sum(preds == label.data)

        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)

        print('Epoch [{}/{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}\n'
              .format(epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc))

        '''在每个epoch之后将该轮的loss与acc存入列表'''
        # 计算训练集的损失和准确率
        train_losses.append(train_loss)
        train_accuracies.append(train_acc.item()) # type: ignore
        # 计算验证集的损失和准确率
        val_losses.append(val_loss)
        val_accuracies.append(val_acc.item()) # type: ignore

        # 更新学习率
        scheduler.step()
        # 检查验证损失是否有改善
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # 判断是否停止训练
        if no_improvement_count > patience:
            '若希望引入早停机制，则取消第1,2行注释；若不早停则取消第三行注释'
            # print(f"Early stopping at epoch {epoch+1}")
            # break
            continue

    '''训练结束，绘图'''
    # 绘制损失曲线
    plt.figure()
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('LOSS')
    plt.legend()
    plt.show()

    # 绘制准确率曲线
    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('ACC')
    plt.legend()
    plt.show()


def test(model, test_loader, criterion):
    '''测试函数'''
    # 存储预测概率分数和实际标签
    y_preds = []
    y_scores = []
    y_true = []

    # 设置GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion.to(device)

    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)
            outputs = model(data)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            scores = outputs[:, 1].cpu().numpy()  # 假设模型输出的是两个类别的概率分数，这里选择第二个类别的概率作为预测概率分数
            truelabel = label.cpu().numpy()
            y_scores.extend(scores)
            y_true.extend(truelabel)
            y_preds.extend(preds)

            label_loss = label.to(torch.int64)
            loss = criterion(outputs, label_loss)
            test_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            test_acc += torch.sum(preds == label.data)

        test_loss /= len(test_loader.dataset)
        test_acc /= len(test_loader.dataset)

        print('Test Loss: {:.4f}, Test Acc: {:.4f}'.format(test_loss, test_acc))

    '''ROC曲线'''
    # 计算FPR、TPR和阈值
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    '''混淆矩阵'''
    # 计算混淆矩阵
    confusion_mat = confusion_matrix(y_true, y_preds)
    # 绘制混淆矩阵
    classes = np.unique(y_true)
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues) # type: ignore
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # 添加数值标签
    thresh = confusion_mat.max() / 2.
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            plt.text(j, i, format(confusion_mat[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if confusion_mat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()


"""
# 定义超参数
"""
# 加载模型
model = TCNN2()

# 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
criterion = focalloss.FocalLoss(gamma=1, alpha=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.00001) # 使用最基本的，
scheduler = lr_scheduler.StepLR(optimizer, step_size=21, gamma=0.1)
num_epochs = 64

'''早停相关'''
best_val_loss = float('inf')  # 初始化最佳验证损失
patience = 10  # 定义耐心值，即连续多少个epoch验证损失没有提升就停止训练
no_improvement_count = 0  # 没有改善的epoch数

# 训练
train(best_val_loss, patience, no_improvement_count, scheduler, model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs)
# train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=90)

# 验证
test(model, test_dataloader, criterion)

# 保存模型参数
torch.save(model.state_dict(), 'pth_files/model.pth')
