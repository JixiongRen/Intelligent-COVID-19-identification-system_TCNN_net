"""--------------------------------------------导入包-------------------------------------------"""
import random
from datetime import datetime

import librosa
import soundfile as sf
import numpy as np
import os
import torch
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm.contrib import itertools

# TCNN 模型
from model.TCNN4 import TCNN4
# 绘图
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
# 损失函数
from loss_function import focalloss
# 余弦退火学习率衰减机制
from torch.optim.lr_scheduler import CosineAnnealingLR

"""--------------------------------------------函数区域-------------------------------------------"""

"""
# wave -> np.array
"""
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


"""
建立文件路径与标签的对应关系
"""
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

"""
# 训练函数
"""
def train(best_val_loss, patience, no_improvement_count, scheduler, model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """训练与验证函数的定义"""
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
        print('-' * 30, '\n', '共', num_epochs, '个epoch, 第', epoch + 1, '个epoch', '\n')
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
            # 更新学习率
            scheduler.step()
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        # 验证
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
        print('\nEpoch [{}/{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}\n'
              .format(epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc))

        '''在每个epoch之后将该轮的loss与acc存入列表'''
        # 计算训练集的损失和准确率
        train_losses.append(train_loss)
        train_accuracies.append(train_acc.item()) # type: ignore
        # 计算验证集的损失和准确率
        val_losses.append(val_loss)
        val_accuracies.append(val_acc.item()) # type: ignore
        # 检查验证损失是否有改善
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # 判断是否停止训练
        if no_improvement_count > patience:
            '若希望引入早停机制，则取消第1,2行注释；若不早停则取消第三行注释'
            print(f"Early stopping at epoch {epoch+1}")
            break
            # continue

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

"""
# 测试函数
"""
def test(pre_core_k, labels_k, model, test_loader, criterion, k_num):
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
            y_scores += scores.tolist()
            y_true += truelabel.tolist()
            y_preds.extend(preds)
            # 预测分数的最大值
            predict_y = torch.max(outputs, dim=1)[1]
            cm_labels = label.cpu().numpy().astype(int)
            # print(cm_labels)
            # print(predict_y)
            for index in range(len(label)):
                cnf_matrix[cm_labels[index]][predict_y[index]] += 1

            label_loss = label.to(torch.int64)
            loss = criterion(outputs, label_loss)
            test_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            test_acc += torch.sum(preds == label.data)

        test_loss /= len(test_loader.dataset)
        test_acc /= len(test_loader.dataset)
        print('Test Loss: {:.4f}, Test Acc: {:.4f}'.format(test_loss, test_acc))
    # 保存k折ROC参数
    pre_core_k.append(y_scores)
    labels_k.append(y_true)
    '''ROC曲线'''
    # 计算FPR、TPR和阈值
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (auc = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(str(k_num) + ' ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    '''混淆矩阵'''
    # 计算混淆矩阵
    confusion_mat = confusion_matrix(y_true, y_preds)
    # 绘制混淆矩阵
    classes = ['negative', 'positive']
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues) # type: ignore
    plt.title(str(k_num) + ' Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes, rotation=45)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # 添加数值标签
    thresh = confusion_mat.max() / 2.
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            plt.text(j, i, format(confusion_mat[i, j], 'd'),
                     horizontalalignment="center",
                     color="red" if confusion_mat[i, j] > thresh else "red",
                     fontdict={'fontsize': 40})

    plt.tight_layout()
    plt.show()

"""
# 保存超参数
"""

def save_superParam2text(path, netname, batch_size, num_epochs, learning_rate, patience, certerion, optimizer, scheduler, timestampe):
    # 保存超参数

    with open(path + timestampe +  '-' + netname + '-' + '-superParam.txt', 'w') as f:
        f.write('batch_size: ' + str(batch_size) + '\n')
        f.write('num_epochs: ' + str(num_epochs) + '\n')
        f.write('learning_rate: ' + str(learning_rate) + '\n')
        f.write('patience: ' + str(patience) + '\n')
        f.write('certerion: ' + str(certerion) + '\n')
        f.write('optimizer: ' + str(optimizer) + '\n')
        f.write('scheduler: ' + str(scheduler) + '\n')
        f.close()

"""
# 绘制混淆矩阵
"""
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, path=None):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #         print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    #         print(cm)
    #     else:
    #         print('显示具体数字：')
    #         print(cm)
    plt.figure(dpi=320, figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontdict={'fontsize': 20})
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontdict={'fontsize': 10})
    plt.yticks(tick_marks, classes, rotation=45, fontdict={'fontsize': 10})
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else '.0f'
    # fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="red" if cm[i, j] > thresh else "red",
                 fontdict={'fontsize': 40})

    plt.tight_layout()
    plt.xlabel('Predicted label', fontdict={'fontsize': 20})
    plt.ylabel('True label', fontdict={'fontsize': 20})
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.2, top=0.9)
    # plt.show()
    plt.savefig(path)
def ROC_k(k, labels_k, pre_score_k, timestampe):
    avg_x = []
    avg_y = []
    sum = 0
    clr_1 = 'tab:green'
    clr_2 = 'tab:green'
    clr_3 = 'k'

    plt.figure()
    for i in range(k):
        fpr, tpr, thersholds = roc_curve(labels_k[i], pre_score_k[i])
        avg_x.append(sorted(random.sample(list(fpr), len(list(fpr)))))
        avg_y.append(sorted(random.sample(list(tpr), len(list(tpr)))))
        roc_auc1 = auc(fpr, tpr)

        roc_auc = roc_auc1 * 100
        sum = sum + roc_auc
        plt.plot(fpr, tpr, label='V-' + str(i + 1) + ' (auc = {0:.2f})'.format(roc_auc), c=clr_1, alpha=0.2)

    data_x = np.array(avg_x, dtype=object)
    data_y = np.array(avg_y, dtype=object)
    avg = sum / k

    # 准备数据
    data_x_plt = []

    data_x_num = len(data_x[0])
    if data_x_num >= len(data_x[1]):
        data_x_num = len(data_x[1])
    if data_x_num >= len(data_x[2]):
        data_x_num = len(data_x[2])
    if data_x_num >= len(data_x[3]):
        data_x_num = len(data_x[3])
    if data_x_num >= len(data_x[4]):
        data_x_num = len(data_x[4])

    for i in range(5):
        data_x[i] = sorted(random.sample(data_x[i], data_x_num))

    for i in range(data_x_num):
        a = 0.0
        a += data_x[0][i]
        a += data_x[1][i]
        a += data_x[2][i]
        a += data_x[3][i]
        a += data_x[4][i]
        a = a / k
        data_x_plt.append(a)

    data_y_plt = []
    data_y_num = len(data_y[0])
    if data_y_num >= len(data_y[1]):
        data_y_num = len(data_y[1])
    if data_y_num >= len(data_y[2]):
        data_y_num = len(data_y[2])
    if data_y_num >= len(data_y[3]):
        data_y_num = len(data_y[3])
    if data_y_num >= len(data_y[4]):
        data_y_num = len(data_y[4])

    for i in range(5):
        data_y[i] = sorted(random.sample(data_y[i], data_y_num))

    for i in range(data_y_num):
        a = 0.0
        a += data_y[0][i]
        a += data_y[1][i]
        a += data_y[2][i]
        a += data_y[3][i]
        a += data_y[4][i]
        a = a / k
        data_y_plt.append(a)

    plt.plot(data_x_plt, data_y_plt, label='AVG (auc = {0:.4f})'.format(avg), c=clr_2, alpha=1, linewidth=2)
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.plot([0, 1], [0, 1], linestyle='--', label='chance', c=clr_3, alpha=.5)
    plt.legend(loc='lower right', frameon=False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.grid(color='gray', linestyle='--', linewidth=1, alpha=.3)
    plt.text(0, 1, 'PATIENT-LEVEL ROC', color='gray', fontsize=12)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('ROC_k5/' + 'TCNN4' + "_model_ROC_" + str(timestampe) + ".jpg")
    plt.show()
"""--------------------------------------------类区域-------------------------------------------"""
# 自定义数据集，打包成 DataSet
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


"""--------------------------------------------执行区域-------------------------------------------"""

'''建立文件路径与标签的对应关系'''
# windows 下训练时取消下面一行的注释
# filepath_array, labels = path2target(r'D:\大三下文件夹\大创\DataSet\2sData_bak')
# ubuntu 下训练时取消下面一行的注释
# filepath_array, labels = path2target(r'/media/renjixiong/Data/大三下文件夹/大创/DataSet/2sData_bak')
# WSL 下训练时取消下面一行注释
train_val_data, train_val_label = path2target(r'/home/renjixiong/Model_Data/DataSet/Coswara_increased/train')
test_data, test_label = path2target(r'/home/renjixiong/Model_Data/DataSet/Coswara_increased/test')

'''定义部分超参数'''
batch_size = 64

'''打包成DataSet'''
# 训练集
train_val_dataset = MyDataset(train_val_data, train_val_label)

# 测试集
test_dataset = MyDataset(test_data, test_label)

# 用于保存k折的ROC参数
pre_score_k = []
labels_k = []

# 保存混淆矩阵参数
cnf_matrix = np.zeros([2, 2])

timestampe = datetime.now().strftime('%Y-%m-%d-%H-%M')

'''五折交叉验证'''
# 划分成5份
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=34)
k_num = 0
best_acc_all = 0
for train_index, val_index in kf.split(train_val_dataset):
    best_acc = 0.0
    '''每一折实例化新的模型'''
    # 加载模型
    model = TCNN4()

    '''定义其余超参数'''

    # 将模型放到GPU上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('pth_files/pretrainmodel.pth', map_location=device))

    # 定义损失函数
    # criterion = nn.CrossEntropyLoss()
    criterion = focalloss.FocalLoss(gamma=1, alpha=0.5)

    #定义优化器
    lr = 0.00001
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # 学习率更新策略
    # StepLR()学习率调整策略，每30个epoch学习率变为原来的0.1
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # CosineAnnealingLR()学习率调整策略，每个epoch学习率都在变化，变化范围为[0.000001, 0.00001]
    num_epochs = 64
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.000001)

    '''早停的设置'''
    best_val_loss = float('inf')  # 初始化最佳验证损失
    patience = 10  # 定义耐心值，即连续多少个epoch验证损失没有提升就停止训练
    no_improvement_count = 0  # 没有改善的epoch数

    '''打印‘第k折交叉验证’'''
    k_num += 1
    print('\n')
    print("-" * 30)
    print("第{}折验证".format(k_num))
    train_fold = torch.utils.data.dataset.Subset(train_val_dataset, train_index)
    val_fold = torch.utils.data.dataset.Subset(train_val_dataset, val_index)
    # 计算训练集,验证集,测试集的大小
    train_num = len(train_fold)
    val_num = len(val_fold)
    # 打包成DataLoader类型 用于 训练
    train_dataloader = DataLoader(dataset=train_fold, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(dataset=val_fold, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    if k_num == 1:
        # 保存参数
        save_superParam2text('superParam/', 'TCNN4', batch_size, num_epochs, lr, patience, criterion, optimizer, scheduler, timestampe)
    # 训练
    train(best_val_loss, patience, no_improvement_count, scheduler, model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs)
    # 验证
    test(pre_score_k, labels_k, model, test_dataloader, criterion, k_num)

# 绘制 ROC
ROC_k(k, labels_k, pre_score_k, timestampe)
# 绘制混淆矩阵
plot_confusion_matrix(cnf_matrix, classes=['negative', 'positive'], normalize=False, title='Normalized confusion matrix', path='ConfusionMartix_k5/cm_k5_'+ timestampe +'.jpg')
# 保存模型参数
torch.save(model.state_dict(), 'pth_files/model.pth')
