"""--------------------------------------------导入包-------------------------------------------"""
from datetime import datetime
import os
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from torch import optim
# TCNN 模型
from model.TCNN4 import TCNN4
# 损失函数
from loss_function import focalloss
# 余弦退火学习率衰减机制
from torch.optim.lr_scheduler import CosineAnnealingLR
# 各个编程模块
from ModuleFunctions import trainFun, testFun, dataPreprocessFun, toolsFun

"""--------------------------------------------执行区域-------------------------------------------"""

'''建立文件路径与标签的对应关系'''
# windows 下训练时取消下面一行的注释
# filepath_array, labels = path2target(r'D:\大三下文件夹\大创\DataSet\2sData_bak')
# ubuntu 下训练时取消下面一行的注释
# filepath_array, labels = path2target(r'/media/renjixiong/Data/大三下文件夹/大创/DataSet/2sData_bak')
# WSL 下训练时取消下面一行注释
train_val_data, train_val_label = dataPreprocessFun.path2target(r'/home/renjixiong/Model_Data/DataSet/Coswara_increased/train')
test_data, test_label = dataPreprocessFun.path2target(r'/home/renjixiong/Model_Data/DataSet/Coswara_increased/test')

'''定义部分超参数'''
batch_size = 32

'''打包成DataSet'''
# 训练集
train_val_dataset = dataPreprocessFun.MyDataset(train_val_data, train_val_label)

# 测试集
test_dataset = dataPreprocessFun.MyDataset(test_data, test_label)

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

# 创建文件夹用于保存每一折的图像
graphs_folder_name = 'figs/graphs_for_each_fold-' + str(timestampe)
if not os.path.exists(graphs_folder_name):
    os.makedirs(graphs_folder_name)

# 用于保存每一折的最好准确率的列表
best_acc_for_each_fold = []
# 定义文件夹路径和特定字符串
pth_folder_path = 'pth_files/' + str(timestampe) + 'model_pths'
for train_index, val_index in kf.split(train_val_dataset): # type: ignore
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
    num_epochs = 30
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
    train_fold = torch.utils.data.dataset.Subset(train_val_dataset, train_index) # type: ignore
    val_fold = torch.utils.data.dataset.Subset(train_val_dataset, val_index) # type: ignore
    # 计算训练集,验证集,测试集的大小
    train_num = len(train_fold)
    val_num = len(val_fold)
    # 打包成DataLoader类型用于训练
    train_dataloader = DataLoader(dataset=train_fold, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(dataset=val_fold, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    if k_num == 1:
        # 保存参数
        toolsFun.save_hyperParam2text('hyperParam/', 
                                      'TCNN4', 
                                      batch_size, 
                                      num_epochs, 
                                      lr, 
                                      patience, 
                                      criterion, 
                                      optimizer, 
                                      scheduler, 
                                      timestampe)
        

    if not os.path.exists(pth_folder_path):
        os.makedirs(pth_folder_path)
    # 训练
    best_acc_for_each_fold.append(
    # print(
        trainFun.train(best_val_loss,
                   patience,
                   no_improvement_count,
                   scheduler,
                   model,
                   train_dataloader,
                   val_dataloader,
                   criterion,
                   optimizer,
                   num_epochs,
                   'train_info/' + str(timestampe) + '-train_val_test_info.txt',
                   k_num,
                   graphs_folder_name,
                   pth_folder_path))    
    # 验证
    testFun.test(pre_score_k,
                 labels_k, 
                 model,
                 test_dataloader,
                 criterion,
                 k_num,
                 cnf_matrix,
                 'train_info/' + str(timestampe) + '-train_val_test_info.txt',
                 graphs_folder_name)

# 绘制 ROC
toolsFun.ROC_k(k,
               labels_k,
               pre_score_k,
               timestampe,
               'ROC_k5/')
# 绘制混淆矩阵
toolsFun.plot_confusion_matrix(cnf_matrix,
                               classes=['negative', 'positive'],
                               normalize=False,
                               title='Normalized confusion matrix',
                               path='ConfusionMartix_k5/cm_k5_' + timestampe + '.svg')
# 保存模型参数
torch.save(model.state_dict(), 'pth_files/model.pth') # type: ignore

'''将五折中最好的参数重命名'''
best_acc_for_each_fold_index = best_acc_for_each_fold.index(max(best_acc_for_each_fold))
search_string = 'k=' + str(best_acc_for_each_fold_index + 1) # 找到最大的准确率对应的折数
# 遍历文件夹下的所有文件
for filename in os.listdir(pth_folder_path):
    # 检查文件名是否包含特定字符串
    if search_string in filename:
        # 构建旧文件路径
        old_file_path = os.path.join(pth_folder_path, filename)
        # 构建新文件路径（这里假设您要将特定字符串替换为新字符串）
        new_filename = filename.replace(search_string, 'best_in_5folds-val_acc=' + str(max(best_acc_for_each_fold)))
        new_file_path = os.path.join(pth_folder_path, new_filename)
        # 重命名文件
        os.rename(old_file_path, new_file_path)
