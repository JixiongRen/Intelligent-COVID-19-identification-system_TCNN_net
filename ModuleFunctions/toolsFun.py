import os
import random
import numpy as np
from tqdm.contrib import itertools
# 绘图
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


"""
# 保存超参数
"""

def save_superParam2text(path, netname, batch_size, num_epochs, learning_rate, patience, certerion, optimizer, scheduler, timestampe):
    # 保存超参数
    # 获取当前脚本文件的路径
    script_path = os.path.abspath(__file__)
    # 获取当前脚本文件的上级目录
    parent_dir = os.path.dirname(script_path)
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
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", verticalalignment="center",
                 color="#DE4F2E" if cm[i, j] > thresh else "#DE4F2E",
                 fontdict={'fontsize': 40})

    plt.tight_layout()
    plt.xlabel('Predicted label', fontdict={'fontsize': 20})
    plt.ylabel('True label', fontdict={'fontsize': 20})
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.2, top=0.9)
    # plt.show()
    plt.savefig(path)
def ROC_k(k, labels_k, pre_score_k, timestampe, path):
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
        plt.plot(fpr, tpr, label='V-' + str(i + 1) + ' (auc = {0:.4f})'.format(roc_auc/100.0), c=clr_1, alpha=0.2)

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

    plt.plot(data_x_plt, data_y_plt, label='AVG (auc = {0:.4f})'.format(avg/100.0), c=clr_2, alpha=1, linewidth=2)
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
    plt.savefig(path + 'TCNN4' + "_model_ROC_" + str(timestampe) + ".jpg")
    plt.show()