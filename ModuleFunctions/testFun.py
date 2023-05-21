import numpy as np
import torch
# 绘图
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from ModuleFunctions import toolsFun


"""
# 测试函数
"""
from sklearn.metrics import confusion_matrix


def test(pre_core_k, labels_k, model, test_loader, criterion, k_num, cnf_matrix, save_info_path, save_graph_path):
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
        info = 'Test Loss: {:.4f}, Test Acc: {:.4f}\n'.format(test_loss, test_acc)
        toolsFun.save_train_val_info(k_num, save_info_path, info)
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
    plt.savefig(save_graph_path + '/' +  str(k_num) +'-fold-roc_curve.jpg')
    plt.close()


    '''混淆矩阵'''
    # 计算混淆矩阵
    confusion_mat = confusion_matrix(y_true, y_preds)
    # 绘制混淆矩阵
    classes = ['negative', 'positive']
    plt.figure()
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
    plt.savefig(save_graph_path + '/' +  str(k_num) +'-fold-cm-figure.jpg')
    plt.close()
