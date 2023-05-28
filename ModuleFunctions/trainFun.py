import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from ModuleFunctions import toolsFun

"""
# 训练函数
"""
def train(best_val_loss, patience, no_improvement_count, scheduler, model, train_loader, val_loader, criterion, optimizer, num_epochs, save_info_path, k_num, save_graph_path, save_pth_path=None):
    """训练与验证函数的定义"""
    # 定义一些列表用于存储训练过程中各个epochs的loss和acc
    global best_val_acc
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
        best_val_acc = 0.0
        train_loss = 0.0
        train_loss_per_batch = 0.0
        train_acc = 0.0
        train_acc_per_batch = 0.0
        val_loss = 0.0
        val_acc = 0.0
        model.train()
        for i, (data, label) in tqdm(enumerate(train_loader), total=len(train_loader), desc='训练模型', ncols=80, colour='white'):
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
            train_acc += torch.sum(preds == label.data) # type: ignore
            train_acc_per_batch += torch.sum(preds == label.data) # type: ignore
            # 更新学习率
            scheduler.step()
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        # 验证
        model.eval()
        for i, (data, label) in tqdm(enumerate(val_loader), total=len(val_loader), desc='验证模型', ncols=80, colour='white'):
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
        info = str('Epoch [{}/{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}\n'
            .format(epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc))
        print(info)
        # 将训练结果存入txt
        toolsFun.save_train_val_info(k_num, save_info_path, info)
        
        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            # 保存模型
            torch.save(model, str(save_pth_path) + '/' + 'k='+ str(k_num) + '-best_val_acc-model.pth')

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
    plt.savefig(save_graph_path + '/' +  str(k_num) +'-fold-loss_curve.jpg')
    plt.close()


    # 绘制准确率曲线
    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('ACC')
    plt.legend()
    plt.savefig(save_graph_path + '/' +  str(k_num) +'-fold-acc_curve.jpg')
    plt.close()
    
    return float(best_val_acc)
