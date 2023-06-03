# import torch
# import torch.nn as nn
# from torchstat import stat  # 查看网络参数
# from torchsummary import summary  # 查看网络结构

# '''
#     建立模型
#     (batch_size,input_channels,step)
#     两秒的音频，step=2*16000=32000
# '''


# class TCNN0(nn.Module):
#     def __init__(self):
#         super(TCNN0, self).__init__()  # 继承父类的初始化方法
#         # b*1*32000 -> b*32*32000
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=2, dilation=2) # 卷积核太小，改 7 或 9 查文献
#         # b*32*32000 -> b*32*32000
#         self.bn1 = nn.BatchNorm1d(32)
#         # b*32*32000 -> b*32*32000
#         self.relu1 = nn.ReLU()
#         # b*32*32000 -> b*32*10666
#         self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
#         # b*32*10666 -> b*64*10666
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3)
#         # b*64*10666 -> b*64*10666
#         self.bn2 = nn.BatchNorm1d(64)
#         # b*64*10666 -> b*64*10666
#         self.relu2 = nn.ReLU()
#         # b*64*10666 -> b*64*2666
#         self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
#         # b*64*2666 -> b*64*2666*1
#         # self.conv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 188), stride=188)

#         self.fc1 = nn.Linear(64 * 2666, 128)
#         self.dropout = nn.Dropout(0.3)
#         self.relu3 = nn.ReLU()
#         # fc2为全连接层，128为输入通道数，2为输出通道数
#         self.fc2 = nn.Linear(128, 2)
#         self.dropout = nn.Dropout(0.3)
        
#         # 卷积层初始化
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 # 随机初始化卷积核权重
#                 nn.init.xavier_uniform_(m.weight)
#                 # 打印卷积核权重
#                 print("Conv1d Weight is :")
#                 print(m.weight)


#     def forward(self, x):
#         x = x.view(x.size(0), 1, -1)
#         inputs_size = x.size(0)
#         x = self.conv1(x)
#         # x = self.dropout(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         x = self.pool2(x)

#         # Dumped
#         # x = x.unsqueeze(0).unsqueeze(0)
#         # x = self.conv3(x)
#         # return x


#         x = x.view(-1, 64 * 2666)
#         x = self.fc1(x)
#         x = self.dropout(x)
#         x = self.relu3(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         return x

import torch
import torch.nn as nn
from torchsummary import summary  # 查看网络结构

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu2(out)
        return out

class TCNN0(nn.Module):
    def __init__(self):
        super(TCNN0, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        
        self.residual_block = ResidualBlock(32, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.fc1 = nn.Linear(64 * 2666, 128)
        self.dropout = nn.Dropout(0.3)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.3)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.residual_block(x)
        
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 2666)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

if __name__ == '__main__':
    # 接收网络模型
    model = TCNN0()
    # print(model)
    # 查看网络参数量，不需要指定输入特征图像的batch维度
    # stat(model, input_size=(1, 1, 32000))

    # 查看网络结构及参数
    summary(model, input_size=[(1, 1, 32000)], device='cpu')