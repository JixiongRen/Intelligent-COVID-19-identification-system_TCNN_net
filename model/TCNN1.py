import torch
import torch.nn as nn
from torchstat import stat  # 查看网络参数
from torchsummary import summary  # 查看网络结构

'''
    建立模型
    (batch_size,input_channels,step)
    两秒的音频，step=2*16000=32000
'''


class TCNN1(nn.Module):
    def __init__(self):
        super(TCNN1, self).__init__()  # 继承父类的初始化方法
        # b*1*32000 -> b*32*4549
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9, stride=7, padding=1, dilation=20) # padding = (kernel_size-stride)/2
        # b*32*4549 -> b*32*4549
        self.bn1 = nn.BatchNorm1d(32)
        # b*32*4549 -> b*32*4549
        self.relu1 = nn.ReLU()
        # b*32*4549 -> b*32*1515
        self.pool1 = nn.MaxPool1d(kernel_size=7, stride=3)
        # b*32*1515 -> b*64*731
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=3, dilation=30)
        # b*64*731 -> b*64*731
        self.bn2 = nn.BatchNorm1d(64)
        # b*64*731 -> b*64*731
        self.relu2 = nn.ReLU()
        # b*64*731 -> b*64*182
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        # b*64*182 -> b*128*90
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=1, dilation=5)
        # b*128*90 -> b*128*90
        self.bn3 = nn.BatchNorm1d(128)
        # b*128*90 -> b*128*90
        self.relu3 = nn.ReLU()
        # b*128*90 -> b*128*44
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=2)
        self.fc1 = nn.Linear(128 * 44, 128)
        self.dropout = nn.Dropout(0.3)
        self.relu3 = nn.ReLU()
        # fc2为全连接层，128为输入通道数，2为输出通道数
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.3)
        
        # 卷积层初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # 随机初始化卷积核权重
                nn.init.xavier_uniform_(m.weight)
                # 打印卷积核权重
                print("Conv1d Weight is :")
                print(m.weight)


    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        inputs_size = x.size(0)
        x = self.conv1(x)
        # x = self.dropout(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Dumped
        # x = x.unsqueeze(0).unsqueeze(0)
        # x = self.conv3(x)
        # return x


        x = x.view(-1, 128 * 44)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


