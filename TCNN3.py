import torch
import torch.nn as nn
from torchstat import stat  # 查看网络参数
from torchsummary import summary  # 查看网络结构

'''
    建立模型
    (batch_size,input_channels,step)
    两秒的音频，step=2*16000=32000
'''


class TCNN3(nn.Module):
    def __init__(self):
        super(TCNN3, self).__init__()  # 继承父类的初始化方法
        # b*1*32000 -> b*32*10615
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9, stride=3, padding=2, dilation=20)
        # b*32*10615 -> b*32*10615
        self.bn1 = nn.BatchNorm1d(32)
        # b*32*10615 -> b*32*10615
        self.relu1 = nn.ReLU()
        # b*32*10615 -> b*32*5037
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        # b*32*5037 -> b*64*5037
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3)
        # b*64*5037 -> b*64*5037
        self.bn2 = nn.BatchNorm1d(64)
        # b*64*5037 -> b*64*5037
        self.relu2 = nn.ReLU()
        # b*64*5037 -> b*64*5304
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=1)
        # b*64*5304 -> b*128*2648
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=5)
        # b*128*2648 -> b*128*2648
        self.bn3 = nn.BatchNorm1d(128)
        # b*128*2648 -> b*128*2648
        self.relu3 = nn.ReLU()
        # b*128*2648 -> b*128*2646
        self.pool3 = nn.AvgPool1d(kernel_size=3, stride=1)
        # b*128*2646 -> b*128*2642
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=2)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()
        # b*128*2642 -> b*128*2641
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=1)
        # b*128*2641 -> b*128*1319
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=3)
        self.bn5 = nn.BatchNorm1d(128)
        self.relu5 = nn.ReLU()
        # b*128*1319 -> b*128*1318
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=1)
        # b*128*1318 -> b*128*1316
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=2)
        self.bn6 = nn.BatchNorm1d(128)
        self.relu6 = nn.ReLU()
        # b*128*1316 -> b*128*658
        self.pool6 = nn.MaxPool1d(kernel_size=2, stride=2)
        # b*128*658 -> b*128*327
        self.conv7 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=3)
        self.bn7 = nn.BatchNorm1d(128)
        self.relu7 = nn.ReLU()
        # b*128*327 -> b*128*163
        self.pool7 = nn.MaxPool1d(kernel_size=2, stride=2)
        # b*128*163 -> b*128*161
        self.conv8 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=2)
        self.bn8 = nn.BatchNorm1d(128)
        self.relu8 = nn.ReLU()
        # b*128*161 -> b*128*80
        self.pool8 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 80, 128)
        self.dropout = nn.Dropout(0.3)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.xavier_uniform_(m.weight)

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
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.pool6(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.pool7(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)
        x = self.pool8(x)
        x = x.view(-1, 128 * 80)
        x = self.dropout(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    model = TCNN3()
    x = torch.randn(5, 1, 32000)
    y = model(x)
    print(y)
