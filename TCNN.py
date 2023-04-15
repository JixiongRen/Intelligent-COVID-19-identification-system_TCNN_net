import torch.nn as nn
from torchstat import stat  # 查看网络参数
from torchsummary import summary  # 查看网络结构


# 建立模型
class TCNN(nn.Module):
    def __init__(self):
        super(TCNN, self).__init__()  # 继承父类的初始化方法
        # conv1为卷积层，in_channels为输入通道数，out_channels为输出通道数，kernel_size为卷积核大小，stride为步长，padding为补零
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # bn1为批归一化层，32为输入通道数
        self.bn1 = nn.BatchNorm1d(32)
        # relu1为激活函数层，这里使用的是ReLU函数，激活层一般放在卷积层和全连接层之后
        self.relu1 = nn.ReLU()
        # pool1为池化层，这里使用的是最大池化，kernel_size为池化核大小，stride为步长
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        # conv2为卷积层，in_channels为输入通道数，out_channels为输出通道数，kernel_size为卷积核大小，stride为步长，padding为补零
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # bn2为批归一化层，64为输入通道数
        self.bn2 = nn.BatchNorm1d(64)
        # relu2为激活函数层，这里使用的是ReLU函数，激活层一般放在卷积层和全连接层之后
        self.relu2 = nn.ReLU()
        # pool2为池化层，这里使用的是最大池化，kernel_size为池化核大小，stride为步长
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        # fc1为全连接层，64 * 13为输入通道数，128为输出通道数
        # self.fc1 = nn.Linear(64 * 13, 128)
        # fc1为全连接层，64 * 5333为输入通道数，128为输出通道数
        self.fc1 = nn.Linear(64 * 2666, 128)
        self.relu3 = nn.ReLU()
        # fc2为全连接层，128为输入通道数，2为输出通道数
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.view(1, 1, -1)
        inputs_size = x.size(0)
        x = self.conv1(x)  # 输入：batchsize*1*32000，输出：batchsize*32*32000
        x = self.bn1(x)  # 输入：batchsize*32*32000，输出：batchsize*32*32000，批量归一化
        x = self.relu1(x)  # 输入：batchsize*32*32000，输出：batchsize*32*32000， 激活层，提高模型表达能力
        x = self.pool1(x)  # 输入：batchsize*32*32000，输出：batchsize*32*10666
        x = self.conv2(x)  # 输入：batchsize*32*10666，输出：batchsize*64*10666
        x = self.bn2(x)  # 输入：batchsize*64*10666，输出：batchsize*64*10666, 批量归一化
        x = self.relu2(x)  # 输入：batchsize*64*10666，输出：batchsize*64*10666， 激活层，提高模型表达能力
        x = self.pool2(x)  # 输入：batchsize*64*10666，输出：batchsize*64*2666，池化层，减少参数量，计算过程为：(10666-4)/4+1=2666
        # x = x.view(-1, 64 * 13)  # 输入：batchsize*64*5333，输出：batchsize*64*5333，将数据展开，方便全连接层处理，处理过程
        x = x.view(inputs_size, -1)  # 输入：batchsize*64*2666，输出：batchsize*64*2666，将数据展开，方便全连接层处理，展平的过程可以表示为：batchsize
        # *64*2666->batchsize*(64*2666)
        x = self.fc1(x)  # 输入：batchsize*64*2666，输出：batchsize*128，全连接层，将数据从64*2666维度转换为128维度
        x = self.relu3(x)  # 输入：batchsize*128，输出：batchsize*128， 激活层，提高模型表达能力
        x = self.fc2(x)  # 输入：batchsize*128，输出：batchsize*2，全连接层，将数据从128维度转换为2维度
        return x


if __name__ == '__main__':
    # 接收网络模型
    model = TCNN()
    # print(model)

    # 查看网络参数量，不需要指定输入特征图像的batch维度
    stat(model, input_size=(1, 32000, 1))


