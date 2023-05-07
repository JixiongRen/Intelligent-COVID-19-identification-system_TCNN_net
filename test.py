import torch.nn as nn
import torch

import torch.nn as nn

# 定义卷积层
conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 188), stride=188)

# conv_transpose = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=8, stride=2, padding=0)
# 输入张量x的大小为64*2666
x = torch.randn(64, 2666)

# 将x转换为(batch, channels, height, width)的形状
x = x.unsqueeze(0).unsqueeze(0)

# 将x输入到卷积层conv中，得到输出张量y


# y = conv_transpose(y)
# y的大小为(batch, channels, height, width)=(1, 64, 7, 7)
x = x.reshape(64,7,7)

print(y.shape)