import torch
print(torch.__version__)  #查看pytorch版本，如未安装Gpu版本，输出一般是torch版本+Cpu，如安装成功输出是torch版本+cuxxx，xxx表示cuda的版本号
print(torch.cuda.is_available()) #如输出是False说明未安装Gpu版本，cuda可用输出True
