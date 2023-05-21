import torch.utils.data.distributed 

'定义转化后的模型名称'
model_TCNN_pt = 'tcnn4_for_android.pt'

'加载pytorch模型'
model_TCNN_pth = torch.load('pth_files/model.pth')

'使模型在GPU上运行'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_TCNN_pth.to(device)
model_TCNN_pth.eval()

'定义输入张量的大小'
input_tensor = torch.randn(1, 1, 1, 32000)

'转化模型并存储'
model_TCNN_pth = torch.jit.trace(model_TCNN_pth, input_tensor) # type: ignore
model_TCNN_pth.save(model_TCNN_pt) # type: ignore
