import torch.utils.data.distributed 
import numpy as np

'定义转化后的模型名称'
model_TCNN_pt = 'tcnn4_for_android.pt'

'加载pytorch模型'
model_TCNN_pth = torch.load('pth_files/2023-05-22-20-01model_pths/best_in_5folds-val_acc=0.6421052813529968-best_val_acc-model.pth')

'使模型在GPU上运行'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_TCNN_pth.to(device)
model_TCNN_pth.eval()

'定义输入张量的大小'
input_tensor = torch.randn(1, 1, 1, 32000)

'转化模型并存储'
model_TCNN_pth = torch.jit.trace(model_TCNN_pth, input_tensor) # type: ignore
model_TCNN_pth.save(model_TCNN_pt) # type: ignore


# from ModuleFunctions import toolsFun
# cnf_matrix = np.array([[3021, 1446], [1531, 2962]])
# toolsFun.plot_confusion_matrix(cnf_matrix,
#                                classes=['negative', 'positive'],
#                                normalize=False,
#                                title='Normalized confusion matrix',
#                                path='new_cm.svg')