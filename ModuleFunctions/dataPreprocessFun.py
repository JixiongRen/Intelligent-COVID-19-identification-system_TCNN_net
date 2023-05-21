import librosa
import soundfile as sf
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


"""
# wave -> np.array
"""
def preprocess_data(audio_file_path):
    '''将音频数据转化为numpy数组'''
    # 设置参数
    sr = 16000
    duration = 2
    # 初始化变量
    data = []  # 存放音频数据
    # 加载音频文件
    filepath = audio_file_path
    y, sr = sf.read(filepath)
    # 转化为单声道
    if len(y.shape) > 1:
        y = librosa.to_mono(y)
    # 修改为16kHz采样率
    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    # 归一化
    y = librosa.util.normalize(y)  # 归一化
    # 剪切或补 0
    if len(y) < 32000:
        y = np.pad(y, (32000 - len(y), 0), mode='constant')
    else:
        y = y[0:32000]
    # 将数据添加到列表中，包含标签和数据
    data.append(y.reshape(1, -1))  # 将数据转化为 1 行，-1 列的形式
    # 转化为 numpy 数组
    data = np.array(data)
    return data


"""
建立文件路径与标签的对应关系
"""
def path2target(audio_dir):
    '''建立音频路径-标签之关系'''
    # 初始化变量
    filepath_array = []  # 存放音频数据
    labels = []
    # 得到音频文件列表
    files = os.listdir(audio_dir)
    # 处理每个音频文件
    for file in files:
        # 加载音频文件
        filepath = os.path.join(audio_dir, file)
        audios = os.listdir(filepath)
        for audio in audios:
            audiofilepath = os.path.join(filepath, audio)
            filepath_array.append(audiofilepath)
            if 'negative' in audiofilepath:
                labels.append(0)
            else:
                labels.append(1)
    return filepath_array, labels

class MyDataset(Dataset):
    def __init__(self, data_path, data_label):
        self.data_path = data_path
        self.data_label = data_label

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        # 加载数据
        data = self.data_path[idx]
        # print(data)
        # 进行预处理
        data = preprocess_data(data)
        # 获取标签
        label = self.data_label[idx]
        # 转换为 tensor
        data = torch.tensor(data).float()
        label = torch.tensor(label).float()
        return data, label