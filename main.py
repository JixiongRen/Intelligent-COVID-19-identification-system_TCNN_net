import torch
import torchaudio
import os
from pathlib import Path
from torch.utils.data import Dataset
import torchaudio.transforms as transforms


# 1. 加载音频数据集
class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir  # 数据集所在目录
        self.labels = ["positive", "negative"]  # 获取标签列表
        self.label2idx = {label: i for i, label in enumerate(self.labels)}  # 该行意思为：将标签转换为索引，例如：{"positive": 0,
        # "negative": 1}

        # 加载所有音频文件并获取其标签
        self.audio_files = []  # 存储音频文件路径
        self.targets = []  # 存储音频文件对应的标签
        for label in self.labels:  # 遍历标签列表
            label_dir = os.path.join(data_dir, label)  # 获取标签对应的目录，例如：data_dir/positive
            for audio_file in os.listdir(label_dir):  # 遍历该目录下的所有音频文件
                audio_file_path = os.path.join(label_dir, audio_file)  # 获取音频文件的路径, 例如：data_dir/positive/XXXXX.wav
                self.audio_files.append(audio_file_path)  # 将音频文件路径添加到列表中
                self.targets.append(self.label2idx[label])  # 将音频文件对应的标签添加到列表中, 例如：data_dir/positive/XXXXX.wav对应的标签为0

    def __len__(self):
        return len(self.audio_files)  # 返回数据集的长度

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]  # 获取音频文件路径
        waveform, sample_rate = torchaudio.load(audio_file)  # 加载音频文件
        # 进行预处理，例如将音频数据裁剪或填充到指定长度等
        waveform = torchaudio.transforms.PadTrim(32000)(waveform)  # 将音频数据裁剪或填充到指定长度

        # 将标签转换为PyTorch张量并返回数据和标签
        target = self.targets[idx]  # 获取音频文件对应的标签
        target = torch.tensor(target, dtype=torch.long)  # 将标签转换为PyTorch张量
        return waveform, target  # 返回数据和标签
