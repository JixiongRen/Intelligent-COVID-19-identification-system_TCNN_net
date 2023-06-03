"""_summary_
这是一个清洁、整理项目文件的脚本。考虑到无效训练(因各种原因导致训练未完成)的存在，需要删除此类训练生成的不完全训练产生的残缺日志文件，
包括 train_info, figs, hyperParam, pth_files, ROC_k5, ConfusionMartix_k5. 一般只要生成了五折交叉验证的ROC曲线，我们认为一轮完整
的训练便结束了，所以本脚本提取到ROC_k5文件夹下的时间戳信息，并以此为据对其他文件夹作清洁。本脚本应定期执行以保证项目整洁可读。本脚本不
会误删文件。
"""

import os
import re

folder_path = "ROC_k5"
roc_fig_name = []
roc_timestamp = []

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        match = re.search(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}', filename)
        if match:
            roc_fig_name.append(filename)
            roc_timestamp.append(match.group())

def check_and_delete_folders(folder_path, timestamp_list):
    for subdir in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subfolder_path) and any(timestamp in subdir for timestamp in timestamp_list):
            print(f"Timestamps found in subfolder '{subdir}'")
        else:
            print(f"Deleting subfolder '{subdir}'")
            delete_folder(subfolder_path)

def delete_folder(folder_path):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        for name in dirs:
            dir_path = os.path.join(root, name)
            os.rmdir(dir_path)
            print(f"Deleted directory: {dir_path}")

def check_and_delete_files(folder_path, timestamp_list):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and any(timestamp in filename for timestamp in timestamp_list):
            print(f"Timestamp found in file '{filename}'")
        else:
            print(f"Deleting file '{filename}'")
            os.remove(file_path)

def remove_empty_folders(folder_path):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):  # 检查文件夹是否为空
                os.rmdir(dir_path)
                print(f"Deleted empty folder: {dir_path}")

# 文件夹路径
pth_folder = "pth_files/"
figs_folder = "figs/"
train_info_folder = "train_info/"
hyperParam_folder = "hyperParam/"
cm_k5_folder = "ConfusionMartix_k5/"

# 检查并删除不符合要求的子文件夹
check_and_delete_folders(pth_folder, roc_timestamp)
check_and_delete_folders(figs_folder, roc_timestamp)

# 检查并删除不符合要求的文件
check_and_delete_files(train_info_folder, roc_timestamp)
check_and_delete_files(hyperParam_folder, roc_timestamp)
check_and_delete_files(cm_k5_folder, roc_timestamp)

# 清除空文件夹
remove_empty_folders(pth_folder)
remove_empty_folders(figs_folder)

