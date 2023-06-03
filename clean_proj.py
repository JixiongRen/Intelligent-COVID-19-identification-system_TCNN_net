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

print("ROC Figure Names:")
print(roc_fig_name)
print("ROC Timestamps:")
print(roc_timestamp)

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

# 文件夹路径
pth_folder = "pth_files/"
figs_folder = "figs/"
train_info_folder = "train_info/"
hyperParam_folder = "hyperParam/"

# 检查并删除不符合要求的子文件夹
check_and_delete_folders(pth_folder, roc_timestamp)
check_and_delete_folders(figs_folder, roc_timestamp)

# 检查并删除不符合要求的文件
check_and_delete_files(train_info_folder, roc_timestamp)
check_and_delete_files(hyperParam_folder, roc_timestamp)
