import os
import json
import h5py
import pandas as pd
import numpy as np

class TaskDatasetLoader:
    def __init__(self, task_root="/root/CSI_system/TaskName"):
        self.root = task_root
        self.meta_path = os.path.join(task_root, "metadata/sample_metadata.csv")
        self.label_path = os.path.join(task_root, "metadata/label_mapping.json")
        
        # 加载元数据 + 标签
        self.metadata = pd.read_csv(self.meta_path)
        with open(self.label_path, 'r') as f:
            self.label_map = json.load(f)
        self.idx2label = {v: k for k, v in self.label_map.items()}

    def load_csi(self, file_path):
        """读取单个CSI样本h5文件"""
        full_path = os.path.join(self.root, file_path)
        with h5py.File(full_path, 'r') as f:
            csi = f['csi'][:]
            label = f.attrs['label']
        return csi, label

    def get_samples_by_ids(self, sample_ids):
        """根据sample_id加载批量数据（LODO专用）"""
        X, y = [], []
        df = self.metadata[self.metadata['sample_id'].isin(sample_ids)]
        
        for _, row in df.iterrows():
            csi, label = self.load_csi(row['file_path'])
            X.append(csi)
            y.append(label)
        
        # 统一数据形状（适配模型输入）
        X = np.array(X).astype(np.float32)
        y = np.array(y).astype(np.int64)
        return X, y

    def load_split(self, split_name):
        """加载划分文件：train_id/val_id/test_cross_user等"""
        split_path = os.path.join(self.root, f"splits/{split_name}.json")
        with open(split_path, 'r') as f:
            return json.load(f)['sample_ids']