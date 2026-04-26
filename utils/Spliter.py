import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

class StandardDatasetSplitter:
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)

    def _parse_sample_id(self, sample_id):
        """严格对齐CSI-Bench样本格式"""
        parts = sample_id.split('_')
        return {
            'type': parts[0],
            'user': parts[1],
            'action': parts[2],
            'env': parts[3],
            'device': parts[4],
            'version': parts[5],
            'number': parts[6]
        }

    def _load_all(self, cache_dir, config_name):
        all_x, all_y = [], []
        all_sample_ids = []
        
        for file in os.listdir(cache_dir):
            if file.startswith(config_name) and file.endswith(".npy"):
                path = os.path.join(cache_dir, file)
                data = np.load(path, allow_pickle=True).item()
                all_x.append(data['x'])
                all_y.append(data['y'])
                all_sample_ids.extend(data['sample_ids'])
        
        X = np.vstack(all_x)
        y = np.concatenate(all_y)
        sample_ids = np.array(all_sample_ids)
        return X, y, sample_ids

    # ==============================
    # 🔥 论文真实划分：严格按设备隔离
    # ==============================
    def load_benchmark_cross_device(self, cache_dir, config_name):
        X, y, sample_ids = self._load_all(cache_dir, config_name)
        
        # 论文官方设备分组（完全固定）
        TEST_DEVICES = {"AmazonPlug", "EchoSpot"}
        
        train_val_mask = []
        test_mask = []
        
        for sid in sample_ids:
            device = self._parse_sample_id(sid)['device']
            if device in TEST_DEVICES:
                test_mask.append(True)
                train_val_mask.append(False)
            else:
                test_mask.append(False)
                train_val_mask.append(True)
        
        train_val_mask = np.array(train_val_mask)
        test_mask = np.array(test_mask)
        
        # 训练集 + 跨设备测试集
        X_train_val = X[train_val_mask]
        y_train_val = y[train_val_mask]
        ids_train_val = sample_ids[train_val_mask]
        
        X_test = X[test_mask]
        y_test = y[test_mask]
        ids_test = sample_ids[test_mask]
        
        # 训练集内部 9:1 划分（论文比例）
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            X_train_val, y_train_val, ids_train_val,
            test_size=0.1, random_state=self.seed, stratify=y_train_val
        )
        
        # 数据部分（用于训练）
        data_split = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test)
        }
        
        # ID部分（用于生成benchmark JSON）
        id_splits = {
            "train_id": ids_train.tolist(),
            "val_id": ids_val.tolist(),
            "test_cross_device": ids_test.tolist()
        }
        
        return data_split, id_splits

    def save_splits(self, id_splits, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for name, ids in id_splits.items():
            with open(os.path.join(save_dir, f"{name}.json"), 'w') as f:
                json.dump(ids, f, indent=2)
        print("✅ 已生成论文标准划分文件")