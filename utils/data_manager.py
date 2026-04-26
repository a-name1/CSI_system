import os
import numpy as np
from sklearn.model_selection import train_test_split
from ablation_base import BaseCacheManager, BaseDatasetSplitter
from typing import Tuple, Any

# ====================== 本地文件缓存实现 ======================
class LocalFileCacheManager(BaseCacheManager):
    def __init__(self, cache_dir: str = "./ablation_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.npy")

    def exists(self, key: str) -> bool:
        return os.path.exists(self._get_cache_path(key))

    def save(self, key: str, data: Any) -> None:
        np.save(self._get_cache_path(key), data)

    def load(self, key: str) -> Any:
        return np.load(self._get_cache_path(key))

# ====================== 标准70/15/15数据集划分实现 ======================
class StandardDatasetSplitter(BaseDatasetSplitter):
    def __init__(self, save_dir: str = "./ablation_datasets", random_seed: int = 42):
        self.save_dir = save_dir
        self.random_seed = random_seed
        os.makedirs(save_dir, exist_ok=True)
        np.random.seed(random_seed)

    def _get_split_path(self, config_name: str, split: str) -> str:
        return os.path.join(self.save_dir, f"{config_name}_{split}.npz")

    def split(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 70%训练集，30%剩余
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels, test_size=0.3, random_state=self.random_seed, stratify=labels
        )
        # 15%验证集，15%测试集
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_seed, stratify=y_temp
        )
        return X_train, y_train, X_val, y_val, X_test, y_test

    def save(self, save_dir: str, config_name: str, **datasets) -> None:
        for split_name, data in datasets.items():
            save_path = self._get_split_path(config_name, split_name)
            np.savez(save_path, **{split_name: data})

    def load(self, save_dir: str, config_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_data = np.load(self._get_split_path(config_name, "train") + ".npz")
        val_data = np.load(self._get_split_path(config_name, "val") + ".npz")
        test_data = np.load(self._get_split_path(config_name, "test") + ".npz")
        return (
            train_data["X_train"], train_data["y_train"],
            val_data["X_val"], val_data["y_val"],
            test_data["X_test"], test_data["y_test"]
        )