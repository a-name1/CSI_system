from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any, Tuple

# ====================== 算子抽象基类：所有预处理算子必须实现这个接口 ======================
class BasePreprocessOp(ABC):
    @property
    @abstractmethod
    def op_name(self) -> str:
        """算子唯一名称，对应配置里的开关key"""
        pass

    @abstractmethod
    def fit(self, data: np.ndarray, **kwargs) -> None:
        """训练阶段拟合算子参数（如PCA、均值方差）"""
        pass

    @abstractmethod
    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """推理阶段执行算子逻辑"""
        pass

    def fit_transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """拟合+转换，默认实现"""
        self.fit(data, **kwargs)
        return self.transform(data, **kwargs)

# ====================== 缓存抽象基类：可替换不同缓存实现 ======================
class BaseCacheManager(ABC):
    @abstractmethod
    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        pass

    @abstractmethod
    def save(self, key: str, data: Any) -> None:
        """保存缓存"""
        pass

    @abstractmethod
    def load(self, key: str) -> Any:
        """加载缓存"""
        pass

# ====================== 数据集划分抽象基类：可自定义划分规则 ======================
class BaseDatasetSplitter(ABC):
    @abstractmethod
    def split(
        self, 
        features: np.ndarray, 
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        划分数据集
        返回：X_train, y_train, X_val, y_val, X_test, y_test
        """
        pass

    @abstractmethod
    def save(self, save_dir: str, config_name: str, **datasets) -> None:
        """保存划分好的数据集"""
        pass

    @abstractmethod
    def load(self, save_dir: str, config_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """加载划分好的数据集"""
        pass