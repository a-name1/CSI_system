import json
import os
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class AblationConfig:
    """消融实验配置数据类"""
    name: str
    op_switches: Dict[str, bool]  # 算子开关，对应op_name
    desc: str = ""
    
class ConfigManager:
    def __init__(self, config_file: str = None):
        """
        配置管理器
        :param config_file: 外部JSON配置文件路径，不传则用默认配置
        """
        self.configs: List[AblationConfig] = []
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
        else:
            self._load_default_config()

    def _load_default_config(self):
        """默认消融配置，与代码解耦"""
        default_configs = [
            {
                "name": "Full_Path_A_WLS",
                "desc": "完整流水线：MRC + WLS (恢复绝对相位)",
                "op_switches": {"conjugate": False, "denoise": True, "agc": True, "wls": True, "resample": True, "spline": True, "pca": True}
            },
            {
                "name": "Full_Path_B_CONJ",
                "desc": "完整流水线：Conjugate (物理差分去噪)",
                "op_switches": {"conjugate": True, "denoise": True, "agc": True, "wls": False, "resample": True, "spline": True, "pca": True}
            },      
            {
                "name": "Ablation_No_Denoise",
                "desc": "缺失环节：未进行 Hampel 与 S-G 滤波 (高频噪声干扰)",
                "op_switches": {"conjugate": False, "denoise": False, "agc": True, "wls": True, "resample": True, "spline": True, "pca": True}
            },
            {
                "name": "Ablation_No_AGC",
                "desc": "缺失环节：未进行幅度校准 (远近效应对分类的影响)",
                "op_switches": {"conjugate": False, "denoise": True, "agc": False, "wls": True, "resample": True, "spline": True, "pca": True}
            },
            {
                "name": "Ablation_No_WLS",
                "desc": "缺失环节：未进行相位去斜 (验证 WLS 对 SFO 的压制作用)",
                "op_switches": {"conjugate": False, "denoise": True, "agc": True, "wls": False, "resample": True, "spline": True, "pca": True}
            },
            {
                "name": "Ablation_No_Resample",
                "desc": "缺失环节：未进行 Kaiser 重采样 (验证非均匀采样对 STFT 的影响)",
                "op_switches": {"conjugate": False, "denoise": True, "agc": True, "wls": True, "resample": False, "spline": True, "pca": True}
            },
            {
                "name": "Ablation_No_Spline",
                "desc": "缺失环节：未进行频率对齐 (跨设备频率一致性测试)",
                "op_switches": {"conjugate": False, "denoise": True, "agc": True, "wls": True, "resample": True, "spline": False, "pca": True}
            },
            {
                "name": "Ablation_No_PCA",
                "desc": "缺失环节：直接输入全子载波 STFT (测试维度压缩的必要性)",
                "op_switches": {"conjugate": False, "denoise": True, "agc": True, "wls": True, "resample": True, "spline": True, "pca": False}
            },
            {
                "name": "Baseline_Raw",
                "desc": "基准对比：仅进行基本格式化，无任何物理校准",
                "op_switches": {"conjugate": False, "denoise": False, "agc": False, "wls": False, "resample": False, "spline": False, "pca": False}
            }
        ]
        self.configs = [AblationConfig(**cfg) for cfg in default_configs]

    def _load_from_file(self, config_file: str):
        """从外部JSON文件加载配置"""
        with open(config_file, 'r', encoding='utf-8') as f:
            raw_configs = json.load(f)
        self.configs = [AblationConfig(**cfg) for cfg in raw_configs]
        return self.configs

    def get_all_configs(self) -> List[AblationConfig]:
        """获取所有消融配置"""
        return self.configs

    def get_config_by_name(self, name: str) -> AblationConfig:
        """按名称获取配置"""
        for cfg in self.configs:
            if cfg.name == name:
                return cfg
        raise ValueError(f"未找到名称为{name}的消融配置")