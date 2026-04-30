# ===================== 论文实验统一调用入口 =====================
import os
import torch
from Class_MMD_CNN_BiGRU_Atten_train import run_experiment
from pathlib import Path

import sys

# ========= 1. 路径挂载 =========
current_dir = Path(__file__).resolve().parent 
root_dir = current_dir.parent 
utils_dir = root_dir / "utils"

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(utils_dir) not in sys.path:
    sys.path.append(str(utils_dir))


from utils.config_manager import ConfigManager

# ===================== 路径配置 =====================
CFG_JSON = "/root/CSI_system/config/ablation_configs_dft.json"

# 数据路径（你之前已经统一好了）
BASE_DATA_DIR = "/root/CSI_system/ablation_processed_features_cross_dev"

# 输出路径
OUTPUT_DIR = "/root/CSI_system/paper_results/TMC_STYLE"

# ===================== 实验模式配置 =====================
MODES = {
    # 同域（训练=测试设备）
    "in_domain": {
        "train_devices": ["AmazonPlug"],
        "test_devices": ["AmazonPlug"]
    },

    # 跨域（DA）
    "cross_domain": {
        "train_devices": ["AmazonEchoShow8"],
        "test_devices": ["AmazonPlug"]
    },

    # Leave-one-device-out（DG）
    "dg": {
        "train_devices": ["AmazonEchoShow8", "AmazonEchoSpot"],
        "test_devices": ["AmazonPlug"]
    }
}

# ===================== 主函数 =====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载配置（Ablation + Full pipeline）
    configs = ConfigManager()._load_from_file(CFG_JSON)

    # 初始化实验执行器
    runner = run_experiment(
        base_dir=BASE_DATA_DIR,
        save_root=OUTPUT_DIR,
        device=device,
        label_names=["行走", "坐姿呼吸", "跳跃", "挥手", "跑步"]
    )

    # ===================== 逐模式执行 =====================
    for mode_name, mode_cfg in MODES.items():
        print(f"\n{'='*80}")
        print(f"🚀 Running Mode: {mode_name}")
        print(f"{'='*80}")

        runner.run_mode(
            mode_name=mode_name,
            configs=configs,
            train_devices=mode_cfg["train_devices"],
            test_devices=mode_cfg["test_devices"]
        )

    print("\n✅ All experiments completed. Results saved to:", OUTPUT_DIR)


# ===================== 入口 =====================
if __name__ == "__main__":
    main()