# ===================== 修复路径 =====================
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
# ====================================================

import os
import gc
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.pipeline_executor import PreprocessPipeline
from utils.config_manager import AblationConfig

# ===================== 配置 =====================
FULL_PATH_A = AblationConfig(
    name="Full_Path_A_MRC_WLS_PCA",
    op_switches={
        "Unwrap": True, "conjugate": False, "wls": True, "agc": True,
        "hampel": True, "SG": True, "static_removal": True,
        "resample": True, "spline": True, "pca": True, "zscore": True
    }
)

ALL_DEVICES = [
    "AmazonEchoPlus", "AmazonEchoShow8", "AmazonEchoSpot",
    "AmazonPlug", "AppleHomePod", "EighttreePlug",
    "GoogleNest", "GoveeSmartPlug", "WyzePlug"
]

RAW_ROOT = "/root/CSI_system/TaskName"
SAVE_ROOT = "/root/CSI_system/LODO_SAFE_PREPROCESSED"
BATCH_SIZE = 8  # 同设备可以放心 batch
# ====================================================

os.makedirs(SAVE_ROOT, exist_ok=True)
metadata = pd.read_csv(f"{RAW_ROOT}/metadata/sample_metadata.csv")

def load_single_sample(row):
    file_path = f"{RAW_ROOT}/{row['file_path']}"
    data = np.load(file_path)
    return data["csi"], data["timestamp"], int(data["label"])

print("🚀 启动【物理正确】LODO 预处理")

# ====================================================
for test_device in tqdm(ALL_DEVICES, desc="📊 LODO 设备进度"):
    print(f"\n========== 测试设备: {test_device} ==========")

    train_df = metadata[metadata["device"] != test_device].reset_index(drop=True)
    test_df  = metadata[metadata["device"] == test_device].reset_index(drop=True)

    # ========== 关键：每台设备独立 Pipeline ==========
    pipelines = {dev: PreprocessPipeline() for dev in ALL_DEVICES}

    # ====================================================
    # Step 1. 各设备分别拟合
    # ====================================================
    print("🔧 Step1: 按设备独立拟合预处理参数")

    for dev in ALL_DEVICES:
        if dev == test_device:
            continue

        dev_df = train_df[train_df["device"] == dev].reset_index(drop=True)
        pipe = pipelines[dev]

        for i in tqdm(range(0, len(dev_df), BATCH_SIZE),
                      desc=f"  拟合 {dev}", leave=False):

            batch = dev_df[i:i+BATCH_SIZE]
            csi_list, time_list = [], []

            for _, row in batch.iterrows():
                csi, t, _ = load_single_sample(row)
                csi_list.append(csi)
                time_list.append(t)

            pipe.process_batch(
                csi_list=csi_list,
                time_list=time_list,
                device_type=dev,
                config=FULL_PATH_A,
                is_training=True
            )

            gc.collect()

    # ====================================================
    # Step 2. 生成训练特征
    # ====================================================
    print("📦 Step2: 生成训练特征")
    train_feat, train_y = [], []

    for dev in ALL_DEVICES:
        if dev == test_device:
            continue

        dev_df = train_df[train_df["device"] == dev].reset_index(drop=True)
        pipe = pipelines[dev]

        for i in tqdm(range(0, len(dev_df), BATCH_SIZE),
                      desc=f"  训练特征 {dev}", leave=False):

            batch = dev_df[i:i+BATCH_SIZE]
            csi_list, time_list, lbl_list = [], [], []

            for _, row in batch.iterrows():
                csi, t, lbl = load_single_sample(row)
                csi_list.append(csi)
                time_list.append(t)
                lbl_list.append(lbl)

            feat = pipe.process_batch(
                csi_list=csi_list,
                time_list=time_list,
                device_type=dev,
                config=FULL_PATH_A,
                is_training=False
            )

            train_feat.append(feat)
            train_y.extend(lbl_list)
            gc.collect()

    train_x = np.concatenate(train_feat, axis=0)
    train_y = np.array(train_y)

    # ====================================================
    # Step 3. 生成测试特征（只用测试设备自己的 pipeline）
    # ====================================================
    print("📦 Step3: 生成测试特征")
    test_feat, test_y = [], []
    pipe = pipelines[test_device]

    for i in tqdm(range(0, len(test_df), BATCH_SIZE),
                  desc="  测试特征", leave=False):

        batch = test_df[i:i+BATCH_SIZE]
        csi_list, time_list, lbl_list = [], [], []

        for _, row in batch.iterrows():
            csi, t, lbl = load_single_sample(row)
            csi_list.append(csi)
            time_list.append(t)
            lbl_list.append(lbl)

        feat = pipe.process_batch(
            csi_list=csi_list,
            time_list=time_list,
            device_type=test_device,
            config=FULL_PATH_A,
            is_training=False
        )

        test_feat.append(feat)
        test_y.extend(lbl_list)
        gc.collect()

    test_x = np.concatenate(test_feat, axis=0)
    test_y = np.array(test_y)

    # ====================================================
    # 保存
    # ====================================================
    save_path = f"{SAVE_ROOT}/{test_device}.npz"
    np.savez_compressed(
        save_path,
        train_x=train_x, train_y=train_y,
        test_x=test_x, test_y=test_y
    )

    print(f"✅ 保存完成: {save_path}")

    del pipelines, train_x, train_y, test_x, test_y
    gc.collect()

print("\n🎉 LODO 预处理全部完成（物理一致 & 无错误）")