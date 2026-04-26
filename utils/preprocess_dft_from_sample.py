import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

# 导入你的核心组件
from config_manager import ConfigManager
from pipeline_executor import PreprocessPipeline
from data_manager import LocalFileCacheManager
from preprocess_ops import *
from config_manager import AblationConfig
from preprocess_ops import FREQ_TABLES
# =========================
# Doppler统一抽象（替代 PCA + STFT）
# =========================
class DopplerExtractor:
    def __init__(self, fs=100, nperseg=64, noverlap=48):
        self.dfs = DFSExtractionOp(fs=fs, nperseg=nperseg, noverlap=noverlap)
        self.norm = RobustScaleOp()

    def transform(self, x):
        # x: [K, T]
        spec = self.dfs.transform(x)
        spec = self.norm.transform(spec)

        # Δ + ΔΔ（增强动态信息）
        d1 = np.diff(spec, axis=1, prepend=spec[:, :1])
        d2 = np.diff(d1, axis=1, prepend=d1[:, :1])

        d1 = self.norm.transform(d1)
        d2 = self.norm.transform(d2)

        return np.stack([spec, d1, d2], axis=0)  # [3, F, T]

def main():

    # ======================
    # 1. 路径配置
    # ======================
    samples_root = "/root/CSI_system/sample_cross_dev"
    cache_root = "/root/CSI_system/ablation_processed_features_cross_dev_dft"

    target_users = ["U02"]
    target_envs = ["E01"]
    target_devices = ["AmazonEchoSpot", "AmazonPlug"]

    # ======================
    # 2. Ablation Config（你给的版本）
    # ======================
    all_configs = [
        AblationConfig(
            name="Full_Path_A_WLS",
            op_switches={
                "conjugate": False,
                "denoise": True,
                "agc": True,
                "wls": True,
                "resample": False,
                "spline": True
            }
        ),
        AblationConfig(
            name="Full_Path_B_CONJ",
            op_switches={
                "conjugate": True,
                "denoise": True,
                "agc": True,
                "wls": False,
                "resample": False,
                "spline": True
            }
        ),
        AblationConfig(
            name="Ablation_No_WLS",
            op_switches={
                "conjugate": False,
                "denoise": True,
                "agc": True,
                "wls": False,
                "resample": False,
                "spline": True
            }
        ),
        AblationConfig(
            name="Baseline_Raw",
            op_switches={
                "conjugate": False,
                "denoise": False,
                "agc": False,
                "wls": False,
                "resample": False,
                "spline": False
            }
        ),
    ]

    # ======================
    # 3. 算子初始化（已拆干净）
    # ======================
    mimo = MIMOCombineOp(mode="mrc")
    conj = ConjugateCorrelationOp(ref_ant_idx=0)

    hampel = HampelFilterOp(k=9)
    sg = SavitzkyGolayOp()
    agc = AGCOp()
    wls = EnhancedWLSPhaseOp()

    align = DimAlignOp(model_input_T=64, model_input_F=32)

    doppler = DopplerExtractor()

    # ======================
    # 4. 文件筛选
    # ======================
    file_pattern = re.compile(r"^(U\d+)_(E\d+)_(.+)\.npz$")

    all_files = [f for f in os.listdir(samples_root) if f.endswith(".npz")]
    selected = []

    for f in all_files:
        m = file_pattern.match(f)
        if m:
            u, e, dev = m.groups()
            if u in target_users and e in target_envs and dev in target_devices:
                selected.append((f, u, e, dev))

    print(f"🎯 selected files: {len(selected)}")

    # ======================
    # 5. 主循环
    # ======================
    for cfg in all_configs:

        print(f"\n🚀 Config: {cfg.name}")
        out_dir = os.path.join(cache_root, cfg.name)
        os.makedirs(out_dir, exist_ok=True)

        for file, u, e, dev in tqdm(selected):

            data = np.load(os.path.join(samples_root, file), allow_pickle=True)
            samples = data["samples"]

            feats, labels = [], []

            for s in samples:

                raw = s["raw_feature"]  # [Nt, Nr, K, T]
                nt, nr, k, t = raw.shape
                total = nt * nr

                # ======================
                # Step 1: 空间处理
                # ======================
                if cfg.op_switches["conjugate"] and total > 1:
                    x = conj.transform(raw)
                    x = np.mean(x, axis=(0, 1))
                    x = np.squeeze(x)
                    strategy = "CONJ"
                elif total > 1:
                    x = mimo.transform(raw)
                    strategy = "MIMO"
                else:
                    x = raw.squeeze((0, 1))
                    strategy = "SISO"

                # ======================
                # Step 2: 幅度相位拆分
                # ======================
                amp = np.abs(x)
                phase = np.angle(x)

                # ======================
                # Step 3: Denoise
                # ======================
                if cfg.op_switches["denoise"]:
                    amp = hampel.transform(amp)
                    amp = sg.transform(amp)

                # ======================
                # Step 4: AGC
                # ======================
                if cfg.op_switches["agc"]:
                    amp = agc.transform(amp)

                # ======================
                # Step 5: WLS（只修 phase）
                # ======================
                if cfg.op_switches["wls"] and strategy != "CONJ":
                    phase = wls.transform((phase, amp), device_type="AUTO")

                # ======================
                # Step 6: 统一 Doppler（替代 PCA + STFT）
                # ======================
                x_clean = np.mean(amp, axis=0)  # [K, T]

                doppler_feat = doppler.transform(x_clean)

                # ======================
                # Step 7: 维度对齐
                # ======================
                feat = align.transform(doppler_feat)

                feats.append(feat)
                labels.append(s["label"])

            # ======================
            # Save
            # ======================
            np.savez_compressed(
                os.path.join(out_dir, f"{u}_{e}_{dev}.npz"),
                x=np.array(feats),
                y=np.array(labels),
                cfg=cfg.name
            )

            del feats, labels, samples
            gc.collect()

    print("✅ Done")


if __name__ == "__main__":
    main()