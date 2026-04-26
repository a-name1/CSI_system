import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ✅ 你提供的标签映射（已加入）
LABEL_MAP = {
    "walking": 0,
    "seated-breathing": 1,
    "jumping": 2,
    "wavinghand": 3,
    "running": 4
}

# 你现成的绘图函数（完全不动）
def save_step_dual(amp, phase, step_name, filename):
    output_dir = Path("./pipeline_steps_detailed")
    output_dir.mkdir(exist_ok=True)
    plt.rcParams['font.family'] = 'serif'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=150)
    ax1.plot(amp.T, lw=0.6)
    ax1.set_title(f"{step_name} - Amplitude")
    ax1.set_ylabel("Magnitude")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax2.plot(phase.T, lw=0.6)
    ax2.set_title(f"{step_name} - Phase")
    ax2.set_ylabel("Phase (rad)")
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.suptitle(f"Pipeline Visualization: {step_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()
    print(f"--- [Exported] {filename} ---")

# ------------------------------------------------------------------------------
# 从 npz 加载数据 + 按设备/动作画图
# ------------------------------------------------------------------------------
def plot_from_npz(npz_path="all_csi_samples.npz"):
    # 1. 加载之前保存的 npz 文件
    data = np.load(npz_path, allow_pickle=True)
    all_samples = data["samples"].tolist()

    # 2. 自动反转标签：数字 → 动作名
    label_reverse = {v: k for k, v in LABEL_MAP.items()}

    # 3. 分组处理
    df = pd.DataFrame(all_samples)

    # 4. 按设备分组
    for dev, dev_group in df.groupby("device"):
        print(f"\n📱 设备：{dev}")

# 5. 按动作标签分组
    for label_id, group in dev_group.groupby("label"):
        act_name = label_reverse[label_id]
        print(f"  🎬 动作：{act_name} (label={label_id})")

        amp_list = []
        phase_list = []

        # 遍历每个样本
        for _, row in group.iterrows():
            csi = row["raw_feature"]  # (Nt, Nr, sc, samples)

            # ==================== 幅度 ====================
            amp = np.abs(csi)
            # 去掉异常尖峰（99% 分位截断，让图更干净）
            amp = np.clip(amp, 0, np.percentile(amp, 99))
            # 对 天线×子载波 取平均 → 变成 (1, samples)
            amp = amp.reshape(-1, amp.shape[-1]).mean(axis=0, keepdims=True)

            # ==================== 相位（解缠绕 + 去均值） ====================
            phase = np.angle(csi)
            phase = np.unwrap(phase, axis=-1)  # 解卷绕 ✅
            phase = phase - np.mean(phase, axis=-1, keepdims=True)  # 去直流偏移
            # 对 天线×子载波 取平均
            phase = phase.reshape(-1, phase.shape[-1]).mean(axis=0, keepdims=True)

            amp_list.append(amp)
            phase_list.append(phase)

        # 拼接所有样本
        amp_all = np.concatenate(amp_list, axis=1)
        phase_all = np.concatenate(phase_list, axis=1)

        # 绘图（只画一条均值曲线，超级干净）
        save_step_dual(
            amp_all,
            phase_all,
            step_name=f"{dev}_{act_name}",
            filename=f"{dev}_{act_name}.png"
        )

# ------------------------------------------------------------------------------
# 🔥 直接运行这里
# ------------------------------------------------------------------------------
plot_from_npz("/root/CSI-Bench-Real-WiFi-Sensing-Benchmark/CSI_system/saved_samples/AmazonEchoSpot.npz")
plot_from_npz("/root/CSI-Bench-Real-WiFi-Sensing-Benchmark/CSI_system/saved_samples/AmazonPlug.npz")