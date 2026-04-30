import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.ablation_base import BasePreprocessOp
from utils.preprocess_ops import *
from utils.config_manager import AblationConfig
import matplotlib

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 统一字体大小 =====================
FONT_MAIN = 24
FONT_SUB = 20
FONT_AXIS = 18
FONT_TICK = 14

# 绘图：幅度/相位
def save_step_dual(amp, phase, t_sec, step_name_zh, filename, agc=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    t_len = min(amp.shape[1], len(t_sec))
    time_axis = t_sec[:t_len]
    amp_plot = amp[:, :t_len]
    phase_plot = phase[:, :t_len]

    ax1.plot(time_axis, amp_plot.T, lw=1.0)
    ax1.set_title(f"{step_name_zh} - 幅度", fontsize=FONT_SUB, pad=20)
    ax1.set_xlabel("时间 (秒, s)", fontsize=FONT_AXIS)
    ax1.set_ylabel("归一化信道增益 |H| (AGC校准)" if agc else "幅度 |H| (线性幅值)", fontsize=FONT_AXIS)
    ax1.tick_params(labelsize=FONT_TICK)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(time_axis, phase_plot.T, lw=1.0)
    ax2.set_title(f"{step_name_zh} - 相位", fontsize=FONT_SUB, pad=20)
    ax2.set_xlabel("时间 (秒, s)", fontsize=FONT_AXIS)
    ax2.set_ylabel("相位 ∠H (rad)", fontsize=FONT_AXIS)
    ax2.tick_params(labelsize=FONT_TICK)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle(f"CSI 处理步骤：{step_name_zh}", fontsize=FONT_MAIN, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_dir = Path("./pipeline_steps_detailed")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()
    print(f"已导出：{filename}")

# 绘图：PCA
def save_step_pca(pca_out_list, t_sec, filename="step07_pca.png"):
    sample_pca = np.array(pca_out_list[0])
    new_len = sample_pca.shape[0]
    time_axis = np.linspace(0, t_sec[-1], new_len)
    plt.figure(figsize=(12,7), dpi=150)
    plot_count = min(3, sample_pca.shape[1] if sample_pca.ndim>1 else 1)
    plt.plot(time_axis, sample_pca[:,:plot_count], lw=1.5)
    plt.title("PCA 主成分", fontsize=FONT_MAIN, fontweight='bold')
    plt.xlabel("时间 (s)", fontsize=FONT_AXIS)
    plt.ylabel("幅值", fontsize=FONT_AXIS)
    plt.legend([f"PC{i+1}" for i in range(plot_count)])
    plt.grid(alpha=0.7)
    output_dir = Path("./pipeline_steps_detailed")
    output_dir.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close()

# 绘图：STFT时频图
def save_step_stft(final_feat, filename="step08_final_stft.png", fs=100):
    fig, axes = plt.subplots(1,3,figsize=(20,8),dpi=150,constrained_layout=True)
    n_freq = final_feat.shape[1]
    n_frames = final_feat.shape[2]
    freq_axis = np.linspace(0, fs/2, n_freq)
    time_axis = np.linspace(0, n_frames/fs, n_frames)
    is_phase = "pha" in filename
    chan_num = min(3, final_feat.shape[0])
    plot_data = [np.abs(final_feat[i]) for i in range(chan_num)]
    vmin, vmax = np.min([d.min() for d in plot_data]), np.max([d.max() for d in plot_data])

    for i in range(chan_num):
        im = axes[i].pcolormesh(time_axis, freq_axis, plot_data[i], shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
        axes[i].set_title(f"{'相位' if is_phase else '幅度'} PC{i+1}", fontsize=18)
        axes[i].set_xlabel("时间 (s)", fontsize=16)
        axes[i].set_ylabel("频率 (Hz)", fontsize=16)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02)
    cbar.set_label("幅度")
    plt.suptitle("时频特征图", fontsize=24, fontweight='bold')
    output_dir = Path("./pipeline_steps_detailed")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / filename, bbox_inches='tight')
    plt.close()
    print(f"已保存：{filename}")

# ===================== 主流水线 =====================
class PreprocessPipeline:
    def __init__(self, target_len=360, target_k=30, target_fs=100, n_components=3):
        self.target_len = target_len
        self.target_k = target_k
        self.target_fs = target_fs
        self.n_components = n_components

        self.ops: Dict[str, List[BasePreprocessOp]] = {
            "PhaseUnwrap": [PhaseUnwrapOp()],
            "mimo": [MIMOCombineOp()],
            "conj": [ConjugateCorrelationOp()],
            "Hampel": [HampelFilterOp()],
            "SavitzkyGolay": [SavitzkyGolayOp()],
            "agc": [AGCOp()],
            "wls": [EnhancedWLSPhaseOp()],
            "static_removal": [ButterworthStaticRemovalOp(mode="timestamp", cutoff_freq=0.3)],
            "resample": [TimeResizeOp(target_len=target_len)],
            "resample_ablation": [PadOnlyTimeResizeOp(target_len=target_len)],
            "spline": [SplineFreqAlignOp(target_k=target_k)],
            "spline_ablation": [LinearFreqResizeOp(target_k=target_k)],
            "pca": [PostSTFTPCAOp(n_components=n_components)],
            "stft": [PerCarrierSTFTOp(fs=target_fs)],
            "norm": [STFTCarrierNormOp()],
            "zscore": [ZScoreNormOp()],
            "Channels": [ComplexToChannelsOp()]
        }

    def process_batch(self, csi_list: List[np.ndarray], time_list: List[np.ndarray],
                      device_type: str, config: AblationConfig, is_training=False) -> np.ndarray:
        op_sw = config.op_switches
        raw_freq = FREQ_TABLES.get(device_type, np.linspace(-28,28,self.target_k))
        t_sec_list = [(t-t[0])/1e6 for t in time_list]
        processed_csi = []
        strategy_map = []

        # 相位解缠绕
        for csi in csi_list:
            if op_sw.get("Unwrap", True):
                csi = self.ops["PhaseUnwrap"][0].transform(csi)
            nt, nr, K, T = csi.shape
            total_links = nt*nr

            # MIMO/共轭/SISO
            if op_sw.get("conjugate") and total_links>1:
                cf = self.ops["conj"][0].transform(csi)
                cf = np.squeeze(np.mean(cf, axis=(0,1)))
                strategy_map.append("CONJ_EGC")
            elif total_links>1:
                cf = self.ops["mimo"][0].transform(csi)
                strategy_map.append("MRC_WLS")
            else:
                cf = csi.squeeze((0,1))
                strategy_map.append("SISO_WLS")
            processed_csi.append(cf)

        amp_list = [np.abs(h) for h in processed_csi]
        phase_list = [np.angle(h) for h in processed_csi]

        # WLS相位校准
        for i in range(len(processed_csi)):
            s = strategy_map[i]
            if s in ["MRC_WLS","SISO_WLS"] and op_sw.get("wls",True):
                phase_list[i] = self.ops["wls"][0].transform((phase_list[i], amp_list[i]), device_type=device_type)

        # AGC
        if op_sw.get("agc",True):
            new_amp = []
            for a, t in zip(amp_list, t_sec_list):
                dur = t[-1]-t[0]
                fs = len(t)/dur if dur>0 else 20
                new_amp.append(self.ops["agc"][0].transform(a, fs=fs))
            amp_list = new_amp

        # 去噪
        if op_sw.get("hampel",True):
            amp_list = [self.ops["Hampel"][0].transform(a) for a in amp_list]
        if op_sw.get("SG",True):
            phase_list = [self.ops["SavitzkyGolay"][0].transform(p) for p in phase_list]

        # 静态消除
        if op_sw.get("static_removal",True):
            sr = self.ops["static_removal"][0]
            amp_list = [sr.transform(a, t_sec=t) for a,t in zip(amp_list, t_sec_list)]

        # 重采样 + 频域对齐
        res_op = self.ops["resample"][0] if op_sw.get("resample",True) else self.ops["resample_ablation"][0]
        align_op = self.ops["spline"][0] if op_sw.get("spline",True) else self.ops["spline_ablation"][0]

        proc_amp, proc_phase = [], []
        for a,p,t in zip(amp_list, phase_list, t_sec_list):
            a_t = res_op.transform(a)
            p_t = res_op.transform(p)
            a_f = align_op.transform(a_t, raw_freq=raw_freq)
            p_f = align_op.transform(p_t, raw_freq=raw_freq)
            proc_amp.append(a_f)
            proc_phase.append(p_f)

        # STFT + PCA + 通道变换
        stft = self.ops["stft"][0]
        norm = self.ops["norm"][0]
        pca = self.ops["pca"][0]
        to_ch = self.ops["Channels"][0]
        zscore = self.ops["zscore"][0]
        final = []

        for a, p in zip(proc_amp, proc_phase):

            h = a * np.exp(1j * p)

            stft_map = stft.transform(h)  # [K,F,T] complex

            mag = np.abs(stft_map)
            phase = np.angle(stft_map)

            feat = np.concatenate([
                mag,
                np.sin(phase),
                np.cos(phase)
            ], axis=0).astype(np.float32)

            final.append(feat)

        final = np.array(final)
        if is_training:
            zscore.fit(final)
        final = zscore.transform(final)
        return final