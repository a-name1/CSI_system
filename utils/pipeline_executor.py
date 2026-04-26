import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
from ablation_base import BasePreprocessOp
from preprocess_ops import *
from config_manager import AblationConfig
from preprocess_ops import FREQ_TABLES

# 绘图函数：确保每步都有幅度/相位对比
import matplotlib.pyplot as plt
from pathlib import Path    
def save_step_dual(amp, phase, step_name, filename):
                # --- 初始化 ---
        output_dir = Path("./pipeline_steps_detailed")
        output_dir.mkdir(exist_ok=True)
        plt.rcParams['font.family'] = 'serif'
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=150)
        # 左图：幅度
        ax1.plot(amp.T, lw=0.6)
        ax1.set_title(f"{step_name} - Amplitude")
        ax1.set_ylabel("Magnitude")
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        # 右图：相位
        ax2.plot(phase.T, lw=0.6)
        ax2.set_title(f"{step_name} - Phase")
        ax2.set_ylabel("Phase (rad)")
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        plt.suptitle(f"Pipeline Visualization: {step_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / filename, bbox_inches='tight')
        plt.close()
        print(f"--- [Exported] {filename} ---")

class PreprocessPipeline:
    def __init__(
        self,
        target_len=360,
        target_k=30,
        target_fs=100,
        n_components=3,
        model_input_T=128,
        model_input_F=32
    ):
        """
        更新后的流水线：集成了 MIMO 融合、共轭相乘特征和静态消除
        """
        self.ops: Dict[str, List[BasePreprocessOp]] = {
            #相位解缠绕
            "PhaseUnwrap": [PhaseUnwrapOp()],
            # 空间处理算子
            "mimo": [MIMOCombineOp()], 
            "conj": [ConjugateCorrelationOp()],            
            # 基础净化算子
            "Hampel": [HampelFilterOp()], # 调优参数
            "SavitzkyGolay": [SavitzkyGolayOp()], # 调优参数
            "agc": [AGCOp()], # 内部可改为 Log-AGC
            "wls": [EnhancedWLSPhaseOp()], # 增强版：含锚点展开和时间去趋势
            "static_removal": [ButterworthStaticRemovalOp(mode="timestamp", cutoff_freq=0.3)], # 1秒滑动窗口            
            "static_removal_abletion": [MovingAverageStaticRemovalOp()],
            # 对齐算子
            "resample": [TimeResizeOp()],
            "resample_ablation": [PadOnlyTimeResizeOp()],
            "spline": [SplineFreqAlignOp()],
            "spline_ablation": [LinearFreqResizeOp()],           
            # 后处理算子
            "pca": [PCAOp()],
            "stft": [STFTOp()], # 增加重叠度提高平滑性
            "zscore": [ZScoreNormOp()],
            "dim_align": [DimAlignOp()],
            #dft
            "DFSExtraction": [DFSExtractionOp()],
            "RobustScaleOp":[RobustScaleOp()],
        }

    def process_batch(
        self,
        csi_list: List[np.ndarray], # [N, Nt, Nr, K, T]
        time_list: List[np.ndarray],
        device_type: str,
        config: AblationConfig,
        is_training: bool = False
    ) -> np.ndarray:
        """批量执行预处理流水线"""
        op_switches = config.op_switches
        raw_freq = FREQ_TABLES.get(device_type, np.linspace(-28, 28, 30))

# --------------------------
        # Step 0: 空间维度决策引擎 (MIMO vs. Conjugate vs. SISO)
        # --------------------------
        is_conj_sw = op_switches.get("conjugate")
        processed_csi_list = []
        t_sec_list = [(t - t[0]) / 1e6 for t in time_list]
        # 策略标记：用于下游决定是否触发 WLS
        # 0: CONJ_EGC (相对相位路径, 跳过WLS)
        # 1: MRC_WLS  (绝对相位路径, 需要WLS)
        # 2: SISO_WLS (单天线退化路径, 强制WLS)
        strategy_map = [] 

        # # 绘图：处理前的原始状态（取第一个样本的第一根天线观测）
        # save_step_dual(
        #     np.abs(csi_list[0][0, 0, :, :]), 
        #     np.angle(csi_list[0][0, 0, :, :]),
        #     "Raw CSI (Hardware Non-ideal Phase)",
        #     "step00_raw_input.png"
        # )
        
        for csi in csi_list:
            nt, nr, n_sub, n_time = csi.shape
            if op_switches.get("Unwrap", True):
                csi = self.ops["PhaseUnwrap"][0].transform(csi)
            else:
                pass
            total_links = nr * nt
            if nr * nt == 1:
                print(f"警告：样本 {i} 是单天线数据，无法执行路径 B (Conjugate)！")            
            # --- 场景 A: 满足共轭相乘条件 (多天线 + 开关开启) ---
            if is_conj_sw and total_links > 1:
                # 1. 物理差分：提取相对信道特征，消除共模噪声
                csi_feat = self.ops["conj"][0].transform(csi)
                
                # 2. 空间压平：由于共轭已完成相位对齐，此处使用 EGC (等权平均) 抑制白噪声
                if csi_feat.ndim > 2:
                    # np.mean 即为 EGC 在此处的物理实现
                    csi_feat = np.mean(csi_feat, axis=(0, 1))
                # 3. 彻底降维：将残留的 size-1 维度全部删掉，确保变为 (K, T)
                csi_feat = np.squeeze(csi_feat)
                
                # 这里的保险检查：确保最终只有 2 维
                if csi_feat.ndim != 2:
                    # 万一 Nt 或 Nr 的均值没压干净，强制取最后一层
                    csi_feat = csi_feat.reshape(-1, n_time)[-n_sub:, :]
                strategy_map.append("CONJ_EGC")
                
            # --- 场景 B: 多天线绝对相位恢复 (MIMO 路径) ---
            elif total_links > 1:
                # 1. 空间融合：使用 MRC 最大比合并提升 SNR
                csi_feat = self.ops["mimo"][0].transform(csi)
                
                strategy_map.append("MRC_WLS")
                
            # --- 场景 C: 单天线退化 (SISO) ---
            else:
                # 直接降维，由于无空间自由度，MRC/EGC 均退化为原始信号
                csi_feat = csi.squeeze((0, 1))
                strategy_map.append("SISO_WLS")
                
            processed_csi_list.append(csi_feat)

        # --------------------------
        # Step 1: 相位校准策略自适应 (数学拟合 vs. 物理对冲)
        # --------------------------
        amp_list = [np.abs(h) for h in processed_csi_list]
        phase_list = [np.angle(h) for h in processed_csi_list]

        # 核心逻辑：只有非共轭路径（MRC 或 SISO）才需要 WLS 进行去斜率
        # 共轭路径（CONJ_EGC）由于已物理对冲，相位已水平，跳过 WLS 以防过拟合
        for i in range(len(processed_csi_list)):
            current_s = strategy_map[i]
            
            if current_s in ["MRC_WLS", "SISO_WLS"] and op_switches.get("wls", True):
                wls_op = self.ops["wls"][0]
                phase_list[i] = wls_op.transform(
                    (phase_list[i], amp_list[i])
                )
                # print(f"Sample {i}: 路径 {current_s} -> 执行 WLS 数学校准")
                # logger.debug(f"Sample {i}: 路径 {current_s} -> 执行 WLS 数学校准")
            else:
                # print(f"Sample {i}: 路径 {current_s} -> 物理去噪已完成，跳过 WLS")
                # logger.debug(f"Sample {i}: 路径 {current_s} -> 物理去噪已完成，跳过 WLS")
                pass

        # # 绘图：展示最终校准效果（此时相位应呈现水平且清晰的演变趋势）
        # # 标注当前采用的全局决策策略
        # final_strategy = strategy_map[0] if len(set(strategy_map)) == 1 else "HYBRID"
        # save_step_dual(
        #     amp_list[0], phase_list[0], 
        #     f"Phase After Calibration ({final_strategy})", 
        #     "step01_final.png"
        # )
        # --------------------------
        # Step 2: AGC
        # --------------------------
        if op_switches.get("agc", True):
            agc_op = self.ops["agc"][0]
            amp_list = [agc_op.transform(amp) for amp in amp_list]
        
        # print(f"AGC处理后样本数: {len(amp_list)}, 单样本形状: {amp_list[0].shape}")
        # save_step_dual(
        #     amp_list[0],
        #     phase_list[0],
        #     "After AGC (First Sample, First Rx-Tx Pair)",
        #     "step05_agc.png"
        # )
        # --------------------------
        # Step 3: 基础去噪 Hampel
        # --------------------------
        if op_switches.get("hampel", True):
            amp_list = [self.ops["Hampel"][0].transform(amp) for amp in amp_list]
        # --------------------------
        # Step 4: 基础去噪 SG
        # --------------------------
        if op_switches.get("SG", True):
            phase_list = [self.ops["SavitzkyGolay"][0].transform(phase) for phase in phase_list]
        
        # print(f"去噪处理后样本数: {len(amp_list)}, 单样本形状: {amp_list[0].shape}")

        # save_step_dual(
        #     amp_list[0],
        #     phase_list[0],
        #     "After Denoising (First Sample, First Rx-Tx Pair)",
        #     "step03_denoise.png"
        # )
        # --------------------------
        # Step 5: 静态消除 (针对 LODO 的核心)
        # --------------------------
        if op_switches.get("static_removal", True):
            sr_op = self.ops["static_removal"][0]
            amp_list = [
                sr_op.transform(amp, t_sec=t_sec) 
                for amp, t_sec in zip(amp_list, t_sec_list)
            ]

        # 注意：相位通常不进行静态消除，因为相位差已经具备类似的物理意义

        # print(f"静态消除处理后样本数: {len(amp_list)}, 单样本形状: {amp_list[0].shape}")

        # save_step_dual(
        #     amp_list[0],
        #     phase_list[0],
        #     "After Static Removal (First Sample, First Rx-Tx Pair)",
        #     "step04_static_removal.png"
        # )
        
        # --------------------------
        # Step 5: 时间重采样 & 频率对齐 (归一化到统一 shape)
        # --------------------------
        res_op = self.ops["resample"][0] if op_switches.get("resample", True) else self.ops["resample_ablation"][0]
        align_op = self.ops["spline"][0] if op_switches.get("spline", True) else self.ops["spline_ablation"][0]

        joint_feat_list = []
        for amp, phase, t_sec in zip(amp_list, phase_list, t_sec_list):
            # 注意：这里的 amp 需要是 [K, T] 还是 [T, K] 取决于 res_op 的实现
            # 假设 res_op 内部处理的是 [T, K]
            a_t = res_op.transform(amp.T, t_sec=t_sec)
            p_t = res_op.transform(phase.T, t_sec=t_sec)
            
            a_final = align_op.transform(a_t, raw_freq=raw_freq)
            p_final = align_op.transform(p_t, raw_freq=raw_freq)
            
            joint_feat_list.append(np.concatenate([a_final, p_final], axis=1))
        
        # # 修正 1：打印列表长度和单个样本形状
        # print(f"时间重采样 & 频率对齐处理后样本数: {len(joint_feat_list)}, 单样本形状: {joint_feat_list[0].shape}")

        # # 修正 2：绘图逻辑
        # # 取最后一个处理完的样本 (a_final, p_final) 进行可视化
        # # 传入 [T, K] 矩阵，save_step_dual 内部会画出所有子载波随时间变化的曲线
        # save_step_dual(
        #     a_final.T, # 转置回 [K, T] 以符合 save_step_dual 的绘图习惯
        #     p_final.T,
        #     "After Time Resampling & Frequency Alignment",
        #     "step05_resample_align.png"
        # )
# --------------------------
        # Step 6: PCA -> STFT -> DimAlign
        # --------------------------
        if op_switches.get("PCA", True):
            pca_op = self.ops["pca"][0]
            stft_op = self.ops["stft"][0]
            if op_switches.get("zscore", True):
                norm_op = self.ops["zscore"][0]
            else:
                norm_op = self.ops["RobustScaleOp"][0]
            dim_align_op = self.ops["dim_align"][0]
            if is_training:
                pca_op.fit(joint_feat_list)        
            # 1. 先生成所有样本的 PCA 结果
            pca_out_list = [pca_op.transform(jf) for jf in joint_feat_list]
        
            # # 2. 修正绘图：取第一个样本可视化
            # sample_pca = pca_out_list[0] # 形状通常为 (target_len, n_components) -> (360, 3)
            # print(f"PCA处理后样本数: {len(pca_out_list)}, 单样本形状: {pca_out_list[0].shape}")
            # plt.figure(figsize=(10, 4), dpi=100)
            # plt.plot(sample_pca, lw=1.2)
            # plt.title("Step 6: PCA Reduced Features (Top 3 Components)")
            # plt.legend(['PC1', 'PC2', 'PC3'])
            # plt.xlabel("Time Samples")
            # plt.ylabel("Amplitude")
            # plt.grid(True, linestyle=':', alpha=0.7)
            # plt.savefig(Path("./pipeline_steps_detailed") / "06_pca.png")
            # plt.close()       
            final_features = []
            for pca_out in pca_out_list:
                # 执行 STFT 并内部剔除 0Hz
                spec = stft_op.transform(pca_out)
                spec_norm = norm_op.fit_transform(spec)
                # 对齐到 CNN 输入尺寸 [3, 64, 32]
                final_feat = dim_align_op.transform(spec_norm)
                final_features.append(final_feat)

            # # 3. 修正绘图：展示最后一个处理完的 final_feat
            # print(f"final_stft处理后样本数: {len(final_feat)}, 单样本形状: {final_feat[0].shape}")
            # fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)
            # for i in range(min(3, final_feat.shape[0])):
            #     # final_feat[i] 形状为 (64, 32) -> (Time, Freq)
            #     # 转置是为了让频率轴作为纵轴，时间轴作为横轴，符合常规频谱图习惯
            #     im = axes[i].pcolormesh(final_feat[i].T, shading='gouraud', cmap='jet')
            #     axes[i].set_title(f"PC{i+1} Spectrogram (64x32)")
            #     axes[i].set_xlabel("Time Bin")
            #     axes[i].set_ylabel("Frequency Bin")
            #     plt.colorbar(im, ax=axes[i])
                
            # plt.suptitle("Step 7: Final Model Input Features (Normalized STFT)", fontsize=14, fontweight='bold')
            # plt.tight_layout()
            # plt.savefig(Path("./pipeline_steps_detailed") / "07_final_stft.png")
            # plt.close()
            # print("数据处理完成。")
            # input("按下回车键以继续执行后续逻辑...")
            # print("继续运行中...")
            # 返回形状为 (Batch, 3, 64, 32) 的四维张量
            return np.stack(final_features, axis=0)
        else:
            dfs = self.ops["DFSExtraction"][0]
            RobustScalenorm_spec = self.ops["RobustScaleOp"][0]
            RobustScalenorm_d1 = self.ops["RobustScaleOp"][0]
            RobustScalenorm_d2 = self.ops["RobustScaleOp"][0]
            # joint_feat_arr: [T, 2K]
            joint_feat_arr = np.array(joint_feat_list)
            spec = dfs.transform(joint_feat_arr)
            spec = RobustScalenorm_spec.fit_transform(spec)

                # Δ + ΔΔ（增强动态信息）
            d1 = np.diff(spec_norm, axis=-1, prepend=spec_norm[..., [0]])
            d2 = np.diff(d1, axis=-1, prepend=d1[..., [0]])

            d1_norm = RobustScalenorm_d1.fit_transform(d1)
            d2_norm = RobustScalenorm_d2.fit_transform(d1)
            stacked = np.stack([spec_norm, d1_norm, d2_norm], axis=0)

            final_feat = dim_align_op.transform(stacked)
            final_features.append(final_feat)
            return np.stack(final_features, axis=0)
