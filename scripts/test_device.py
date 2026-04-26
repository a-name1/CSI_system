import numpy as np
import scipy.io as sio

def analyze_csi_mat(file_path):
    """
    从 CSIBench MAT 文件中计算采样率、SNR 和抖动
    """
    # 1. 加载 MAT 文件
    data = sio.loadmat(file_path)
    # 根据结构提取 csi_trace
    csi_trace = data['csi_trace']
    
    # 提取字段 (注意：scipy.io 读入的结构可能有多层嵌套)
    csi = csi_trace['csi'][0, 0]           # [N_tx, N_rx, N_sub, N_samples]
    mactimer = csi_trace['mactimer'][0, 0].flatten()  # 单位: micro sec
    
    # ---------------------------
    # 计算抖动 (Jitter) 与 采样率
    # ---------------------------
    # 计算相邻样本的时间差 (diff)
    # 处理可能存在的计数器回绕 (Wraparound)
    intervals = np.diff(mactimer)
    intervals = intervals[intervals > 0] # 过滤回绕产生的负值或异常跳变
    
    # 抖动: 采样间隔的标准差 (单位: micro sec)
    jitter = np.std(intervals)
    
    # 实际平均采样率: 1 / (平均间隔秒数)
    avg_interval_sec = np.mean(intervals) / 1e6
    fs = 1.0 / avg_interval_sec
    
    # ---------------------------
    # 计算信噪比 (SNR)
    # ---------------------------
    # 1. 计算幅度
    amplitude = np.abs(csi) # [N_tx, N_rx, N_sub, N_samples]
    
    # 2. 估计信噪比: 
    # 这里采用基于时域方差的估计法 (针对相对平稳的数据段)
    # 对所有天线对和子载波取平均
    signal_power = np.mean(amplitude**2, axis=-1)
    noise_power = np.var(amplitude, axis=-1)
    
    # 避免除以 0
    noise_power[noise_power == 0] = 1e-10
    
    snr_linear = signal_power / noise_power
    avg_snr_db = 10 * np.log10(np.mean(snr_linear))
    
    # ---------------------------
    # 结果输出
    # ---------------------------
    print(f"--- 设备数据分析报告 ---")
    print(f"有效采样点数: {len(mactimer)}")
    print(f"估计采样率  : {fs:.2f} Hz")
    print(f"平均信噪比  : {avg_snr_db:.2f} dB")
    print(f"采样抖动    : {jitter:.2f} us")
    
    return {
        'fs': fs,
        'snr': avg_snr_db,
        'jitter': jitter
    }

# 使用示例
# result = analyze_csi_mat('Continuous_Data_Collection-2025..._AmazonPlug_5G.mat')