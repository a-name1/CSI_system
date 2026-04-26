import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from ablation_base import BasePreprocessOp
from typing import Union, Tuple
from scipy.signal import butter, filtfilt
import numpy as np
from typing import Optional
# ====================== 全局常量配置（与业务逻辑解耦） ======================
FREQ_TABLES = {
    "INTEL_14": np.array([-28, -26, -24, -22, -20, -18, -16, 16, 18, 20, 22, 24, 26, 28]),
    "BROADCOM_58": np.arange(-28, 30)
}

# ====================== 每个预处理步骤拆分为独立算子 ======================
# 空间处理算子与相位处理算子
class PhaseUnwrapOp(BasePreprocessOp):
    """
    多天线 CSI 【高级锚点相位解缠绕】算子（纯解缠绕，无其他处理）
    输入：[Nt, Nr, K, T] 复数 CSI
    输出：[Nt, Nr, K, T] 解缠绕后的复数 CSI（幅度不变，相位连续）
    
    核心解缠策略：
    1. 先沿时间轴整体 unwrap（保证时序连续）
    2. 逐时刻沿频率轴，以【最大幅度子载波】为锚点双向展开（最稳定）
    """

    def __init__(self, time_axis: int = -1):
        self.time_axis = time_axis  # 时间维度，默认 T 在最后

    @property
    def op_name(self) -> str:
        return "phase_unwrap_anchor"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        # 输入形状：[Nt, Nr, K, T]
        amp = np.abs(data)
        phase = np.angle(data)

        # ==============================================
        # 【步骤1】先沿时间轴解缠绕（时序连续化）
        # ==============================================
        phase_unwrap = np.unwrap(phase, axis=self.time_axis)

        # ==============================================
        # 【步骤2】逐帧 + 频率轴锚点解缠绕（核心增强）
        # ==============================================
        K = phase.shape[-2]  # 子载波维度
        T = phase.shape[-1]  # 时间维度

        # 将高维数据展平，统一处理所有天线链路
        original_shape = phase_unwrap.shape
        p_flat = phase_unwrap.reshape(-1, K, T)  # [N_links, K, T]
        a_flat = amp.reshape(-1, K, T)
        N_links = p_flat.shape[0]

        result = []
        for link_idx in range(N_links):
            p = p_flat[link_idx].copy()  # [K, T]
            a = a_flat[link_idx]         # [K, T]

            # 逐时刻 t，做频率轴锚点解缠
            for t in range(T):
                p_t = p[:, t]
                a_t = a[:, t]

                # 以【幅度最大子载波】为锚点
                anchor_idx = np.argmax(a_t)

                # 锚点双向展开（你原版的核心逻辑）
                p[:, t] = self._anchor_unwrap(p_t, anchor_idx)

            result.append(p)

        # 恢复原始形状
        phase_final = np.stack(result, axis=0).reshape(original_shape)

        # 重构复数信号
        data_unwrap = amp * np.exp(1j * phase_final)
        return data_unwrap

    def _anchor_unwrap(self, p: np.ndarray, anchor_idx: int):
        """
        锚点相位展开：从最强子载波向左右两边展开
        完全保留你原版的逻辑
        """
        # 向右展开
        right = np.unwrap(p[anchor_idx:])
        # 向左展开（翻转 -> 解缠 -> 翻转回来）
        left = np.flip(np.unwrap(np.flip(p[:anchor_idx + 1])))
        return np.concatenate([left[:-1], right])
    
class MIMOCombineOp(BasePreprocessOp):
    """
    多天线空域融合算子
    支持：等权平均 (EGC) 或 最大比合并 (MRC)
    """
    def __init__(self, mode='mrc'):
        self.mode = mode

    @property
    def op_name(self) -> str:
        return "mimo_combine"
    
    def fit(self, data: np.ndarray, **kwargs) -> None:
        # 无参数拟合，空实现
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        data 形状假设为: [Nt, Nr, K, T] 
        即：接收天线 x 发送天线 x 子载波 x 时间轴
        """
        # 1. 展平天线维度 [Nr*Nt, K, T] -> [N_links, K, T]
        shape = data.shape
        links = data.reshape(-1, shape[-2], shape[-1])
        
        if self.mode == 'mrc':
            # 计算每个链路的幅度作为权重
            amps = np.abs(links)
            weights = amps**2
            # 归一化权重
            weights_norm = weights / (np.sum(weights, axis=0, keepdims=True) + 1e-8)
            # 加权合并（注意：这里是对复数进行加权，以保留相位演变）
            combined = np.sum(links * weights_norm, axis=0)
        else:
            # 简单等权平均
            combined = np.mean(links, axis=0)
            
        return combined

class ConjugateCorrelationOp(BasePreprocessOp):
    """
    共轭相乘特征提取算子
    利用天线间的相位差抵消共模时钟噪声
    """
    def __init__(self, ref_ant_idx=0):
        self.ref_ant_idx = ref_ant_idx # 指定参考天线

    @property
    def op_name(self) -> str:
        return "conj_correlation"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        # 无参数拟合，空实现
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        data: [Nt, Nr, K, T] -> 根据你的打印 [1, 4, 114, 500]
        目标输出: [Nt, Nr-1, K, T] -> [1, 3, 114, 500]
        """
        Nt, Nr, K, T = data.shape
        
        # 提取参考接收天线 (通常是第 1 个 Tx 对应的第 ref_ant_idx 个 Rx)
        # 保持维度为 [1, 1, K, T] 方便后续广播计算
        h_ref = data[:, [self.ref_ant_idx], :, :] 
        
        features = []
        for i in range(Nr):
            if i == self.ref_ant_idx:
                continue
            
            # data[:, [i], :, :] 保持维度为 [Nt, 1, K, T]
            # 执行共轭相乘，物理上是对冲了 Nt 端的时钟噪声
            corr = data[:, [i], :, :] * np.conj(h_ref)
            features.append(corr)
            
        # 堆叠回 Nr 维度，结果为 [Nt, Nr-1, K, T]
        # 这样后续的 np.mean(axis=(0, 1)) 就会先压平天线对，再压平发送端
        return np.stack(features, axis=1)
    
class EnhancedWLSPhaseOp(BasePreprocessOp):
    """
    1. 移除所有相位解缠绕逻辑
    2. 仅执行 WLS 线性去偏
    3. 仅执行 时间轴去趋势 (Detrending) 消除大规模漂移
    输入：(phase: [K, T], amp: [K, T])
    输出：clean_phase [K, T]
    """
    @property
    def op_name(self) -> str:
        return "wls_enhanced_no_unwrap"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def transform(self, data: Tuple[np.ndarray, np.ndarray], **kwargs) -> np.ndarray:
        phase, amp = data
        device_type = kwargs.get("device_type", "INTEL_14")
        k = FREQ_TABLES.get(device_type, np.arange(phase.shape[0]))
        sanitized = np.zeros_like(phase)

        # --- 步骤 1: 逐帧 WLS 线性去偏
        A = np.vstack([k, np.ones(len(k))]).T
        for t in range(phase.shape[1]):
            y = phase[:, t]  # 直接使用原始相位，不做锚点展开
            w = amp[:, t]
            
            # 加权最小二乘拟合斜率
            sol = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
            sanitized[:, t] = y - (sol[0] * k + sol[1])

        # --- 步骤 2: 时间轴去趋势
        sanitized = signal.detrend(sanitized, axis=1, type='linear')

        return sanitized

#去噪算子：
class HampelFilterOp(BasePreprocessOp):
    def __init__(self, k=13, t0=3):
        self.k = k
        self.t0 = t0

    @property
    def op_name(self) -> str:
        return "denoise"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        # 无参数拟合，空实现
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
            """
            data: [K, T] 可能是复数也可能是实数(幅度)
            """
            # 1. 显式检查输入类型并处理
            is_complex = np.iscomplexobj(data)
            
            # 确保 x_out 的类型与后续 SavGol 期望的实数一致（如果是实部消融）
            if not is_complex:
                x_out = data.copy().astype(np.float64)
            else:
                x_out = data.copy()

            for i in range(data.shape[0]):
                # 统一取幅度进行异常值判断
                series = np.abs(data[i, :])
                
                # 处理样本点数过少的情况（medfilt 要求 kernel_size < len）
                current_k = self.k
                if series.shape[0] <= current_k:
                    current_k = series.shape[0] if series.shape[0] % 2 != 0 else series.shape[0] - 1
                
                if current_k < 3: # 如果数据太短，无法做中值滤波，跳过当前 subcarrier
                    continue

                med = signal.medfilt(series, kernel_size=current_k)
                mad = np.median(np.abs(series - med))
                sigma = mad / 1.4826 if mad != 0 else 1e-6
                outliers = np.abs(series - med) > self.t0 * sigma
                
                # 2. 根据输入类型决定赋值逻辑
                if is_complex:
                    # 复数：保留相位，替换幅度
                    x_out[i, outliers] = med[outliers] * np.exp(1j * np.angle(data[i, outliers]))
                else:
                    # 实数：直接替换幅度（也就是实部本身）
                    x_out[i, outliers] = med[outliers]
                    
            return x_out

class SavitzkyGolayOp(BasePreprocessOp):
    def __init__(self, window_length=31, polyorder=3):
        self.window_length = window_length
        self.polyorder = polyorder

    @property
    def op_name(self) -> str:
        return "denoise"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    # 修改后（增加自适应逻辑）：
    def transform(self, data):
        # 获取数据在滤波轴上的长度
        n_samples = data.shape[1] 
        
        # 动态确定窗口：不能超过数据长度，且必须是奇数
        # 如果数据太短（比如还没 polyorder 高），则跳过滤波或设为最小值
        current_window = self.window_length
        if n_samples <= current_window:
            current_window = n_samples if n_samples % 2 != 0 else n_samples - 1
        
        # SavGol 还要求 window_length > polyorder
        if current_window <= self.polyorder:
            # 如果数据点数实在太少，连多项式拟合都做不了，直接返回原数据
            return data
            
        return signal.savgol_filter(data, window_length=current_window, polyorder=self.polyorder, axis=1)

class AGCOp(BasePreprocessOp):
    @property
    def op_name(self) -> str:
        return "agc"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """data: [K, T] 幅度"""
        ref = np.median(data, axis=1, keepdims=True)
        return data / (ref + 1e-8)

#时频域重采样
class TimeResizeOp(BasePreprocessOp):
    def __init__(self, target_len=360):
        self.target_len = target_len

    @property
    def op_name(self) -> str:
        return "resample"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """data: [T, K]"""
        T_in, K_in = data.shape
        x_old = np.linspace(0, 1, T_in)
        x_new = np.linspace(0, 1, self.target_len)
        f = interp1d(x_old, data, axis=0, kind='linear', fill_value="extrapolate")
        return f(x_new)

class PadOnlyTimeResizeOp(BasePreprocessOp):
    """
    【重采样消融算子】仅在长度不足时补0，长度足够则直接返回
    结合了恒等映射和补0，更激进地验证重采样的作用
    """
    def __init__(self, target_len=360, pad_side="post"):
        self.target_len = target_len
        self.pad_side = pad_side

    @property
    def op_name(self) -> str:
        return "pad_only_resample"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        T_in, K_in = data.shape
        if T_in >= self.target_len:
            return data.copy()
        
        # 仅在长度不足时补0
        pad_len = self.target_len - T_in
        pad_width = ((0, pad_len), (0, 0)) if self.pad_side == "post" else ((pad_len, 0), (0, 0))
        return np.pad(data, pad_width, mode="constant", constant_values=0)

class SplineFreqAlignOp(BasePreprocessOp):
    def __init__(self, target_k=30):
        self.target_k = target_k
        self.target_freq_grid = np.linspace(-28, 28, target_k)

    @property
    def op_name(self) -> str:
        return "spline"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """data: [T, K]"""
        raw_freq = kwargs.get("raw_freq", np.linspace(-28, 28, data.shape[1]))
        f = interp1d(raw_freq, data, axis=1, kind='cubic', fill_value="extrapolate")
        return f(self.target_freq_grid)

class LinearFreqResizeOp(BasePreprocessOp):
    def __init__(self, target_k=30):
        self.target_k = target_k

    @property
    def op_name(self) -> str:
        return "spline"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """消融替代算子：简单线性缩放子载波轴,三次样条对照消融"""
        T_in, K_in = data.shape
        x_old = np.linspace(0, 1, K_in)
        x_new = np.linspace(0, 1, self.target_k)
        f = interp1d(x_old, data, axis=1, kind='linear', fill_value="extrapolate")
        return f(x_new)


#静态去除
class ButterworthStaticRemovalOp(BasePreprocessOp):
    def __init__(
        self,
        mode: str = "normalized",
        normalized_cutoff: float = 0.006,
        cutoff_freq: float = 0.3,
        fs: Optional[float] = 100,
        order: int = 1
    ):
        self.mode = mode
        self.norm_cut = normalized_cutoff
        self.cutoff_hz = cutoff_freq
        self.fs = fs
        self.order = order

        # 初始化时校验归一化频率
        if self.mode == "normalized" and not (0 < self.norm_cut < 1):
            raise ValueError(f"归一化截止频率必须在 (0, 1) 之间，当前值: {self.norm_cut}")

    @property
    def op_name(self) -> str:
        return "butterworth_static_removal"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        res = data.copy()
        T = res.shape[-1]

        # 校验数据长度
        min_length = 3 * self.order
        if T < min_length:
            raise ValueError(f"数据长度 ({T}) 过短，建议至少为 {min_length}")

        # 计算归一化截止频率
        if self.mode == "normalized":
            normal_cutoff = self.norm_cut

        elif self.mode == "timestamp":
            t_sec = kwargs.get("t_sec", None)
            if t_sec is None:
                raise ValueError("mode='timestamp' 时，必须传入 t_sec")
            if len(t_sec) != T:
                raise ValueError(f"t_sec 长度 ({len(t_sec)}) 需与数据长度 ({T}) 一致")
            
            fs_local = 1.0 / np.mean(np.diff(t_sec))
            nyq = 0.5 * fs_local
            normal_cutoff = self.cutoff_hz / nyq

        elif self.mode == "traditional":
            if self.fs is None:
                raise ValueError("mode='traditional' 时，必须输入 fs")
            nyq = 0.5 * self.fs
            normal_cutoff = self.cutoff_hz / nyq

        else:
            raise ValueError(f"未知模式: {self.mode}")

        # 零相位滤波
        b, a = butter(self.order, normal_cutoff, btype="high", analog=False)
        res = filtfilt(b, a, res, axis=-1)
        
        return res

class MovingAverageStaticRemovalOp(BasePreprocessOp):
    """
    【消融实验用】滑动平均估计背景消除静态分量
    用途：仅用于消融对比实验，验证巴特沃斯的优势
    特点：简单快速，但有相位延迟/过渡带模糊，不推荐正式训练
    输入支持：[Nt, Nr, K, T] / [Nr, K, T] / [K, T]
    """
    def __init__(
        self,
        window_size: int = 100  # 滑动平均窗口大小
    ):
        """
        :param window_size: 滑动平均窗口大小
            - 对应近似截止频率 ≈ fs/(2*window_size)
            - 例如：fs=100Hz时，window_size=100对应≈0.5Hz截止
        """
        self.window_size = window_size

    @property
    def op_name(self) -> str:
        return "moving_average_static_removal"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        res = data.copy()
        T = res.shape[-1]  # 时间轴始终在最后一维
        
        # 构造滑动平均核
        kernel = np.ones(self.window_size) / self.window_size
        
        # 展平高维数据，统一处理所有非时间维度
        original_shape = res.shape
        res_flat = res.reshape(-1, T)
        background_flat = np.zeros_like(res_flat)
        
        # 对每个独立通道做零相位滑动平均（尽量减少相位延迟，仅用于对比）
        for i in range(res_flat.shape[0]):
            background_flat[i] = filtfilt(kernel, 1.0, res_flat[i])
        
        # 恢复形状，减去背景
        background = background_flat.reshape(original_shape)
        dynamic_residual = res - background
        
        return dynamic_residual

#维度压缩
class PCAOp(BasePreprocessOp):
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.pca_model = None

    @property
    def op_name(self) -> str:
        return "pca"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        """data: [N, T, 2K] 批量特征"""
        if self.n_components <= 0:
            return
        all_data = np.vstack(data)
        self.pca_model = PCA(n_components=self.n_components)
        self.pca_model.fit(all_data)

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """data: [T, 2K]"""
        if self.n_components <= 0 or self.pca_model is None:
            # 消融替代：截断/补零对齐维度
            T_in, K_in = data.shape
            if K_in >= self.n_components:
                return data[:, :self.n_components]
            else:
                return np.pad(data, ((0, 0), (0, self.n_components - K_in)), mode='constant')
        return self.pca_model.transform(data)

class STFTOp(BasePreprocessOp):
    def __init__(self, target_fs=100, nperseg=64, noverlap=48):
        self.target_fs = target_fs
        self.nperseg = nperseg
        self.noverlap = noverlap

    @property
    def op_name(self) -> str:
        return "stft"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        data: [T, D] PCA后的多维数据
        目标输出: [T_prime, F, D] 或者是展开后的 [T_prime, F*D]
        """
        all_specs = []
        # 遍历所有 PCA 维度 (D)
        for d in range(data.shape[1]):
            f, t_s, spec = signal.stft(
                data[:, d], 
                nfft = 256,
                fs=self.target_fs, 
                nperseg=self.nperseg, 
                noverlap=self.noverlap,
                detrend=False # 重点：这里设为 False，由我们手动控制剔除
            )
            all_specs.append(np.abs(spec)) # [F, T_prime]
            # --- 核心修改：剔除 0Hz (DC) 分量 ---
            # 在返回的频率轴 f 中，寻找接近 0 的索引
            # 对于单边谱，通常是第一行；对于双边谱，是中间行
            center_idx = np.argmin(np.abs(f))
            
            # 将 0Hz 及其邻域设为极小值或直接置零
            # 建议置为均值或一个很小的数，防止 Z-Score 时出现 NaN
            spec[center_idx, :] = 1e-10 # 极小值而非绝对零，防止后续 log 或 std 计算溢出 
            # 进阶建议：可以把 0Hz 左右 1-2 个 Bin 也抹掉（解决频率泄露）
            # spec[max(0, center_idx-1) : center_idx+2, :] = 0            
        # 将所有维度的时频图堆叠在一起
        # 方式 A: 作为通道处理 [D, F, T_prime] -> 推荐，这样 CNN 卷积时可以融合不同主成分
        full_spec = np.stack(all_specs, axis=0) 
        
        # 根据你模型的习惯，可能需要转置
        # 返回形状 [D, F,T_prime] 对应你之前的逻辑
        return full_spec.transpose(0, 1, 2)

class ZScoreNormOp(BasePreprocessOp):
    @property
    def op_name(self) -> str:
        return "zscore"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

class DimAlignOp(BasePreprocessOp):
    def __init__(self, model_input_T=128, model_input_F=32):
        self.model_input_T = model_input_T
        self.model_input_F = model_input_F

    @property
    def op_name(self) -> str:
        return "dim_align"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        强制对齐输出形状为 [D, model_input_T, model_input_F]
        """
        # --- 1. 严格的维度标准化 ---
        # 确保 data 最终形状为 [D, F_in, T_in]
        if data.ndim == 2:
            # 处理 DFS 直接输出的 [F, T]
            data = data[np.newaxis, ...]
        elif data.ndim == 4:
            # 移除可能存在的 batch 维度 [1, D, F, T]
            data = np.squeeze(data, axis=0)
        
        if data.ndim != 3:
            raise ValueError(f"DimAlignOp 期待 3 维输入 (D, F, T)，实际得到: {data.shape}")

        D, F_in, T_in = data.shape
        
        # --- 2. 关键修改：预分配固定形状的容器 ---
        # 这样无论内部如何插值，最终填入这个容器的部分形状永远是统一的
        # 形状定义为 (通道, 时间, 频率)
        output_aligned = np.zeros((D, self.model_input_T, self.model_input_F), dtype=np.float32)
        
        # --- 3. 遍历通道进行双线性插值 ---
        for d in range(D):
            spec = data[d] # [F_in, T_in]
            
            # 安全检查：如果输入的时间或频率维度太小，无法插值
            if T_in < 2 or F_in < 2:
                # 这种情况通常是数据损坏，直接跳过插值，保持 output_aligned[d] 为 0 或返回异常
                continue

            # 步骤 A: 对齐时间维度 (axis=1: T_in -> model_input_T)
            x_t = np.linspace(0, 1, T_in)
            x_t_new = np.linspace(0, 1, self.model_input_T)
            f_t = interp1d(x_t, spec, axis=1, kind='linear', fill_value="extrapolate")
            spec_t_fixed = f_t(x_t_new) # [F_in, T_model]

            # 步骤 B: 对齐频率维度 (axis=0: F_in -> model_input_F)
            x_f = np.linspace(0, 1, F_in)
            x_f_new = np.linspace(0, 1, self.model_input_F)
            f_f = interp1d(x_f, spec_t_fixed, axis=0, kind='linear', fill_value="extrapolate")
            feat_final = f_f(x_f_new) # [F_model, T_model]
            
            # 步骤 C: 转置并存入预分配的容器
            # feat_final.T 形状为 [T_model, F_model]
            output_aligned[d] = feat_final.T

        # --- 4. 最终验证 ---
        # 这一步保证了 np.stack(processed_features) 时绝对不会因为形状不同而报错
        return output_aligned
    
from scipy import signal
import numpy as np
from typing import Optional

class DFSExtractionOp(BasePreprocessOp):
    def __init__(self, fs=100, nperseg=64, noverlap=48, nfft=256):
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft

    @property
    def op_name(self) -> str:
        return "dfs_transform"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def _get_adaptive_params(self, T_len: int) -> tuple[int, int]:
        """
        内部辅助函数：根据数据长度自适应调整 nperseg 和 noverlap
        :param T_len: 时间维度长度
        :return: (adaptive_nperseg, adaptive_noverlap)
        """
        current_nperseg = self.nperseg
        current_noverlap = self.noverlap

        if T_len < current_nperseg:
            # 数据短于默认窗口：缩小窗口到数据长度
            current_nperseg = T_len
            # 重叠长度必须小于窗口长度（至少留1个点不重叠）
            current_noverlap = min(self.noverlap, current_nperseg - 1)
        
        # 处理极短数据（至少保留2个点做STFT）
        if current_nperseg < 2:
            current_nperseg = 2
            current_noverlap = 0
        
        return current_nperseg, current_noverlap

    def _transform_single_channel(self, signal_1d: np.ndarray) -> np.ndarray:
        """
        内部辅助函数：对单个一维时间序列做 STFT 并处理
        :param signal_1d: 一维时间序列 [T,]
        :return: 处理后的 DFS 能量谱 [F, T_stft]（F=频率Bin数，T_stft=STFT时间帧数）
        """
        T_len = len(signal_1d)
        current_nperseg, current_noverlap = self._get_adaptive_params(T_len)

        # 执行 STFT
        f, t_s, spec = signal.stft(
            signal_1d, 
            fs=self.fs, 
            nperseg=current_nperseg, 
            noverlap=current_noverlap, 
            nfft=self.nfft, 
            return_onesided=False
        )

        # 后续处理：fftshift + 直流分量抑制 + 能量归一化
        spec = np.fft.fftshift(spec, axes=0)
        center_idx = spec.shape[0] // 2
        # 抑制直流分量（中心附近3个Bin）
        spec[center_idx-1 : center_idx+2, :] = 1e-10
        dfs_energy = np.abs(spec)
        # 归一化到 [0, 1]
        dfs_norm = (dfs_energy - np.min(dfs_energy)) / (np.max(dfs_energy) - np.min(dfs_energy) + 1e-8)
        
        return dfs_norm

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        适配 joint_feat_list 的核心 transform
        :param data: 输入数据，支持两种形状：
            - 单样本：[T, 2K]（T=时间点数，2K=特征数：前K幅度、后K相位）
            - 多样本：[N, T, 2K]（N=样本数）
        :return: 处理后的 DFS 特征，形状对应为：
            - 单样本：[2K, F, T_stft]（2K=特征数，F=频率Bin数，T_stft=STFT时间帧数）
            - 多样本：[N, 2K, F, T_stft]
        """
        # ===================== 1. 统一输入形状：转为 [N, T, 2K] 多样本格式 =====================
        if data.ndim == 2:
            # 单样本 [T, 2K] -> 扩展为 [1, T, 2K]
            data = np.expand_dims(data, axis=0)
            is_single_sample = True
        elif data.ndim == 3:
            # 多样本 [N, T, 2K]，保持原样
            is_single_sample = False
        else:
            raise ValueError(f"输入数据维度必须是 2（单样本 [T,2K]）或 3（多样本 [N,T,2K]），当前维度: {data.ndim}")

        N, T, K_feat = data.shape  # K_feat = 2K（幅度+相位的总特征数）

        # ===================== 2. 对每个样本、每个特征单独做 STFT =====================
        dfs_result_list = []
        for n in range(N):
            sample_dfs_list = []
            for k in range(K_feat):
                # 取出当前样本的当前特征：一维时间序列 [T,]
                signal_1d = data[n, :, k]
                # 做 STFT 处理
                dfs_1ch = self._transform_single_channel(signal_1d)
                sample_dfs_list.append(dfs_1ch)
            # 拼接当前样本的所有特征：[2K, F, T_stft]
            sample_dfs = np.stack(sample_dfs_list, axis=0)
            dfs_result_list.append(sample_dfs)

        # ===================== 3. 拼接所有样本，恢复单/多样本格式 =====================
        dfs_result = np.stack(dfs_result_list, axis=0)  # [N, 2K, F, T_stft]
        if is_single_sample:
            # 如果是单样本，去掉样本轴：[N, 2K, F, T_stft] -> [2K, F, T_stft]
            dfs_result = dfs_result[0]

        return dfs_result
    
class RobustScaleOp(BasePreprocessOp):
    """
    鲁棒性归一化：使用中位数和四分位距，防止异常值破坏分布对齐
    支持高维输入，默认对【最后一维（时间维度）】做独立归一化
    """
    def __init__(self, norm_axis: int = -1, eps: float = 1e-8):
        """
        :param norm_axis: 归一化轴，默认 -1（最后一维，通常是时间维度）
        :param eps: 防止除零的小常数
        """
        self.norm_axis = norm_axis
        self.eps = eps
        # 保存 fit 阶段学到的统计量
        self.median_: Optional[np.ndarray] = None
        self.iqr_: Optional[np.ndarray] = None

    @property
    def op_name(self) -> str:
        return "robust_norm"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        """
        在训练集上计算统计量并保存
        :param data: 训练集数据，形状任意，norm_axis 指定的维度会被压缩
        """
        # 计算中位数和 IQR，保持非归一化轴的维度
        self.median_ = np.median(data, axis=self.norm_axis, keepdims=True)
        q75 = np.percentile(data, 75, axis=self.norm_axis, keepdims=True)
        q25 = np.percentile(data, 25, axis=self.norm_axis, keepdims=True)
        self.iqr_ = q75 - q25

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        用 fit 阶段保存的统计量做归一化
        :param data: 待归一化数据，形状需与 fit 时的非归一化轴一致
        :return: 归一化后的数据，形状与输入一致
        """
        if self.median_ is None or self.iqr_ is None:
            raise ValueError("必须先调用 fit() 方法，再调用 transform()")
        
        # 复用 fit 阶段的统计量做归一化
        return (data - self.median_) / (self.iqr_ + self.eps)

    def fit_transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        便捷方法：fit + transform（仅用于训练集）
        """
        self.fit(data, **kwargs)
        return self.transform(data, **kwargs)