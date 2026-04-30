import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.signal import butter, filtfilt
from utils.ablation_base import BasePreprocessOp
from typing import Union, Tuple, Optional

# ====================== 频率表 ======================
COMMON_14_FREQ = np.array([-28, -24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24, 28])
FREQ_TABLES = {
    "AmazonEchoPlus": COMMON_14_FREQ,
    "AmazonEchoShow8": COMMON_14_FREQ,
    "AmazonEchoSpot": COMMON_14_FREQ,
    "AmazonPlug": COMMON_14_FREQ,
    "AppleHomePod": COMMON_14_FREQ,
    "EighttreePlug": COMMON_14_FREQ,
    "GoogleNest": COMMON_14_FREQ,
    "GoveeSmartPlug": COMMON_14_FREQ,
    "WyzePlug": COMMON_14_FREQ,
}

def assert_kt(x, name=""):
    assert x.ndim == 2, f"{name} 必须是 [K,T]，得到 {x.shape}"

# ====================== 相位解缠绕 ======================
class PhaseUnwrapOp(BasePreprocessOp):
    def __init__(self, time_axis: int = -1):
        self.time_axis = time_axis

    @property
    def op_name(self) -> str:
        return "phase_unwrap_anchor"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        amp = np.abs(data)
        phase = np.angle(data)
        phase_unwrap = np.unwrap(phase, axis=self.time_axis)
        K = phase.shape[-2]
        T = phase.shape[-1]
        original_shape = phase_unwrap.shape
        p_flat = phase_unwrap.reshape(-1, K, T)
        a_flat = amp.reshape(-1, K, T)
        N_links = p_flat.shape[0]
        result = []

        for link_idx in range(N_links):
            p = p_flat[link_idx].copy()
            a = a_flat[link_idx]
            for t in range(T):
                p_t = p[:, t]
                a_t = a[:, t]
                anchor_idx = np.argmax(a_t)
                p[:, t] = self._anchor_unwrap(p_t, anchor_idx)
            result.append(p)

        phase_final = np.stack(result, axis=0).reshape(original_shape)
        return amp * np.exp(1j * phase_final)

    def _anchor_unwrap(self, p: np.ndarray, anchor_idx: int):
        right = np.unwrap(p[anchor_idx:])
        left = np.flip(np.unwrap(np.flip(p[:anchor_idx + 1])))
        return np.concatenate([left[:-1], right])

# ====================== MIMO 合并 ======================
class MIMOCombineOp(BasePreprocessOp):
    def __init__(self, mode='mrc'):
        self.mode = mode

    @property
    def op_name(self) -> str:
        return "mimo_combine"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        shape = data.shape
        links = data.reshape(-1, shape[-2], shape[-1])
        if self.mode == 'mrc':
            amps = np.abs(links)
            weights = amps ** 2
            weights_norm = weights / (np.sum(weights, axis=0, keepdims=True) + 1e-8)
            combined = np.sum(links * weights_norm, axis=0)
        else:
            combined = np.mean(links, axis=0)
        return combined

# ====================== 共轭相关 ======================
class ConjugateCorrelationOp(BasePreprocessOp):
    def __init__(self, ref_ant_idx=0):
        self.ref_ant_idx = ref_ant_idx

    @property
    def op_name(self) -> str:
        return "conj_correlation"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        Nt, Nr, K, T = data.shape
        h_ref = data[:, [self.ref_ant_idx], :, :]
        features = []
        for i in range(Nr):
            if i == self.ref_ant_idx:
                continue
            corr = data[:, [i], :, :] * np.conj(h_ref)
            features.append(corr)
        return np.stack(features, axis=1)

# ====================== WLS 相位校准 ======================
class EnhancedWLSPhaseOp(BasePreprocessOp):
    @property
    def op_name(self) -> str:
        return "wls_enhanced_no_unwrap"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def transform(self, data, **kwargs) -> np.ndarray:
        if isinstance(data, tuple) and len(data) == 2:
            phase, amp = data
            assert_kt(phase, "WLS phase")
            assert_kt(amp, "WLS amp")
        else:
            phase = np.angle(data)
            amp = np.abs(data)
            assert_kt(phase, "WLS phase")

        device_type = kwargs.get("device_type", "INTEL_14")
        k = FREQ_TABLES.get(device_type, np.arange(phase.shape[0]))
        sanitized = np.zeros_like(phase)
        A = np.vstack([k, np.ones(len(k))]).T

        for t in range(phase.shape[1]):
            y = phase[:, t]
            w = amp[:, t] + 1e-8
            sol = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
            sanitized[:, t] = y - (sol[0] * k + sol[1])

        return signal.detrend(sanitized, axis=1, type='linear')

# ====================== 去噪 ======================
class HampelFilterOp(BasePreprocessOp):
    def __init__(self, k=13, t0=3):
        self.k = k
        self.t0 = t0

    @property
    def op_name(self) -> str:
        return "denoise"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        assert_kt(data, "Hampel")
        is_complex = np.iscomplexobj(data)
        x_out = data.copy()

        for k_idx in range(data.shape[0]):
            series = np.abs(data[k_idx, :])
            win = min(self.k, len(series))
            if win % 2 == 0:
                win -= 1
            if win < 3:
                continue

            med = signal.medfilt(series, kernel_size=win)
            mad = np.median(np.abs(series - med))
            sigma = mad / 1.4826 if mad > 1e-4 else 1e-4
            outliers = np.abs(series - med) > self.t0 * sigma

            if is_complex:
                x_out[k_idx, outliers] = med[outliers] * np.exp(1j * np.angle(data[k_idx, outliers]))
            else:
                x_out[k_idx, outliers] = med[outliers]
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

    def transform(self, data):
        assert_kt(data, "SavGol")
        T = data.shape[1]
        w = min(self.window_length, T)
        if w % 2 == 0:
            w -= 1
        if w <= self.polyorder:
            return data
        return signal.savgol_filter(data, w, self.polyorder, axis=1)

# ====================== AGC ======================
class AGCOp(BasePreprocessOp):
    @property
    def op_name(self) -> str:
        return "agc"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def _low_pass_filter(self, data, cutoff=0.1, fs=100, order=2):
        nyq = 0.5 * fs
        normal_cutoff = np.clip(cutoff / nyq, 0.001, 0.99)
        b, a = butter(order, normal_cutoff, btype='low')
        return filtfilt(b, a, data)

    def _compute_quantization_distortion(self, q):
        if q < 1e-3:
            return 1/12
        d = 1/12
        for k in range(1, 10):
            term = (-1)**k * np.exp(-2 * (np.pi * k * q)**2)
            d += term
        return d

    def transform(self, data: np.ndarray, fs: float = 20.0, **kwargs) -> np.ndarray:
        assert_kt(data, "AGC")
        K, T = data.shape
        if T < 2:
            return data.copy()
        power_linear = np.mean(np.square(data.clip(1e-8)), axis=0)
        gamma_tilde = 10 * np.log10(power_linear + 1e-12)
        lambda_candidates = np.linspace(0.5, 3.0, 6)
        best_J = np.inf
        best_total_gain = None

        for lam in lambda_candidates:
            delta_gamma = np.diff(gamma_tilde)
            if len(delta_gamma) == 0:
                continue
            db = DBSCAN(eps=lam * 0.6, min_samples=1).fit(delta_gamma.reshape(-1, 1))
            delta_quant = np.zeros_like(delta_gamma)
            for lb in np.unique(db.labels_):
                mask = db.labels_ == lb
                c = np.mean(delta_gamma[mask])
                delta_quant[mask] = np.round(c / lam) * lam
            g2 = np.cumsum(np.concatenate([[0], delta_quant]))
            res = gamma_tilde - g2
            g1 = self._low_pass_filter(res, fs=fs)
            total = g1 + g2
            sigma_hat_sq = np.var(gamma_tilde - total)
            sigma_hat = np.sqrt(sigma_hat_sq) if sigma_hat_sq > 0 else 1e-6
            q = lam / sigma_hat
            Dq = self._compute_quantization_distortion(q)
            J = sigma_hat_sq + lam**2 * Dq
            if J < best_J:
                best_J = J
                best_total_gain = total

        if best_total_gain is None:
            return data.copy()
        corr = 10 ** (best_total_gain / 20)
        out = data / (corr + 1e-6)
        return np.nan_to_num(out, nan=1, posinf=1, neginf=1)

# ====================== 静态消除 ======================
class ButterworthStaticRemovalOp(BasePreprocessOp):
    def __init__(self, mode="normalized", normalized_cutoff=0.006, cutoff_freq=0.3, fs=100, order=1):
        self.mode = mode
        self.norm_cut = normalized_cutoff
        self.cutoff_hz = cutoff_freq
        self.fs = fs
        self.order = order

    @property
    def op_name(self) -> str:
        return "butterworth_static_removal"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        assert_kt(data, "Butterworth")
        res = data.copy()
        T = res.shape[-1]
        if T < 3 * self.order:
            return res

        if self.mode == "normalized":
            norm = np.clip(self.norm_cut, 0.001, 0.99)
        elif self.mode == "timestamp":
            t_sec = kwargs["t_sec"]
            fs_local = 1.0 / np.mean(np.diff(t_sec))
            norm = np.clip(self.cutoff_hz / (0.5 * fs_local), 0.001, 0.99)
        else:
            norm = np.clip(self.cutoff_hz / (0.5 * self.fs), 0.001, 0.99)

        b, a = butter(self.order, norm, btype="high")
        return filtfilt(b, a, res, axis=-1)

# ====================== 时域重采样 ======================
class TimeResizeOp(BasePreprocessOp):
    def __init__(self, target_len=360):
        self.target_len = target_len

    @property
    def op_name(self) -> str:
        return "resample"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        assert_kt(data, "TimeResize")
        K, T = data.shape
        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, self.target_len)
        f = interp1d(x_old, data, axis=1, kind='linear', fill_value="extrapolate", bounds_error=False)
        return f(x_new)

class PadOnlyTimeResizeOp(BasePreprocessOp):
    def __init__(self, target_len=360):
        self.target_len = target_len

    @property
    def op_name(self):
        return "pad_only_resample"

    def fit(self, data, **kwargs): pass

    def transform(self, data, **kwargs):
        assert_kt(data, "PadOnlyResize")
        K, T = data.shape
        if T >= self.target_len:
            return data[:, :self.target_len]
        pad = self.target_len - T
        return np.pad(data, ((0,0),(0,pad)), mode='edge')

# ====================== 频域对齐 ======================
class SplineFreqAlignOp(BasePreprocessOp):
    def __init__(self, target_k=30):
        self.target_k = target_k

    @property
    def op_name(self):
        return "spline"

    def fit(self, data, **kwargs):
        pass

    def transform(self, data, **kwargs):
        assert_kt(data, "SplineFreqAlign")
        raw_freq = kwargs["raw_freq"]
        target_freq = np.linspace(raw_freq[0], raw_freq[-1], self.target_k)
        f = interp1d(raw_freq, data, axis=0, kind='cubic', fill_value="extrapolate", bounds_error=False)
        return f(target_freq)

class LinearFreqResizeOp(BasePreprocessOp):
    def __init__(self, target_k=30):
        self.target_k = target_k

    @property
    def op_name(self):
        return "linear_resize"

    def fit(self, data, **kwargs):
        pass

    def transform(self, data, **kwargs):
        assert_kt(data, "LinearFreqResize")
        raw_freq = kwargs["raw_freq"]
        target_freq = np.linspace(raw_freq[0], raw_freq[-1], self.target_k)
        f = interp1d(raw_freq, data, axis=0, kind='linear', fill_value="extrapolate", bounds_error=False)
        return f(target_freq)

# ====================== PCA (修复版) ======================
class PostSTFTPCAOp(BasePreprocessOp):
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.pca = None

    @property
    def op_name(self) -> str:
        return "post_stft_pca"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def transform(self, stft_maps: np.ndarray) -> np.ndarray:
        """
        stft_maps: [K, F, T] complex
        return: [C, F, T] complex
        """
        K, F, T = stft_maps.shape
        X = stft_maps.reshape(K, -1).T
        X_real = np.concatenate([X.real, X.imag], axis=1)

        pca = PCA(n_components=self.n_components)
        Xr = pca.fit_transform(X_real)
        Xr_complex = Xr.T.reshape(self.n_components, F, T).astype(np.complex64)
        return Xr_complex

# ====================== STFT (自适应无硬编码) ======================
class PerCarrierSTFTOp(BasePreprocessOp):
    def __init__(self, fs=100, nperseg=64, noverlap=48, nfft=256, target_freq_bins=64):
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.target_freq_bins = target_freq_bins

    @property
    def op_name(self) -> str:
        return "per_carrier_stft"

    def fit(self, data, **kwargs):
        pass

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        data: [K, T] complex
        return: [K, F, T] complex
        """
        assert_kt(data, "PerCarrierSTFT")
        K, T = data.shape
        maps = []

        for k in range(K):
            f, t, Zxx = signal.stft(
                data[k],
                fs=self.fs,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                nfft=self.nfft,
                boundary=None,
                detrend=False,
                return_onesided=False
            )
            Zxx = np.fft.fftshift(Zxx, axes=0)
            maps.append(Zxx)

        maps = np.stack(maps, axis=0)
        F_total = maps.shape[1]
        f_mid = F_total // 2
        half = self.target_freq_bins // 2
        return maps[:, f_mid - half : f_mid + half, :]

# ====================== 归一化 ======================
class ZScoreNormOp(BasePreprocessOp):
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    @property
    def op_name(self) -> str:
        return "zscore"

    def fit(self, data: np.ndarray, **kwargs) -> None:
        self.mean = np.mean(data)
        self.std = np.std(data) + 1e-8

    def transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return (data - self.mean) / self.std

class STFTCarrierNormOp(BasePreprocessOp):
    @property
    def op_name(self):
        return "stft_carrier_norm"

    def fit(self, data, **kwargs):
        pass

    def transform(self, data, **kwargs):
        energy = np.mean(np.abs(data)**2, axis=(1,2), keepdims=True)
        return data / (np.sqrt(energy) + 1e-8)

class ComplexToChannelsOp(BasePreprocessOp):
    @property
    def op_name(self):
        return "complex_to_channels"

    def fit(self, data, **kwargs): pass

    def transform(self, data, **kwargs):
        return np.concatenate([np.real(data), np.imag(data)], axis=0)