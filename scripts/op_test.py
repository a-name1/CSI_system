import numpy as np
import pandas as pd
import os
from pathlib import Path
import traceback
import sys

# ========= 1. 路径挂载 =========
current_dir = Path(__file__).resolve().parent 
root_dir = current_dir.parent 
utils_dir = root_dir / "utils"

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(utils_dir) not in sys.path:
    sys.path.append(str(utils_dir))

from t_sne_mmd_auto import plot_tsne_comparison
from utils.preprocess_ops import *

# ========= 2. 算子适配器 =========
def apply_isolated_op(op, csi, time_sec):
    x = csi.copy()
    
    # 基础维度处理：如果不是多天线/共轭处理，先对天线维度取均值
    if x.ndim > 2:
        if op is None or not isinstance(op, (MIMOCombineOp, ConjugateCorrelationOp)):
            x = np.mean(x, axis=(0, 1))
    
    if op is None:
        return x    
    
    # 特殊算子逻辑适配
    if isinstance(op, EnhancedWLSPhaseOp):
        return op.transform((np.angle(x), np.abs(x)), device_type="INTEL_14")
    
    if isinstance(op, (KaiserResampleOp, SplineFreqAlignOp, PCAOp)):
        # 确保时间轴 T 为第一维 [T, K]
        if x.shape[0] < x.shape[1]: 
            x = x.T 
        
        # PCA 不支持复数，必须先取绝对值
        if isinstance(op, PCAOp):
            if np.iscomplexobj(x):
                x = np.abs(x)
            op.fit([x]) 
            
        return op.transform(x, t_sec=time_sec)
    
    # STFT 动态参数调整
    if isinstance(op, STFTOp):
        T_len = x.shape[0] if x.shape[0] < x.shape[1] else x.shape[1]
        if hasattr(op, 'nperseg') and op.nperseg > T_len:
            op.nperseg = T_len
        if hasattr(op, 'noverlap') and op.noverlap >= op.nperseg:
            op.noverlap = op.nperseg // 2 
            
    return op.transform(x)

# ========= 3. 数据加载 (按用户、环境、设备) =========
def load_all_device_data(base_path, filter_user="U02", filter_env=None):
    """
    按文件名解析 {User}_{Env}_{Device} 逻辑加载数据
    """
    device_map = {
        "AmazonEchoSpot": 0,
        "AmazonPlug": 1
    }
    all_data_list = []
    
    if not os.path.exists(base_path):
        print(f"❌ 路径不存在: {base_path}")
        return []

    print(f"🔍 正在从 {base_path} 检索样本...")
    
    for fname in os.listdir(base_path):
        if not fname.endswith(".npz"):
            continue
            
        # 解析文件名元数据
        clean_name = fname.replace("_feat.npz", "").replace(".npz", "")
        parts = clean_name.split("_")
        
        if len(parts) < 3:
            continue
            
        meta = {"user": parts[0], "env": parts[1], "device": parts[2]}
        
        # 过滤
        if filter_user and meta["user"] != filter_user:
            continue
        if filter_env and meta["env"] != filter_env:
            continue
            
        if meta["device"] in device_map:
            dev_idx = device_map[meta["device"]]
        else:
            continue

        file_path = os.path.join(base_path, fname)
        try:
            with np.load(file_path, allow_pickle=True) as data:
                # 兼容存储格式
                if 'samples' in data:
                    samples = data['samples']
                    for s in samples:
                        all_data_list.append({
                            'csi': s['raw_feature'],
                            'label': s['label'],
                            'time': s.get('time', 0),
                            'dev_idx': dev_idx
                        })
                elif 'x' in data and 'y' in data:
                    xs, ys = data['x'], data['y']
                    ts = data['time'] if 'time' in data else [0] * len(xs)
                    for x, y, t in zip(xs, ys, ts):
                        all_data_list.append({
                            'csi': x,
                            'label': y,
                            'time': t,
                            'dev_idx': dev_idx
                        })
            print(f"✅ 已加载 {fname} | 总样本: {len(all_data_list)}")
        except Exception as e:
            print(f"❌ 加载文件 {fname} 出错: {e}")

    return all_data_list

# ========= 4. 运行消融实验主流程 =========
def run_ablation_study():
    BASE_PATH = "/root/CSI_system/sample_cross_dev"
    SAVE_DIR = Path("/root/CSI_system/ablation_results")
    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    
    # 1. 加载数据 (默认 U02 用户)
    combined_samples = load_all_device_data(BASE_PATH, filter_user="U02")

    if not combined_samples:
        print("❌ 错误：未加载到任何数据，请检查路径和文件名格式。")
        return

    # 2. 定义消融实验组
    ablation_cases = {
        "Full_Proposed": [
            MIMOCombineOp(mode='mrc'),
            HampelFilterOp(),
            SavitzkyGolayOp(),
            EnhancedWLSPhaseOp(),
            PCAOp(n_components=3)
        ],
        "W/O_MIMO_Combine": [
            HampelFilterOp(),
            SavitzkyGolayOp(),
            EnhancedWLSPhaseOp(),
            PCAOp(n_components=3)
        ],
        "W/O_Denoise": [
            MIMOCombineOp(mode='mrc'),
            EnhancedWLSPhaseOp(),
            PCAOp(n_components=3)
        ],
        "W/O_Phase_Calib": [
            MIMOCombineOp(mode='mrc'),
            HampelFilterOp(),
            SavitzkyGolayOp(),
            PCAOp(n_components=3)
        ],
        "Raw_Baseline": []
    }

    report = []
    # 动作映射
    ACTION_LABELS = {0: "walking", 1: "seated-breathing", 2: "jumping", 3: "wavinghand", 4: "running"}

    for name, ops in ablation_cases.items():
        print(f"\n>>> 正在测试消融组: {name}")
        all_feats, all_dev_labels, all_act_labels = [], [], []
        TARGET_FLAT_LEN = None 

        for sample in combined_samples:
            feat = sample['csi'].copy()
            t_val = sample['time']
            dev_idx = sample['dev_idx']
            act_label = sample['label']
            
            # 顺序执行算子链
            if not ops:
                feat = apply_isolated_op(None, feat, t_val)
            else:
                for op in ops:
                    feat = apply_isolated_op(op, feat, t_val)
            
            # 取模处理复数
            if np.iscomplexobj(feat):
                feat = np.abs(feat)
            
            flat_feat = feat.flatten()
            
            # 动态维度对齐
            if TARGET_FLAT_LEN is None:
                TARGET_FLAT_LEN = len(flat_feat)
            
            if len(flat_feat) != TARGET_FLAT_LEN:
                if len(flat_feat) > TARGET_FLAT_LEN:
                    flat_feat = flat_feat[:TARGET_FLAT_LEN]
                else:
                    flat_feat = np.pad(flat_feat, (0, TARGET_FLAT_LEN - len(flat_feat)), 'constant')
            
            all_feats.append(flat_feat)
            all_dev_labels.append(dev_idx)
            all_act_labels.append(act_label)

        X = np.array(all_feats)
        y = np.array(all_act_labels)
        d = np.array(all_dev_labels)
        
        try:
            # 绘图评估：注意此处调用的是 plot_tsne_comparison
            stats = plot_tsne_comparison(X, y, d, f"Ablation: {name}", SAVE_DIR/f"ablation_{name}.png", name, ACTION_LABELS)
            report.append(stats)
            
            mmd = stats.get('MMD', 0)
            fisher = stats.get('Act_Fisher', 0)
            print(f"完成! MMD: {mmd:.4f} | Fisher: {fisher:.4f}")
                
        except Exception as e:
            print(f"评估失败 {name}: {e}")
            traceback.print_exc()

    # 5. 保存 CSV 报表
    pd.DataFrame(report).to_csv(SAVE_DIR / "ablation_study_report.csv", index=False)
    print(f"\n✅ 任务结束。结果目录: {SAVE_DIR}")

if __name__ == "__main__":
    run_ablation_study()