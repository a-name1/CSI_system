import os
import re
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
from datetime import datetime, timedelta

# ======================
# 1. CSI 读取与时间解析
# ======================
def extract_csi_payload(file_path):
    try:
        data = sio.loadmat(file_path)
        trace = data['csi_trace']
        csi_raw = trace['csi'][0, 0] # 原始复数数据 [1, 4, 14, N]
        timer_raw = trace['mactimer'][0, 0].flatten().astype(np.float64)
        return csi_raw, timer_raw
    except Exception:
        return None, None

def parse_groundtruth(gt_path, label_map):
    gt_sheets = pd.read_excel(gt_path, sheet_name=None)
    valid_dfs = []
    for sheet_name, df in gt_sheets.items():
        if df.empty or "Sheet" in sheet_name: continue
        uid = sheet_name.split("_")[0]
        df.columns = [c.strip() for c in df.columns]
        # 仅筛选 1.5m 距离的标准实验数据[cite: 2]
        df = df[df["Info"].astype(str).str.contains("1.5m", na=False)].copy()
        df["label_str"] = df["Event"].astype(str).str.strip().str.lower()
        df = df[df["label_str"].isin(label_map.keys())]

        def parse_dt(row):
            try:
                date_str = str(row['Date']).split(' ')[0]
                start = pd.to_datetime(f"{date_str} {row['Start Time (UTC)']}")
                end = pd.to_datetime(f"{date_str} {row['End Time (UTC)']}")
                return start, end
            except: return pd.NaT, pd.NaT

        df[['t_start', 't_end']] = df.apply(parse_dt, axis=1, result_type='expand')
        df["user"] = uid
        valid_dfs.append(df.dropna(subset=["t_start", "t_end"]))
    return pd.concat(valid_dfs, ignore_index=True)

# ======================
# 2. 自动化提取引擎
# ======================
def run_extraction():
    # --- 配置参数 ---
    DATA_ROOT = "/root/CSI_system/data"
    RAW_DIR = os.path.join(DATA_ROOT, "RawContinuousRecording")
    GT_PATH = os.path.join(RAW_DIR, "Groundtruth.xlsx")
    TASK_ROOT = "/root/CSI_system/TaskName"
    
    LABEL_MAP = {"walking":0, "seated-breathing":1, "jumping":2, "wavinghand":3, "running":4}
    target_users = ["U02"]
    target_envs = ["E01"]
    target_devices = ['AmazonEchoPlus','AmazonEchoShow8','AmazonEchoSpot','AmazonPlug',
                      'AppleHomePod','GoogleNest','GoveeSmartPlug','WyzePlug']

    # --- 开始处理 ---
    gt = parse_groundtruth(GT_PATH, LABEL_MAP)
    metadata = []
    sample_idx = 0
    stats = {}

    for udir in os.listdir(RAW_DIR):
        m = re.match(r"(U\d+)_(E\d+)", udir)
        if not m: continue
        user, env = m.group(1), m.group(2)
        if user not in target_users or env not in target_envs: continue

        dev_root = os.path.join(RAW_DIR, udir)
        for dev in os.listdir(dev_root):
            if dev not in target_devices: continue
            dev_dir = os.path.join(dev_root, dev)
            mat_files = [f for f in os.listdir(dev_dir) if f.endswith(".mat")]

            for mat in tqdm(mat_files, desc=f"Extracting {user}-{env}-{dev}"):
                csi_raw, timer_raw = extract_csi_payload(os.path.join(dev_dir, mat))
                if csi_raw is None or np.all(csi_raw == 0): continue

                # 采样率计算与时间校准[cite: 2]
                diffs = np.diff(timer_raw)
                diffs[diffs < 0] += (2**32) # 处理 mactimer 溢出[cite: 2]
                duration = np.sum(diffs) / 1e6
                if duration < 0.5: continue
                fs = len(timer_raw) / duration

                try:
                    ts_str = mat.split("-")[1].split("_")[0]
                    file_end = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
                    file_start = file_end - timedelta(seconds=duration)
                except: continue

                # 匹配动作区间[cite: 2]
                matches = gt[(gt["user"] == user) & (gt["t_start"] < file_end) & (gt["t_end"] > file_start)]

                for _, row in matches.iterrows():
                    # 增加 0.2s Buffer 确保动作完整性[cite: 2]
                    s_off = max(0, (row["t_start"] - file_start).total_seconds() - 0.2)
                    e_off = min(duration, (row["t_end"] - file_start).total_seconds() + 0.2)
                    
                    idx_s, idx_e = int(s_off * fs), int(e_off * fs)
                    if (idx_e - idx_s) < 30: continue

                    # 提取原始切片 (保留复数信息)[cite: 2]
                    # 维度保持为 [1, 4, 14, N] 或 [4, 14, N] 取决于 squeeze 需求
                    seg = csi_raw[:, :, :, idx_s:idx_e]

                    # 保存样本 (使用 .npy 格式保留复数类型)[cite: 2]
                    sample_id = f"H_{user}_{env}_{dev}_{row['label_str']}_{sample_idx:06d}"
                    save_dir = os.path.join(TASK_ROOT, "samples", f"user_{user}", f"dev_{dev}")
                    os.makedirs(save_dir, exist_ok=True)
                    
                    save_path = os.path.join(save_dir, f"{sample_id}.npy")
                    np.save(save_path, seg) # 保存原始 Complex64/128 数据[cite: 2]

                    # 统计与记录
                    stats[row['label_str']] = stats.get(row['label_str'], 0) + 1
                    metadata.append({
                        "sample_id": sample_id, "user_id": user, "activity": row['label_str'],
                        "label": LABEL_MAP[row['label_str']], "environment": env,
                        "device": dev, "file_path": os.path.relpath(save_path, TASK_ROOT)
                    })
                    sample_idx += 1

    # 保存元数据
    meta_df = pd.DataFrame(metadata)
    os.makedirs(os.path.join(TASK_ROOT, "metadata"), exist_ok=True)
    meta_df.to_csv(os.path.join(TASK_ROOT, "metadata", "sample_metadata.csv"), index=False)
    print(f"✅ 原始数据提取完成！类别分布: {stats}")

if __name__ == "__main__":
    run_extraction()