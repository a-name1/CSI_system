import os
import re
import json
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
from datetime import datetime, timedelta

# ======================
# 【你的原版CSI提取 → 100%不变】
# ======================
def extract_csi_payload(file_path):
    """从mat文件读取原始复数CSI和时间戳"""
    try:
        data = sio.loadmat(file_path)
        trace = data['csi_trace']
        csi_raw = trace['csi'][0, 0]
        timer_raw = trace['mactimer'][0, 0].flatten().astype(np.float64)
        return csi_raw, timer_raw
    except Exception:
        return None, None

# ======================
# 【你的原版标注解析 → 100%不变】
# ======================
def parse_groundtruth():
    print(f"🔍 解析标注文件: {GT_PATH}")
    gt_sheets = pd.read_excel(GT_PATH, sheet_name=None)
    valid_dfs = []

    for sheet_name, df in gt_sheets.items():
        if df.empty or "Sheet" in sheet_name:
            continue
        df = df.copy()
        uid = sheet_name.split("_")[0]
        df.columns = [c.strip() for c in df.columns]

        required_cols = ['Event', 'Info', 'Date', 'Start Time (UTC)', 'End Time (UTC)']
        if not all(col in df.columns for col in required_cols):
            continue
        df = df[df["Info"].astype(str).str.contains("1.5m", na=False)]

        df["label_str"] = df["Event"].astype(str).str.strip().str.lower()
        df = df[df["label_str"].isin(LABEL_MAP.keys())]

        def parse_dt(row):
            try:
                date_str = str(row['Date']).split(' ')[0]
                start_str = str(row['Start Time (UTC)']).strip()
                end_str = str(row['End Time (UTC)']).strip()
                return pd.to_datetime(f"{date_str} {start_str}"), pd.to_datetime(f"{date_str} {end_str}")
            except:
                return pd.NaT, pd.NaT

        df[['t_start', 't_end']] = df.apply(parse_dt, axis=1, result_type='expand')
        df["user"] = uid
        df = df.dropna(subset=["t_start", "t_end"])
        
        if not df.empty:
            valid_dfs.append(df)
            print(f"✅ {sheet_name}: {len(df)} 条标注")
    return pd.concat(valid_dfs, ignore_index=True) if valid_dfs else pd.DataFrame()

# ======================
# 【核心修改：NPZ 格式存储，替换 H5】
# ======================
def extract_and_save_samples(gt):
    pattern = re.compile(r"^(U\d+)_(E\d+)")
    all_dirs = os.listdir(RAW_DIR)
    selected_dirs = []
    for d in all_dirs:
        match = pattern.match(d)
        if match:
            u, e = match.group(1), match.group(2)
            if u in target_users and e in target_envs:
                selected_dirs.append((d, u, e))

    metadata = []
    sample_idx = 0

    for udir, current_user, current_env in tqdm(selected_dirs, desc="处理用户-环境"):
        dev_root = os.path.join(RAW_DIR, udir)
        all_devices = [d for d in os.listdir(dev_root) if os.path.isdir(os.path.join(dev_root, d))]
        devices = [d for d in all_devices if d in target_devices]

        for dev in devices:
            dev_dir = os.path.join(dev_root, dev)
            mat_files = [f for f in os.listdir(dev_dir) if f.endswith(".mat")]

            for mat in tqdm(mat_files, desc=f"{current_user}-{current_env}-{dev}", leave=False):    
                file_path = os.path.join(dev_dir, mat)
                csi_raw, timer_raw = extract_csi_payload(file_path)
                if csi_raw is None:
                    continue

                # 你的原版时间/采样率计算 → 100%不变
                diffs = np.diff(timer_raw)
                diffs[diffs < 0] += (2**32)
                duration = np.sum(diffs) / 1e6
                fs = len(timer_raw) / duration if duration > 0 else 100
                if duration < 0.5:
                    continue

                # 你的原版时间解析 → 100%不变
                try:
                    ts_str = mat.split("-")[1].split("_")[0]
                    file_end = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
                    file_start = file_end - timedelta(seconds=duration)
                except:
                    continue

                # 你的原版标注匹配 → 100%不变
                matches = gt[(gt["user"] == current_user) & 
                             (gt["t_start"] < file_end) & 
                             (gt["t_end"] > file_start)]

                for _, row in matches.iterrows():
                    start_off = max(0, (row["t_start"] - file_start).total_seconds())
                    end_off = min(duration, (row["t_end"] - file_start).total_seconds())
                    
                    idx_s, idx_e = int(start_off * fs), int(end_off * fs)
                    idx_s = max(0, idx_s)
                    idx_e = min(csi_raw.shape[-1], idx_e, len(timer_raw))

                    if (idx_e - idx_s) < 30:
                        continue

                    # ======================
                    # 🔥 核心替换：NPZ 压缩保存（替换 H5）
                    # ======================
                    seg = csi_raw[:, :, :, idx_s:idx_e]
                    t_slice = timer_raw[idx_s:idx_e]
                    act_label = row["label_str"]
                    label_idx = LABEL_MAP[act_label]
                    min_len = min(seg.shape[-1], len(t_slice))
                    
                    # 文件名改为 .npz
                    timestamp = ts_str
                    freq = int(round(fs))
                    session_filename = f"session_{timestamp}__freq{freq}.npz"
                    
                    # 目录结构完全不变
                    save_dir = os.path.join(
                        TASK_ROOT, "sub_Human",
                        f"user_{current_user}",
                        f"act_{act_label}",
                        f"env_{current_env}",
                        f"device_{dev}"
                    )
                    os.makedirs(save_dir, exist_ok=True)
                    npz_save_path = os.path.join(save_dir, session_filename)

                    # ✅ NPZ 保存：csi + timestamp + label + activity
                    np.savez_compressed(
                        npz_save_path,
                        csi=seg[:, :, :, :min_len],
                        timestamp=t_slice[:min_len],
                        label=label_idx,
                        activity=act_label
                    )

                    # 元数据完全不变（仅后缀改为 npz）
                    relative_path = os.path.relpath(npz_save_path, TASK_ROOT)
                    sample_id = f"Human_{current_user}_{act_label}_{current_env}_{dev}_{sample_idx:06d}"
                    sample_idx += 1

                    metadata.append({
                        "sample_id": sample_id,
                        "user_id": current_user,
                        "activity": act_label,
                        "label": label_idx,
                        "environment": current_env,
                        "device": dev,
                        "frequency": freq,
                        "file_path": relative_path
                    })

    # 保存元数据
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(os.path.join(METADATA_DIR, "sample_metadata.csv"), index=False, encoding="utf-8")
    
    # 保存标签映射
    with open(os.path.join(METADATA_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 样本存储完成 | 总样本数：{len(metadata)} | 格式：NPZ")
    return meta_df

# ======================
# 【你的原版数据集划分 → 100%不变】
# ======================
def generate_benchmark_splits(meta_df):
    TRAIN_DEVICES = {"AppleHomePod","EighttreePlug","GoogleNest","GoveeSmartPlug","HealthPod1","HealthPod2","HealthPod3","WyzePlug","AmazonEchoPlus","AmazonEchoShow8"}
    TEST_DEVICES = {"AmazonPlug", "AmazonEchoSpot"}
    TRAIN_USERS = {"U01","U03","U04","U05","U06"}
    TEST_USERS = {"U02"}
    TRAIN_ENVS = {"E01","E02","E03","E04","E06"}
    TEST_ENVS = {"E05"}

    train_mask = (
        meta_df["user_id"].isin(TRAIN_USERS) &
        meta_df["environment"].isin(TRAIN_ENVS) &
        meta_df["device"].isin(TRAIN_DEVICES)
    )
    train_val_df = meta_df[train_mask]

    from sklearn.model_selection import train_test_split
    train_val_ids, test_id = train_test_split(train_val_df["sample_id"].tolist(), test_size=0.15, random_state=42)
    train, val = train_test_split(train_val_ids, test_size=0.176, random_state=42)

    test_cross_device = meta_df[meta_df["device"].isin(TEST_DEVICES)]["sample_id"].tolist()
    test_cross_user = meta_df[meta_df["user_id"].isin(TEST_USERS)]["sample_id"].tolist()
    test_cross_env = meta_df[meta_df["environment"].isin(TEST_ENVS)]["sample_id"].tolist()

    splits = {
        "train_id": train, "val_id": val, "test_id": test_id,
        "test_cross_device": test_cross_device,
        "test_cross_user": test_cross_user,
        "test_cross_env": test_cross_env
    }

    for name, ids in splits.items():
        with open(os.path.join(SPLITS_DIR, f"{name}.json"), "w") as f:
            json.dump({"sample_ids": ids}, f, indent=2)

    print("✅ 标准划分文件已生成")

# ======================
# 【主函数 → 100%不变】
# ======================
if __name__ == "__main__":
    DATA_ROOT = "/root/CSI_system/data"
    RAW_DIR = os.path.join(DATA_ROOT, "RawContinuousRecording")
    GT_PATH = os.path.join(RAW_DIR, "Groundtruth.xlsx")

    # 标准数据集输出目录
    TASK_ROOT = "/root/CSI_system/TaskName"
    METADATA_DIR = os.path.join(TASK_ROOT, "metadata")
    SPLITS_DIR = os.path.join(TASK_ROOT, "splits")
    os.makedirs(METADATA_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)

    LABEL_MAP = {"walking":0, "seated-breathing":1, "jumping":2, "wavinghand":3, "running":4}
    target_users = ["U01","U02","U03","U04","U05","U06"]
    target_envs = ["E01","E02","E03","E04","E05","E06"]
    target_devices = ['AmazonEchoPlus','AmazonEchoShow8','AmazonEchoSpot','AmazonPlug','AppleHomePod','EighttreePlug','GoogleNest','GoveeSmartPlug','WyzePlug']

    # 执行
    gt = parse_groundtruth()
    input("\n按回车开始提取数据...")
    meta_df = extract_and_save_samples(gt)
    input("\n按回车生成划分文件...")
    generate_benchmark_splits(meta_df)
    print("\n🎉 TaskName 标准数据集构建完成！(NPZ 格式)")