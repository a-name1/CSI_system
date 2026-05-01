import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ====================== 固定随机种子 ======================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ====================== 数据集（改进填充方式） ======================
class RobustCSIDataset(Dataset):
    def __init__(self, df, root_dir="/root/CSI_system/TaskName",
                 target_frames=360, mean=None, std=None, is_training=False):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.target_frames = target_frames
        self.mean = mean
        self.std = std
        self.is_training = is_training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        full_path = os.path.join(self.root_dir, row['file_path'])

        csi = np.load(full_path)
        amp = np.abs(csi).squeeze(0)

        # 每根天线独立归一化（除以该天线该帧的平均幅值）
        per_antenna_per_frame_mean = np.mean(amp, axis=(1, 2), keepdims=True) + 1e-8
        amp = amp / per_antenna_per_frame_mean

        # 时序对齐
        curr_len = amp.shape[-1]
        if curr_len >= self.target_frames:
            if self.is_training:
                start = np.random.randint(0, curr_len - self.target_frames + 1)
            else:
                start = (curr_len - self.target_frames) // 2
            amp = amp[..., start:start + self.target_frames]
        else:
            # 改进：使用 constant 填充（补0），避免 reflect 引入虚假模式
            pad_width = self.target_frames - curr_len
            amp = np.pad(amp, ((0, 0), (0, 0), (0, pad_width)), mode='constant', constant_values=0)

        x = torch.from_numpy(amp).float()
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-8)

        return x, torch.tensor(row['label'], dtype=torch.long)

# ====================== 统计量计算（与训练截取一致） ======================
def compute_global_stats(dataset):
    """计算全局均值和标准差（使用与训练相同的截取逻辑）"""
    # 强制使用训练模式截取（随机起始），使统计量更能代表训练时看到的数据分布
    dataset.is_training = True
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    sum_, sum_sq, count = None, None, 0
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Computing stats"):
            if sum_ is None:
                sum_ = torch.zeros(x.shape[1:])
                sum_sq = torch.zeros(x.shape[1:])
            sum_ += x.sum(0)
            sum_sq += (x ** 2).sum(0)
            count += x.size(0)
    mean = sum_ / count
    std = torch.sqrt((sum_sq / count) - (mean ** 2) + 1e-8)
    # 恢复原状态
    dataset.is_training = False
    return mean, std

# ====================== 模型（保留子载波维度） ======================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return torch.sum(w * x, dim=1)

class SensingNet(nn.Module):
    def __init__(self, num_classes=5, use_attention=True, gru_hidden_size=64):
        super().__init__()
        self.use_attention = use_attention

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),   # 子载波: 14→7, 时间: T→T/2
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # 在子载波维度上平均池化，保留时间维
        self.subcarrier_pool = nn.AdaptiveAvgPool2d((1, None))

        self.gru = nn.GRU(
            input_size=64,
            hidden_size=gru_hidden_size,
            bidirectional=True,
            batch_first=True
        )

        if self.use_attention:
            self.attention = AttentionLayer(2 * gru_hidden_size)

        self.fc = nn.Linear(2 * gru_hidden_size, num_classes)

    def forward(self, x, return_feat=False):
        if x.dim() == 5:
            x = x.squeeze(1)
        x = self.cnn(x)                     # [B,64,7,T/2]
        x = self.subcarrier_pool(x)         # [B,64,1,T/2]
        x = x.squeeze(2)                    # [B,64,T/2]
        x = x.permute(0, 2, 1)              # [B,T/2,64]
        g, _ = self.gru(x)                  # [B,T/2,2*hidden]
        if self.use_attention:
            feat = self.attention(g)
        else:
            feat = g[:, -1, :]
        out = self.fc(feat)
        return (out, feat) if return_feat else out

# ====================== 训练/评估工具（添加显存释放） ======================
def train_one_fold(model, train_loader, val_loader, device, epochs=15, lr=0.0005):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)  # 去掉 verbose
    best_acc = 0.0
    best_state = None

    for ep in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}", leave=False)
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

        model.eval()
        cor, tot = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                cor += (model(x).argmax(1) == y).sum().item()
                tot += y.size(0)
        acc = cor / tot if tot > 0 else 0
        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return best_acc

def get_all_preds(model, loader, device, num_classes):
    all_labels, all_preds, all_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Inference"):
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# ====================== 绘图工具（修复 ROC 崩溃） ======================
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return cm_norm

def plot_roc_curve(y_true, y_prob, num_classes, save_path):
    y_bin = label_binarize(y_true, classes=range(num_classes))
    # 防止某些类别在 y_true 中缺失导致 AUC 计算错误
    if y_bin.shape[1] < num_classes:
        # 补齐缺失列
        full_bin = np.zeros((len(y_true), num_classes))
        for i in range(y_bin.shape[1]):
            full_bin[:, i] = y_bin[:, i]
        y_bin = full_bin

    fpr, tpr, roc_auc = {}, {}, {}
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        if np.sum(y_bin[:, i]) == 0:
            # 如果该类没有样本，跳过绘制但保留 AUC=0
            roc_auc[i] = 0.0
            continue
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC={roc_auc[i]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Multi-Class ROC Curve (One-vs-Rest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return roc_auc

def save_class_accuracy(y_true, y_pred, save_csv_path):
    cm = confusion_matrix(y_true, y_pred)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    df_cls = pd.DataFrame({
        "ClassID": list(range(len(class_acc))),
        "ClassAccuracy": class_acc
    })
    df_cls.to_csv(save_csv_path, index=False)
    print("\n===== 每类别精度 =====")
    print(df_cls)
    return df_cls

# ====================== 主实验 ======================
def run_experiment():
    seed_everything(42)
    METADATA_PATH = "/root/CSI_system/TaskName/metadata/sample_metadata.csv"
    PLOT_DIR = "/root/CSI_system/plots/CNN_BiGRU_attention_expt_2"
    os.makedirs(PLOT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 5

    df = pd.read_csv(METADATA_PATH)
    df = df[(df["user_id"] == "U02") & (df["environment"] == "E01")].reset_index(drop=True)
    devices = sorted(df['device'].unique().tolist())
    n_dev = len(devices)
    acc_matrix = np.zeros((n_dev, n_dev))

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    for i, train_dev in enumerate(devices):
        dev_df = df[df["device"] == train_dev].reset_index(drop=True)
        print(f"\n🔬 训练设备: {train_dev} | 样本数: {len(dev_df)}")

        if len(dev_df) < 25:
            print(f"⚠️  样本过少，跳过 {train_dev}")
            continue

        # ----- 交叉验证（对角线） -----
        fold_accs = []
        labels = dev_df['label'].values
        for fold, (tr_idx, val_idx) in enumerate(rskf.split(dev_df, labels)):
            tr_data = dev_df.iloc[tr_idx]
            val_data = dev_df.iloc[val_idx]

            m, s = compute_global_stats(RobustCSIDataset(tr_data, is_training=True))
            train_loader = DataLoader(RobustCSIDataset(tr_data, mean=m, std=s, is_training=True), 16, True)
            val_loader = DataLoader(RobustCSIDataset(val_data, mean=m, std=s), 16)

            model = SensingNet(num_classes=num_classes, use_attention=True).to(device)
            best_acc = train_one_fold(model, train_loader, val_loader, device)
            fold_accs.append(best_acc)

            # 释放显存
            del model
            torch.cuda.empty_cache()

        acc_matrix[i, i] = np.mean(fold_accs)
        print(f"✅ 对角线平均ACC: {acc_matrix[i, i]:.4f}")

        # ----- 全量训练（使用全部训练数据，留出10%验证）-----
        train_dev_df, val_dev_df = train_test_split(
            dev_df, test_size=0.1, stratify=dev_df['label'], random_state=42
        )

        full_m, full_s = compute_global_stats(RobustCSIDataset(train_dev_df, is_training=True))
        full_train_loader = DataLoader(
            RobustCSIDataset(train_dev_df, mean=full_m, std=full_s, is_training=True), 16, True
        )
        full_val_loader = DataLoader(
            RobustCSIDataset(val_dev_df, mean=full_m, std=full_s), 16
        )

        final_model = SensingNet(num_classes=num_classes, use_attention=True).to(device)
        train_one_fold(final_model, full_train_loader, full_val_loader, device, epochs=20)

        # 验证集评估
        y_true, y_pred, y_prob = get_all_preds(final_model, full_val_loader, device, num_classes)

        plot_confusion_matrix(y_true, y_pred,
                              class_names=[f'C{k}' for k in range(num_classes)],
                              save_path=os.path.join(PLOT_DIR, f"cm_{train_dev}.png"))
        plot_roc_curve(y_true, y_prob, num_classes,
                       save_path=os.path.join(PLOT_DIR, f"roc_{train_dev}.png"))
        save_class_accuracy(y_true, y_pred,
                            save_csv_path=os.path.join(PLOT_DIR, f"class_acc_{train_dev}.csv"))

        # ----- 跨设备测试 -----
        for j, test_dev in enumerate(devices):
            if i == j:
                continue
            te_df = df[df["device"] == test_dev]
            if len(te_df) == 0:
                continue
            te_loader = DataLoader(RobustCSIDataset(te_df, mean=full_m, std=full_s), 16)
            yt, yp, _ = get_all_preds(final_model, te_loader, device, num_classes)
            acc = (yt == yp).mean()
            acc_matrix[i, j] = acc
            print(f"   {train_dev} → {test_dev}: {acc:.4f}")

        # 释放最终模型显存
        del final_model
        torch.cuda.empty_cache()

    # 保存整体稳定性矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(acc_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=devices, yticklabels=devices)
    plt.title("Cross-Device Stability Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "stability_matrix.png"), dpi=300)
    np.save(os.path.join(PLOT_DIR, "acc_matrix.npy"), acc_matrix)
    print("\n📊 所有图表、类精度CSV已保存到 plots 目录")

if __name__ == "__main__":
    run_experiment()