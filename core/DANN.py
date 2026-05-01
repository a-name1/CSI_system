# -*- coding: utf-8 -*-
# ===================== 论文级绘图环境（中文 + 服务器） =====================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

FONT_MAIN = 24
FONT_AXIS = 18
FONT_TICK = 14

# 其他导入
import os
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ====================== 动作标签映射（中文） ======================
LABEL_MAP = {"walking": 0, "seated-breathing": 1, "jumping": 2, "wavinghand": 3, "running": 4}
class_names_zh = ["步行", "坐姿呼吸", "跳跃", "挥手", "跑步"]
num_classes = len(class_names_zh)

# ====================== 固定随机种子 ======================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ====================== 数据集 ======================
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
        per_antenna_per_frame_mean = np.mean(amp, axis=(1, 2), keepdims=True) + 1e-8
        amp = amp / per_antenna_per_frame_mean
        curr_len = amp.shape[-1]
        if curr_len >= self.target_frames:
            if self.is_training:
                start = np.random.randint(0, curr_len - self.target_frames + 1)
            else:
                start = (curr_len - self.target_frames) // 2
            amp = amp[..., start:start+self.target_frames]
        else:
            pad_width = self.target_frames - curr_len
            amp = np.pad(amp, ((0,0),(0,0),(0,pad_width)), mode='constant', constant_values=0)
        x = torch.from_numpy(amp).float()
        if self.mean is not None and self.std is not None:
            mean_ = self.mean[:, :, None]
            std_ = self.std[:, :, None]
            x = (x - mean_) / (std_ + 1e-8)
        return x, torch.tensor(row['label'], dtype=torch.long)

# ====================== 统计量计算 ======================
def compute_global_stats(dataset, batch_size=32):
    dataset.is_training = True
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    sum_ = None
    sum_sq = None
    total_elements = 0
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="计算统计量"):
            if sum_ is None:
                sum_ = torch.zeros(x.shape[1:3])
                sum_sq = torch.zeros(x.shape[1:3])
            sum_ += x.sum(dim=(0, 3))
            sum_sq += (x ** 2).sum(dim=(0, 3))
            total_elements += x.size(0) * x.size(3)
    mean = sum_ / total_elements
    std = torch.sqrt((sum_sq / total_elements) - (mean ** 2) + 1e-8)
    dataset.is_training = False
    return mean, std

# ====================== 梯度反转层 ======================
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

# ====================== 最优模型：CNN + 时间全局平均池化 + DANN ======================
class OptimalDANN(nn.Module):
    def __init__(self, num_classes=5, domain_hidden_size=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.subcarrier_pool = nn.AdaptiveAvgPool2d((1, None))
        self.time_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)
        self.domain_classifier = nn.Sequential(
            nn.Linear(64, domain_hidden_size),
            nn.ReLU(),
            nn.Linear(domain_hidden_size, 2)
        )

    def forward(self, x, alpha=None, return_feat=False):
        x = self.cnn(x)
        x = self.subcarrier_pool(x)
        x = x.squeeze(2)
        feat = self.time_pool(x).squeeze(-1)
        class_out = self.fc(feat)
        if alpha is not None:
            rev_feat = GradientReversal.apply(feat, alpha)
            domain_out = self.domain_classifier(rev_feat)
            return class_out, domain_out, feat
        else:
            return (class_out, feat) if return_feat else class_out

# ====================== DANN 训练函数 ======================
def train_dann_for_target(source_loader, target_loader, val_loader, device,
                          epochs=20, lr=0.0005, lambda_domain=0.1):
    model = OptimalDANN(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_acc = 0.0
    best_state = None
    use_dann = lambda_domain > 0

    len_source = len(source_loader)
    if use_dann:
        len_target = len(target_loader)
        max_len = max(len_source, len_target)
        total_iters = epochs * max_len
        iter_target = iter(target_loader)
    else:
        max_len = len_source
        total_iters = epochs * max_len

    iter_source = iter(source_loader)
    current_iter = 0

    for ep in range(epochs):
        model.train()
        loop = tqdm(range(max_len), desc=f"Epoch {ep+1}/{epochs}", leave=False)
        for _ in loop:
            try:
                x_s, y_s = next(iter_source)
            except StopIteration:
                iter_source = iter(source_loader)
                x_s, y_s = next(iter_source)
            x_s, y_s = x_s.to(device), y_s.to(device)

            if use_dann:
                p = current_iter / total_iters
                alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
                current_iter += 1

                try:
                    x_t, _ = next(iter_target)
                except StopIteration:
                    iter_target = iter(target_loader)
                    x_t, _ = next(iter_target)
                x_t = x_t.to(device)

                domain_label_s = torch.zeros(x_s.size(0), dtype=torch.long, device=device)
                domain_label_t = torch.ones(x_t.size(0), dtype=torch.long, device=device)

                class_out_s, domain_out_s, _ = model(x_s, alpha=alpha)
                _, domain_out_t, _ = model(x_t, alpha=alpha)

                loss_class = criterion_class(class_out_s, y_s)
                loss_domain = (criterion_domain(domain_out_s, domain_label_s) +
                               criterion_domain(domain_out_t, domain_label_t)) / 2
                total_loss = loss_class + lambda_domain * loss_domain
                loop.set_postfix(cls=loss_class.item(), dom=loss_domain.item())
            else:
                class_out_s = model(x_s)
                loss_class = criterion_class(class_out_s, y_s)
                total_loss = loss_class
                loop.set_postfix(cls=loss_class.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total if total > 0 else 0
        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_acc

# ====================== 评估函数 ======================
def evaluate_model(model, loader, device):
    model.eval()
    labels, preds, probs = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="推理"):
            x = x.to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1)
            pred = p.argmax(1)
            labels.extend(y.cpu().numpy())
            preds.extend(pred.cpu().numpy())
            probs.extend(p.cpu().numpy())
    return np.array(labels), np.array(preds), np.array(probs)

# ====================== 中文绘图函数 ======================
def plot_confusion_matrix(y_true, y_pred, class_names_zh, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names_zh, yticklabels=class_names_zh)
    plt.xlabel('预测标签', fontsize=FONT_AXIS)
    plt.ylabel('真实标签', fontsize=FONT_AXIS)
    plt.title('归一化混淆矩阵', fontsize=FONT_MAIN)
    plt.xticks(fontsize=FONT_TICK, rotation=45)
    plt.yticks(fontsize=FONT_TICK)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_prob, class_names_zh, save_path):
    y_bin = label_binarize(y_true, classes=range(len(class_names_zh)))
    # 补齐缺失列（防止某些类别在验证集无样本）
    if y_bin.shape[1] < len(class_names_zh):
        full = np.zeros((len(y_true), len(class_names_zh)))
        full[:, :y_bin.shape[1]] = y_bin
        y_bin = full
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names_zh):
        if np.sum(y_bin[:, i]) == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假正率 (FPR)', fontsize=FONT_AXIS)
    plt.ylabel('真正率 (TPR)', fontsize=FONT_AXIS)
    plt.title('多分类 ROC 曲线', fontsize=FONT_MAIN)
    plt.legend(fontsize=FONT_TICK, loc='lower right')
    plt.xticks(fontsize=FONT_TICK)
    plt.yticks(fontsize=FONT_TICK)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_class_accuracy(y_true, y_pred, class_names_zh, save_csv_path):
    cm = confusion_matrix(y_true, y_pred)
    class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    df_cls = pd.DataFrame({
        '类别ID': range(len(class_acc)),
        '中文名称': class_names_zh,
        '类别准确率': class_acc
    })
    df_cls.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
    print("\n===== 各类别准确率 =====")
    print(df_cls)
    return df_cls

# ====================== 主实验（使用最优模型 + 中文绘图） ======================
def run_experiment():
    seed_everything(42)
    METADATA_PATH = "/root/CSI_system/TaskName/metadata/sample_metadata.csv"
    PLOT_DIR = "/root/CSI_system/plots/DANN_optimal"
    os.makedirs(PLOT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(METADATA_PATH)
    df = df[(df["user_id"] == "U02") & (df["environment"] == "E01")].reset_index(drop=True)
    devices = sorted(df['device'].unique())
    n_dev = len(devices)
    acc_matrix = np.zeros((n_dev, n_dev))

    # 预先计算每个设备的统计量
    device_stats = {}
    for dev in devices:
        dev_df = df[df["device"] == dev]
        if len(dev_df) > 0:
            temp_dataset = RobustCSIDataset(dev_df, is_training=False)
            mean, std = compute_global_stats(temp_dataset)
            device_stats[dev] = (mean, std)

    for i, source_dev in enumerate(devices):
        print(f"\n{'='*60}")
        print(f"🔬 源设备: {source_dev}")
        source_df = df[df["device"] == source_dev]
        if len(source_df) < 25:
            print(f"⚠️ 样本过少，跳过 {source_dev}")
            continue

        # 划分训练/验证集
        train_df, val_df = train_test_split(source_df, test_size=0.1,
                                            stratify=source_df['label'], random_state=42)
        src_mean, src_std = device_stats[source_dev]

        train_dataset = RobustCSIDataset(train_df, mean=src_mean, std=src_std, is_training=True)
        val_dataset = RobustCSIDataset(val_df, mean=src_mean, std=src_std, is_training=False)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

        for j, target_dev in enumerate(devices):
            if source_dev == target_dev:
                # 自身验证（无对抗）
                print(f"\n   📍 训练源设备自身模型 (无对抗)...")
                model, best_val_acc = train_dann_for_target(
                    train_loader, train_loader, val_loader, device,
                    epochs=15, lambda_domain=0.0
                )
                acc_matrix[i, j] = best_val_acc
                print(f"   ✅ {source_dev} → {source_dev} (自身) 验证准确率: {best_val_acc:.4f}")
                y_true, y_pred, y_prob = evaluate_model(model, val_loader, device)
                plot_confusion_matrix(y_true, y_pred, class_names_zh,
                                      os.path.join(PLOT_DIR, f"cm_{source_dev}_self.png"))
                plot_roc_curve(y_true, y_prob, class_names_zh,
                               os.path.join(PLOT_DIR, f"roc_{source_dev}_self.png"))
                save_class_accuracy(y_true, y_pred, class_names_zh,
                                    os.path.join(PLOT_DIR, f"class_acc_{source_dev}_self.csv"))
                continue

            # 跨设备 DANN 适应
            target_df = df[df["device"] == target_dev]
            if len(target_df) == 0:
                continue
            print(f"\n   🎯 目标设备: {target_dev} (样本数 {len(target_df)})")
            target_dataset = RobustCSIDataset(target_df, mean=src_mean, std=src_std, is_training=True)
            target_loader = DataLoader(target_dataset, batch_size=16, shuffle=True,
                                       num_workers=4, pin_memory=True, drop_last=True)

            model, _ = train_dann_for_target(
                train_loader, target_loader, val_loader, device,
                epochs=20, lambda_domain=0.1
            )

            test_dataset = RobustCSIDataset(target_df, mean=src_mean, std=src_std, is_training=False)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
            y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)
            acc = (y_true == y_pred).mean()
            acc_matrix[i, j] = acc
            print(f"   🚀 {source_dev} → {target_dev}: {acc:.4f}")

            plot_confusion_matrix(y_true, y_pred, class_names_zh,
                                  os.path.join(PLOT_DIR, f"cm_{source_dev}_to_{target_dev}.png"))
            plot_roc_curve(y_true, y_prob, class_names_zh,
                           os.path.join(PLOT_DIR, f"roc_{source_dev}_to_{target_dev}.png"))
            save_class_accuracy(y_true, y_pred, class_names_zh,
                                os.path.join(PLOT_DIR, f"class_acc_{source_dev}_to_{target_dev}.csv"))

        torch.cuda.empty_cache()

    # 保存整体稳定性矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(acc_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=devices, yticklabels=devices)
    plt.title("跨设备稳定性矩阵（最优 DANN 模型）", fontsize=FONT_MAIN)
    plt.xlabel("目标设备", fontsize=FONT_AXIS)
    plt.ylabel("源设备", fontsize=FONT_AXIS)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "stability_matrix.png"), dpi=300)
    plt.close()
    np.save(os.path.join(PLOT_DIR, "acc_matrix.npy"), acc_matrix)

    print("\n📊 所有结果已保存到", PLOT_DIR)
    print("\n最终准确率矩阵:")
    print(acc_matrix)

if __name__ == "__main__":
    run_experiment()