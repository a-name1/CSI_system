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

# 以下为原有导入，保持不变
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
class_names_en = list(LABEL_MAP.keys())
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

# ====================== 数据集（优雅广播） ======================
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
        amp = np.abs(csi).squeeze(0)          # [4,14,N]

        # 每帧每天线归一化（保留帧维）
        per_antenna_per_frame_mean = np.mean(amp, axis=(1, 2), keepdims=True) + 1e-8
        amp = amp / per_antenna_per_frame_mean

        curr_len = amp.shape[-1]
        if curr_len >= self.target_frames:
            if self.is_training:
                start = np.random.randint(0, curr_len - self.target_frames + 1)
            else:
                start = (curr_len - self.target_frames) // 2
            amp = amp[..., start:start + self.target_frames]
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
def compute_global_stats_from_df(df, batch_size=32, is_training=False):
    temp_dataset = RobustCSIDataset(df, is_training=is_training)
    return compute_global_stats(temp_dataset, batch_size)

def compute_global_stats(dataset, batch_size=32):
    dataset.is_training = True
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    sum_ = None
    sum_sq = None
    total_elements = 0
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Computing stats"):
            if sum_ is None:
                sum_ = torch.zeros(x.shape[1:3])      # [4,14]
                sum_sq = torch.zeros(x.shape[1:3])
            sum_ += x.sum(dim=(0,3))
            sum_sq += (x**2).sum(dim=(0,3))
            total_elements += x.size(0) * x.size(3)
    mean = sum_ / total_elements
    std = torch.sqrt((sum_sq / total_elements) - (mean**2) + 1e-8)
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

# ====================== 最优模型（CNN + 时间全局平均池化 + DANN） ======================
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
        # x: [B, 4, 14, T]
        x = self.cnn(x)                     # [B, 64, 7, T/2]
        x = self.subcarrier_pool(x)         # [B, 64, 1, T/2]
        x = x.squeeze(2)                    # [B, 64, T/2]
        feat = self.time_pool(x).squeeze(-1)  # [B, 64]

        class_out = self.fc(feat)

        if alpha is not None:
            rev_feat = GradientReversal.apply(feat, alpha)
            domain_out = self.domain_classifier(rev_feat)
            return class_out, domain_out, feat
        else:
            return (class_out, feat) if return_feat else class_out

# ====================== DANN 训练函数（与原接口完全一致） ======================
def train_dann_uda(source_loader, target_loader, val_loader, device,
                   epochs=20, lr=0.0005, lambda_domain=0.1):
    """
    无监督域适应训练
    source_loader: 有标签源域数据
    target_loader: 无标签目标域数据（分布与源域不同）
    val_loader:   从源域划分的验证集（用于早停）
    """
    model = OptimalDANN(num_classes=num_classes).to(device)   # 使用最优模型
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_acc = 0.0
    best_state = None

    len_source = len(source_loader)
    len_target = len(target_loader)
    max_len = max(len_source, len_target)
    total_iters = epochs * max_len
    iter_source = iter(source_loader)
    iter_target = iter(target_loader)
    current_iter = 0

    for ep in range(epochs):
        model.train()
        loop = tqdm(range(max_len), desc=f"Epoch {ep+1}/{epochs}", leave=False)

        for _ in loop:
            # 源域 batch
            try:
                x_s, y_s = next(iter_source)
            except StopIteration:
                iter_source = iter(source_loader)
                x_s, y_s = next(iter_source)
            x_s, y_s = x_s.to(device), y_s.to(device)

            # 目标域 batch
            try:
                x_t, _ = next(iter_target)
            except StopIteration:
                iter_target = iter(target_loader)
                x_t, _ = next(iter_target)
            x_t = x_t.to(device)

            # 动态 alpha
            p = current_iter / total_iters
            alpha = 2.0 / (1.0 + np.exp(-10*p)) - 1.0
            current_iter += 1

            domain_label_s = torch.zeros(x_s.size(0), dtype=torch.long, device=device)
            domain_label_t = torch.ones(x_t.size(0), dtype=torch.long, device=device)

            class_out_s, domain_out_s, _ = model(x_s, alpha=alpha)
            _, domain_out_t, _ = model(x_t, alpha=alpha)

            loss_class = criterion_class(class_out_s, y_s)
            loss_domain = (criterion_domain(domain_out_s, domain_label_s) +
                           criterion_domain(domain_out_t, domain_label_t)) / 2
            total_loss = loss_class + lambda_domain * loss_domain

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loop.set_postfix(cls=loss_class.item(), dom=loss_domain.item())

        # 验证
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
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
        for x, y in tqdm(loader, desc="Evaluating"):
            x = x.to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1)
            pred = torch.argmax(p, dim=1)
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
    return cm_norm

def plot_roc_curve(y_true, y_prob, class_names_zh, save_path):
    y_bin = label_binarize(y_true, classes=range(len(class_names_zh)))
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
        'ClassID': range(len(class_acc)),
        'ClassName_ZH': class_names_zh,
        'ClassAccuracy': class_acc
    })
    df_cls.to_csv(save_csv_path, index=False)
    print(df_cls)
    return df_cls

# ====================== 主实验（LODO + 最优 DANN） ======================
def run_experiment():
    seed_everything(42)
    METADATA_PATH = "/root/CSI_system/TaskName/metadata/sample_metadata.csv"
    PLOT_DIR = "/root/CSI_system/plots/LODO_DANN_optimal"
    os.makedirs(PLOT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(METADATA_PATH)
    df = df[(df["user_id"] == "U02") & (df["environment"] == "E01")].reset_index(drop=True)
    devices = sorted(df['device'].unique())
    n_dev = len(devices)

    results = {}

    for test_dev in devices:
        print(f"\n{'='*60}")
        print(f"🔬 留出测试设备: {test_dev}")

        source_dfs = [df[df["device"] == d] for d in devices if d != test_dev]
        if not source_dfs:
            print("⚠️  没有其他设备作为源域，跳过")
            continue
        source_df_all = pd.concat(source_dfs, ignore_index=True)

        train_df, val_df = train_test_split(source_df_all, test_size=0.1,
                                            stratify=source_df_all['label'], random_state=42)

        src_mean, src_std = compute_global_stats_from_df(train_df, is_training=True)

        train_dataset = RobustCSIDataset(train_df, mean=src_mean, std=src_std, is_training=True)
        val_dataset   = RobustCSIDataset(val_df,   mean=src_mean, std=src_std, is_training=False)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False,
                                  num_workers=4, pin_memory=True)

        target_df = df[df["device"] == test_dev]
        if len(target_df) == 0:
            print(f"⚠️  目标设备 {test_dev} 无数据，跳过")
            continue
        target_dataset = RobustCSIDataset(target_df, mean=src_mean, std=src_std, is_training=False)
        target_loader = DataLoader(target_dataset, batch_size=16, shuffle=True,
                                   num_workers=4, pin_memory=True, drop_last=True)

        print(f"   源域样本数: {len(train_df)}, 目标域样本数: {len(target_df)}")
        model, best_val_acc = train_dann_uda(
            source_loader=train_loader,
            target_loader=target_loader,
            val_loader=val_loader,
            device=device,
            epochs=20,
            lambda_domain=0.05
        )
        print(f"   源域验证集最佳准确率: {best_val_acc:.4f}")

        test_dataset = RobustCSIDataset(target_df, mean=src_mean, std=src_std, is_training=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                                 num_workers=4, pin_memory=True)
        y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)
        test_acc = (y_true == y_pred).mean()
        results[test_dev] = test_acc
        print(f"🎯 {test_dev} 测试准确率: {test_acc:.4f}")

        plot_confusion_matrix(y_true, y_pred, class_names_zh,
                              os.path.join(PLOT_DIR, f"cm_{test_dev}.png"))
        plot_roc_curve(y_true, y_prob, class_names_zh,
                       os.path.join(PLOT_DIR, f"roc_{test_dev}.png"))
        save_class_accuracy(y_true, y_pred, class_names_zh,
                            os.path.join(PLOT_DIR, f"class_acc_{test_dev}.csv"))

        torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("LODO 实验最终结果 (留一设备测试准确率):")
    for dev, acc in results.items():
        print(f"   {dev}: {acc:.4f}")
    res_df = pd.DataFrame({'TestDevice': list(results.keys()), 'Accuracy': list(results.values())})
    res_df.to_csv(os.path.join(PLOT_DIR, "lodo_results.csv"), index=False)

    plt.figure(figsize=(10, 6))
    plt.bar(res_df['TestDevice'], res_df['Accuracy'], color='steelblue')
    plt.xticks(rotation=45, fontsize=FONT_TICK)
    plt.yticks(fontsize=FONT_TICK)
    plt.ylabel('准确率', fontsize=FONT_AXIS)
    plt.xlabel('留出设备', fontsize=FONT_AXIS)
    plt.title('留一设备交叉验证准确率 (最优 DANN 模型: CNN+时间池化)', fontsize=FONT_MAIN)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "lodo_bar.png"), dpi=300)
    plt.close()
    print(f"\n📊 所有结果已保存到 {PLOT_DIR}")

if __name__ == "__main__":
    run_experiment()