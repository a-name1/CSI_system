# -*- coding: utf-8 -*-
# ===================== 论文级绘图环境（中文 + 服务器） =====================
import matplotlib
matplotlib.use("Agg")                         # 服务器无 GUI 后端
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']   # 中文黑体
plt.rcParams['axes.unicode_minus'] = False               # 正常显示负号

FONT_MAIN = 24      # 主标题字号
FONT_AXIS = 18      # 坐标轴标签字号
FONT_TICK = 14      # 刻度字号

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
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ====================== 动作标签映射（中文） ======================
LABEL_MAP = {"walking": 0, "seated-breathing": 1, "jumping": 2, "wavinghand": 3, "running": 4}
class_names_en = list(LABEL_MAP.keys())
class_names_zh = ["步行", "坐姿呼吸", "跳跃", "挥手", "跑步"]   # 与上述英文顺序一致
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
        amp = np.abs(csi).squeeze(0)          # [4,14,N]
        # 每帧每天线归一化 (AGC)
        per_antenna_per_frame_mean = np.mean(amp, axis=(1,2), keepdims=True) + 1e-8
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
            mean_ = self.mean[:, :, None]   # 变为 [4,14,1]
            std_ = self.std[:, :, None]     # 变为 [4,14,1]
            x = (x - mean_) / (std_ + 1e-8)
        return x, torch.tensor(row['label'], dtype=torch.long)

# ====================== 统计量计算 ======================
def compute_global_stats(dataset, batch_size=32):
    dataset.is_training = True
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    sum_, sum_sq, count = None, None, 0
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="计算统计量"):
            if sum_ is None:
                sum_ = torch.zeros(x.shape[1:3])
                sum_sq = torch.zeros(x.shape[1:3])
            sum_ += x.sum(dim=(0,3))
            sum_sq += (x**2).sum(dim=(0,3))
            count += x.size(0) * x.size(3)
    mean = sum_ / count
    std = torch.sqrt((sum_sq / count) - (mean**2) + 1e-8)
    dataset.is_training = False
    return mean, std

# ====================== 各种模型定义 ======================
# 1. MLP
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=4*14*360, num_classes=5):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.fc(self.flatten(x))

# 2. LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=4*14, hidden_size=128, num_layers=2, num_classes=5, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size*2 if bidirectional else hidden_size, num_classes)
    def forward(self, x):
        B, C, H, T = x.shape
        x = x.permute(0,3,1,2).reshape(B, T, C*H)   # [B,T,56]
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# 3. ResNet18 (处理 1×14×360 的伪图像)
class ResNet18CSI(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        import torchvision.models as models
        self.resnet = models.resnet18(weights=None)
        # 修改第一层卷积输入通道为1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        # x: [B,4,14,360] -> 对天线维求平均 -> [B,1,14,360]
        x = x.mean(dim=1, keepdim=True)
        return self.resnet(x)

# 4. PatchTST (修改：patch_len 从 16 改为 15，使 360 能被整除)
class PatchTSTClassifier(nn.Module):
    def __init__(self, input_dim=56, patch_len=15, d_model=128, nhead=8, num_layers=3, num_classes=5):
        super().__init__()
        # 确保 360 能被 patch_len 整除
        assert 360 % patch_len == 0, f"360 must be divisible by patch_len, but got {patch_len}"
        self.patch_len = patch_len
        self.patch_num = 360 // patch_len   # 24
        self.proj = nn.Linear(input_dim * patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_num, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
    def forward(self, x):
        B, C, H, T = x.shape
        x = x.permute(0,3,1,2).reshape(B, T, C*H)   # [B,360,56]
        x = x.unfold(1, self.patch_len, self.patch_len)   # [B,24,56,15]
        x = x.reshape(B, self.patch_num, -1)        # [B,24,56*15=840]
        x = self.proj(x) + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)

# 5. ViTClassifier (修改：patch_size 从 (14,16) 改为 (14,15)，使 360 能被 15 整除)
class ViTCSI(nn.Module):
    def __init__(self, img_height=14, img_width=360, patch_size=(14,15), dim=128, depth=6, heads=8, num_classes=5):
        super().__init__()
        from einops import rearrange
        self.patch_height, self.patch_width = patch_size
        self.patch_grid_h = img_height // patch_size[0]   # 1
        self.patch_grid_w = img_width // patch_size[1]    # 24
        num_patches = self.patch_grid_h * self.patch_grid_w
        patch_dim = patch_size[0] * patch_size[1]         # 210
        self.to_patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, batch_first=True), depth
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        # 将4个天线合并为1通道（平均值）
        x = x.mean(dim=1, keepdim=True)   # [B,1,14,360]
        from einops import rearrange
        x = rearrange(x, 'b c (h ph) (w pw) -> b (h w) (c ph pw)',
                      ph=self.patch_height, pw=self.patch_width)
        x = self.to_patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.head(x)

# 6. 最优模型：CNN + 时间全局平均池化（无GRU/无Attention，不含域对抗）
class OptimalCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.subcarrier_pool = nn.AdaptiveAvgPool2d((1, None))  # 子载波池化到1
        self.time_pool = nn.AdaptiveAvgPool1d(1)                # 时间维全局平均池化
        self.fc = nn.Linear(64, num_classes)                    # 分类头
    def forward(self, x):
        x = self.cnn(x)                     # [B,64,7,T/2]
        x = self.subcarrier_pool(x)         # [B,64,1,T/2]
        x = x.squeeze(2)                    # [B,64,T/2]
        x = self.time_pool(x).squeeze(-1)   # [B,64]
        return self.fc(x)

# ====================== 训练函数（单折） ======================
def train_one_fold(model, train_loader, val_loader, device, epochs=15, lr=0.0005):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    best_acc = 0.0
    best_state = None
    for ep in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}", leave=False)
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total if total > 0 else 0
        scheduler.step(acc)
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return best_acc

# ====================== 评估函数 ======================
def get_all_preds(model, loader, device):
    all_labels, all_preds, all_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="推理"):
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

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
    df_cls.to_csv(save_csv_path, index=False)
    print("\n===== 各类别准确率 =====")
    print(df_cls)
    return df_cls

# ====================== 主实验：多模型基准对比（原实验流程）======================
def run_benchmark():
    seed_everything(42)
    METADATA_PATH = "/root/CSI_system/TaskName/metadata/sample_metadata.csv"
    BASE_PLOT_DIR = "/root/CSI_system/plots/model_benchmark"
    os.makedirs(BASE_PLOT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(METADATA_PATH)
    df = df[(df["user_id"] == "U02") & (df["environment"] == "E01")].reset_index(drop=True)
    devices = sorted(df['device'].unique())
    n_dev = len(devices)
    print("设备列表:", devices)

    # 定义要评估的模型
    models_dict = {
        "MLP": MLPClassifier,
        "LSTM": LSTMClassifier,
        "ResNet18": ResNet18CSI,
        "PatchTST": PatchTSTClassifier,
        "ViT": ViTCSI,
        "OptimalCNN": OptimalCNN,
    }

    all_results = {}   # model_name -> {"self_acc": float, "cross_mean": float}

    for model_name, ModelClass in models_dict.items():
        print(f"\n{'='*80}\n>>> 开始训练模型：{model_name}\n{'='*80}")
        acc_matrix = np.zeros((n_dev, n_dev))   # 行: 训练设备, 列: 测试设备
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

        for i, train_dev in enumerate(devices):
            dev_df = df[df["device"] == train_dev].reset_index(drop=True)
            print(f"\n🔬 训练设备: {train_dev} | 样本数: {len(dev_df)}")
            if len(dev_df) < 25:
                print(f"⚠️  样本过少，跳过 {train_dev}")
                continue

            # ----- 交叉验证（自设备）-----
            fold_accs = []
            labels = dev_df['label'].values
            for fold, (tr_idx, val_idx) in enumerate(rskf.split(dev_df, labels)):
                tr_data = dev_df.iloc[tr_idx]
                val_data = dev_df.iloc[val_idx]

                m, s = compute_global_stats(RobustCSIDataset(tr_data, is_training=True))
                train_loader = DataLoader(RobustCSIDataset(tr_data, mean=m, std=s, is_training=True), 16, shuffle=True)
                val_loader = DataLoader(RobustCSIDataset(val_data, mean=m, std=s), 16)

                # 创建模型实例（注意不同模型的构造参数）
                if model_name == "MLP":
                    model = ModelClass(input_dim=4*14*360, num_classes=num_classes).to(device)
                elif model_name == "LSTM":
                    model = ModelClass(input_size=4*14, num_classes=num_classes).to(device)
                elif model_name == "ResNet18":
                    model = ModelClass(num_classes=num_classes).to(device)
                elif model_name == "PatchTST":
                    # 使用默认参数（patch_len=15）
                    model = ModelClass(input_dim=4*14, num_classes=num_classes).to(device)
                elif model_name == "ViT":
                    model = ModelClass(num_classes=num_classes).to(device)
                else:  # OptimalCNN
                    model = ModelClass(num_classes=num_classes).to(device)

                best_acc = train_one_fold(model, train_loader, val_loader, device, epochs=15)
                fold_accs.append(best_acc)

                del model
                torch.cuda.empty_cache()

            acc_matrix[i, i] = np.mean(fold_accs)
            print(f"✅ 自设备交叉验证平均ACC: {acc_matrix[i, i]:.4f}")

            # ----- 全量训练（用于跨设备测试）-----
            train_dev_df, val_dev_df = train_test_split(dev_df, test_size=0.1,
                                                        stratify=dev_df['label'], random_state=42)
            full_m, full_s = compute_global_stats(RobustCSIDataset(train_dev_df, is_training=True))
            full_train_loader = DataLoader(RobustCSIDataset(train_dev_df, mean=full_m, std=full_s, is_training=True),
                                           16, shuffle=True)
            full_val_loader = DataLoader(RobustCSIDataset(val_dev_df, mean=full_m, std=full_s), 16)

            # 创建最终模型
            if model_name == "MLP":
                final_model = ModelClass(input_dim=4*14*360, num_classes=num_classes).to(device)
            elif model_name == "LSTM":
                final_model = ModelClass(input_size=4*14, num_classes=num_classes).to(device)
            elif model_name == "ResNet18":
                final_model = ModelClass(num_classes=num_classes).to(device)
            elif model_name == "PatchTST":
                final_model = ModelClass(input_dim=4*14, num_classes=num_classes).to(device)
            elif model_name == "ViT":
                final_model = ModelClass(num_classes=num_classes).to(device)
            else:
                final_model = ModelClass(num_classes=num_classes).to(device)

            train_one_fold(final_model, full_train_loader, full_val_loader, device, epochs=20)

            # ----- 跨设备测试 -----
            for j, test_dev in enumerate(devices):
                if i == j:
                    continue
                te_df = df[df["device"] == test_dev]
                if len(te_df) == 0:
                    continue
                te_loader = DataLoader(RobustCSIDataset(te_df, mean=full_m, std=full_s), 16)
                yt, yp, _ = get_all_preds(final_model, te_loader, device)
                acc = (yt == yp).mean()
                acc_matrix[i, j] = acc
                print(f"   {train_dev} → {test_dev}: {acc:.4f}")

            del final_model
            torch.cuda.empty_cache()

        # 保存该模型的稳定性矩阵热图
        self_mean_acc = np.mean(np.diag(acc_matrix))
        cross_mean_acc = np.mean(acc_matrix[np.eye(n_dev, dtype=bool) == False])
        all_results[model_name] = {
            "self_acc": self_mean_acc,
            "cross_mean": cross_mean_acc
        }
        plt.figure(figsize=(12,10))
        sns.heatmap(acc_matrix, annot=True, fmt='.2f', cmap='YlGnBu',
                    xticklabels=devices, yticklabels=devices)
        plt.title(f'{model_name} 跨设备稳定性矩阵', fontsize=FONT_MAIN)
        plt.xlabel('测试设备', fontsize=FONT_AXIS)
        plt.ylabel('训练设备', fontsize=FONT_AXIS)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_PLOT_DIR, f"stability_matrix_{model_name}.png"), dpi=300)
        plt.close()
        np.save(os.path.join(BASE_PLOT_DIR, f"acc_matrix_{model_name}.npy"), acc_matrix)

    # ========== 汇总结果对比 ==========
    print("\n" + "="*80)
    print("模型对比汇总（自设备准确率 vs 跨设备平均准确率）")
    summary = []
    for name, res in all_results.items():
        summary.append({
            "模型": name,
            "自设备准确率(交叉验证均值)": res["self_acc"],
            "跨设备平均准确率": res["cross_mean"]
        })
        print(f"{name:25s} | 自设备: {res['self_acc']:.4f} | 跨设备: {res['cross_mean']:.4f}")

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(os.path.join(BASE_PLOT_DIR, "model_comparison_summary.csv"), index=False, encoding='utf-8-sig')

    # 绘制自设备准确率对比柱状图
    plt.figure(figsize=(10,6))
    plt.bar(df_summary["模型"], df_summary["自设备准确率(交叉验证均值)"], color='steelblue')
    plt.xticks(rotation=45, fontsize=FONT_TICK)
    plt.ylabel('准确率', fontsize=FONT_AXIS)
    plt.title('自设备交叉验证准确率 (5×2 CV)', fontsize=FONT_MAIN)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PLOT_DIR, "self_acc_comparison.png"), dpi=300)
    plt.close()

    # 绘制跨设备平均准确率对比柱状图
    plt.figure(figsize=(10,6))
    plt.bar(df_summary["模型"], df_summary["跨设备平均准确率"], color='coral')
    plt.xticks(rotation=45, fontsize=FONT_TICK)
    plt.ylabel('准确率', fontsize=FONT_AXIS)
    plt.title('跨设备平均准确率', fontsize=FONT_MAIN)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PLOT_DIR, "cross_acc_comparison.png"), dpi=300)
    plt.close()

    print(f"\n📊 所有结果已保存到 {BASE_PLOT_DIR}")

if __name__ == "__main__":
    run_benchmark()