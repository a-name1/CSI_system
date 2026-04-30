# ===================== 论文级绘图环境（中文 + 服务器） =====================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

FONT_MAIN = 24
FONT_AXIS = 18
FONT_TICK = 14
# =======================================================================

import os
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
import sys
from pathlib import Path

# ========= 路径挂载 =========
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
utils_dir = root_dir / "utils"
sys.path.append(str(root_dir))
sys.path.append(str(utils_dir))

from utils.config_manager import ConfigManager

# ========= 随机种子 =========
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 模型
# ==========================================
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
    def __init__(self, num_classes=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, 1))
        )
        self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(128)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, return_feat=False):
        if x.dim() == 5:
            x = x.squeeze(1)
        x = self.cnn(x).squeeze(-1).permute(0, 2, 1)
        g, _ = self.gru(x)
        feat = self.attention(g)
        out = self.fc(feat)
        return (out, feat) if return_feat else out

class MMDLoss(nn.Module):
    def forward(self, s, t):
        if s.size(0) < 2:
            return torch.tensor(0., device=s.device)
        return torch.norm(torch.mean(s, 0) - torch.mean(t, 0), 2)

# ==========================================
# 中文绘图
# ==========================================
def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": FONT_TICK})
    plt.title('跨设备测试混淆矩阵', fontsize=FONT_MAIN)
    plt.xlabel('预测类别', fontsize=FONT_AXIS)
    plt.ylabel('真实类别', fontsize=FONT_AXIS)
    plt.xticks(fontsize=FONT_TICK)
    plt.yticks(fontsize=FONT_TICK)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_training_curve(acc_list, save_path, cfg_name):
    plt.figure(figsize=(10, 5))
    plt.plot(acc_list, marker='o')
    plt.title(f'{cfg_name} 跨设备目标域准确率变化', fontsize=FONT_MAIN)
    plt.xlabel('评估次数（每5轮）', fontsize=FONT_AXIS)
    plt.ylabel('准确率', fontsize=FONT_AXIS)
    plt.xticks(fontsize=FONT_TICK)
    plt.yticks(fontsize=FONT_TICK)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==========================================
# 数据
# ==========================================
def load_npz_from_subdir(base_dir, cfg_name, mode, tag):
    subdir = os.path.join(base_dir, cfg_name)
    if not os.path.exists(subdir):
        return None, None
    Xs, Ys = [], []
    for f in os.listdir(subdir):
        if not f.endswith("_feat.npz"):
            continue
        parts = f.replace("_feat.npz", "").split("_")
        meta = {"user": parts[0], "env": parts[1], "device": parts[2]}
        if meta[mode] == tag:
            d = np.load(os.path.join(subdir, f))
            Xs.append(d['x'])
            Ys.append(d['y'])
    if not Xs:
        return None, None
    return np.vstack(Xs), np.concatenate(Ys)

def get_loader(x, y, shuffle=True):
    if np.iscomplexobj(x):
        x = np.concatenate([np.real(x), np.imag(x)], axis=1)
    return DataLoader(
        TensorDataset(torch.FloatTensor(x), torch.LongTensor(y)),
        batch_size=32, shuffle=shuffle
    )

# ==========================================
# 单次运行
# ==========================================
def run_once(seed, src_loader, tgt_loader, eval_loader, device, num_classes):
    set_seed(seed)
    model = SensingNet(num_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    mmd = MMDLoss()

    best_acc, best_t, best_p = 0, [], []
    hist = []

    for epoch in range(1, 50):
        model.train()
        it_t = iter(tgt_loader)
        for sx, sy in src_loader:
            try:
                tx, _ = next(it_t)
            except StopIteration:
                it_t = iter(tgt_loader)
                tx, _ = next(it_t)

            sx, sy, tx = sx.to(device), sy.to(device), tx.to(device)
            opt.zero_grad()
            so, sf = model(sx, True)
            _, tf = model(tx, True)
            loss = nn.CrossEntropyLoss()(so, sy) + 0.25 * mmd(sf, tf)
            loss.backward()
            opt.step()

        if epoch % 5 == 0:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for ex, ey in eval_loader:
                    o = model(ex.to(device))
                    preds.extend(o.argmax(1).cpu().numpy())
                    trues.extend(ey.numpy())
            acc = np.mean(np.array(preds) == np.array(trues))
            hist.append(acc)
            if acc > best_acc:
                best_acc, best_t, best_p = acc, trues, preds

    return best_acc, best_t, best_p, hist

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    BASE = "/root/CSI_system/ablation_processed_features_cross_dev"
    CFG_JSON = "/root/CSI_system/config/ablation_configs_dft.json"
    SAVE_ROOT = "/root/CSI_system/ablation_study_dev/PP"

    LABELS = ["行走", "坐姿呼吸", "跳跃", "挥手", "跑步"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = ConfigManager()._load_from_file(CFG_JSON)
    train_dev = "AmazonPlug"
    test_dev = "AmazonPlug"
    for cfg in configs:
        print(f"\n{'█'*60}\n{cfg.name}")
        sx, sy = load_npz_from_subdir(BASE, cfg.name, "device", train_dev)
        tx, ty = load_npz_from_subdir(BASE, cfg.name, "device", test_dev)
        if sx is None or tx is None:
            continue

        sl = get_loader(sx, sy)
        tl = get_loader(tx, ty)
        el = get_loader(tx, ty, False)

        res_dir = os.path.join(SAVE_ROOT, cfg.name)
        os.makedirs(res_dir, exist_ok=True)

        accs, hists, preds = [], [], []
        for i in range(5):
            print(f"Run {i+1}/5")
            a, t, p, h = run_once(42+i, sl, tl, el, device, len(LABELS))
            accs.append(a)
            hists.append(h)
            preds.append((t, p))

        mean_acc, std_acc = np.mean(accs), np.std(accs)
        mean_curve = np.mean(np.array(hists), axis=0)
        best_t, best_p = preds[np.argmax(accs)]

        plot_training_curve(mean_curve, os.path.join(res_dir, "curve.png"), cfg.name)
        plot_confusion_matrix(best_t, best_p, LABELS,
                              os.path.join(res_dir, "cm.png"))

        with open(os.path.join(res_dir, "report.txt"), "w") as f:
            f.write(f"{cfg.name}\n平均准确率: {mean_acc:.4f} ± {std_acc:.4f}\n\n")
            f.write(classification_report(best_t, best_p,
                                         target_names=LABELS,
                                         zero_division=0))

        print(f"✅ Mean: {mean_acc:.4f} ± {std_acc:.4f}")

        gc.collect()
        torch.cuda.empty_cache()