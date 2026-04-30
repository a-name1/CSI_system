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
import pandas as pd

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
# 统一模型（支持开关注意力）
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

# =====================================================================
# ✅ 加深加宽双分支模型（适配 12 通道实数输入：前6实部+后6虚部）
# =====================================================================
class SensingNet(nn.Module):
    def __init__(self, num_classes=5, use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        # --------------------------
        # 分支 1：前 6 通道（实部）
        # --------------------------
        self.cnn_real = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, 1))
        )

        # --------------------------
        # 分支 2：后 6 通道（虚部）
        # --------------------------
        self.cnn_imag = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, 1))
        )

        # 融合后进入 GRU
        self.gru = nn.GRU(256, 128, bidirectional=True, batch_first=True)

        # 注意力
        if self.use_attention:
            self.attention = AttentionLayer(256)

        # 分类头
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, return_feat=False):
        # 兼容可能的5维输入
        if x.dim() == 5:
            x = x.squeeze(1)

        # 12通道输入：前6实部 + 后6虚部
        x_real = x[:, :6, :, :]
        x_imag = x[:, 6:, :, :]

        # 双分支特征提取
        f_real = self.cnn_real(x_real).squeeze(-1).permute(0, 2, 1)
        f_imag = self.cnn_imag(x_imag).squeeze(-1).permute(0, 2, 1)

        # 特征融合
        f_cat = torch.cat([f_real, f_imag], dim=-1)

        # GRU
        g, _ = self.gru(f_cat)

        # 注意力 / 取最后一步
        if self.use_attention:
            feat = self.attention(g)
        else:
            feat = g[:, -1, :]

        out = self.fc(feat)
        return (out, feat) if return_feat else out

# ==========================================
# MMD 损失
# ==========================================
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
# 数据加载（无额外预处理）
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
        if len(parts) < 3: continue
        meta = {"user": parts[0], "env": parts[1], "device": parts[2]}
        if meta[mode] == tag:
            d = np.load(os.path.join(subdir, f))
            Xs.append(d['x'])
            Ys.append(d['y'])
    if not Xs:
        return None, None

    X = np.vstack(Xs)
    Y = np.concatenate(Ys)

    if np.iscomplexobj(X):
        X = np.concatenate([np.real(X), np.imag(X)], axis=1)
    return X, Y

def get_loader(x, y, shuffle=True):
    return DataLoader(
        TensorDataset(torch.FloatTensor(x), torch.LongTensor(y)),
        batch_size=32, shuffle=shuffle
    )

# ==========================================
# 单次训练（用于平均）
# ==========================================
def run_single(
    src_x, src_y, tgt_x, tgt_y,
    use_attention=True, use_mmd=True, mmd_w=0.7,
    seed=42, device=None
):
    set_seed(seed)
    src_loader = get_loader(src_x, src_y)
    tgt_loader = get_loader(tgt_x, tgt_y)
    eval_loader = get_loader(tgt_x, tgt_y, shuffle=False)

    model = SensingNet(use_attention=use_attention).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    mmd = MMDLoss()

    best_acc = 0.0
    for epoch in range(50):
        model.train()
        tgt_iter = iter(tgt_loader)
        for sx, sy in src_loader:
            try:
                tx, _ = next(tgt_iter)
            except:
                tgt_iter = iter(tgt_loader)
                tx, _ = next(tgt_iter)

            sx, sy, tx = sx.to(device), sy.to(device), tx.to(device)
            opt.zero_grad()
            s_out, s_feat = model(sx, return_feat=True)
            loss = ce(s_out, sy)
            if use_mmd:
                _, t_feat = model(tx, return_feat=True)
                loss += mmd_w * mmd(s_feat, t_feat)
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
            if acc > best_acc:
                best_acc = acc

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return best_acc

# ==========================================
# 5次平均运行
# ==========================================
def run_multi(
    src_x, src_y, tgt_x, tgt_y,
    use_attention=True, use_mmd=True, mmd_w=0.7,
    device=None
):
    accs = []
    for i in range(5):
        a = run_single(src_x, src_y, tgt_x, tgt_y,
                       use_attention=use_attention,
                       use_mmd=use_mmd,
                       mmd_w=mmd_w,
                       seed=42+i, device=device)
        accs.append(a)
    return np.mean(accs), np.std(accs)

# ==========================================
# ✅ 独立函数 1：模型消融实验（注意力/MMD开关）
# ==========================================
def run_ablation_test(src_x, src_y, tgt_x, tgt_y, cfg_name, cfg_dir, device):
    print("\n🔍 开始运行【模型消融实验】(ATT + MMD 开关)")
    ablations = {
        "Full_Model": (True, True, 0.7),
        "No_MMD": (True, False, 0.0),
        "No_Attention": (False, True, 0.7),
    }
    ab_res = {}
    for name, (att, mmd, w) in ablations.items():
        m, s = run_multi(src_x, src_y, tgt_x, tgt_y, att, mmd, w, device)
        ab_res[name] = (m, s)
        print(f"  {name:12s} | Acc: {m:.4f} ± {s:.4f}")

    # 绘图：消融柱状图
    df = pd.DataFrame([{"name": k, "acc": v[0]} for k, v in ab_res.items()])
    plt.figure(figsize=(9,5),dpi=200)
    ax = sns.barplot(x="name", y="acc", hue="name", data=df, palette="viridis", legend=False)
    plt.title(f"{cfg_name} 模型消融实验（5次平均）", fontsize=20)
    plt.ylabel("准确率")
    plt.ylim(0, 1.0)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x()+p.get_width()/2, p.get_height()), ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg_dir, "ablation_bar.png"))
    plt.close()

    return ab_res

# ==========================================
# ✅ 独立函数 2：MMD权重消融实验
# ==========================================
def run_mmd_weight_test(src_x, src_y, tgt_x, tgt_y, cfg_name, cfg_dir, device):
    print("\n📈 开始运行【MMD权重消融实验】")
    MMD_W = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    mmd_mean = []
    mmd_std = []
    for w in MMD_W:
        m, s = run_multi(src_x, src_y, tgt_x, tgt_y, True, (w>0), w, device)
        mmd_mean.append(m)
        mmd_std.append(s)
        print(f"  MMD={w:.1f} | Acc: {m:.4f} ± {s:.4f}")

    # 绘图：MMD曲线
    plt.figure(figsize=(9,4),dpi=200)
    plt.plot(MMD_W, mmd_mean, marker="o", linewidth=2)
    plt.xlabel("MMD Weight")
    plt.ylabel("Accuracy")
    plt.title(f"{cfg_name} MMD权重消融（5次平均）")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg_dir, "mmd_curve.png"))
    plt.close()

    return MMD_W, mmd_mean, mmd_std

# ==========================================
# 运行全套实验（调用两个独立测试）
# ==========================================
def run_full_config(cfg_name, base_dir, save_root, source_dev, target_dev, labels, device):
    print(f"\n{'='*60}")
    print(f"📌 正在运行：{cfg_name}")
    print(f"{'='*60}")

    src_x, src_y = load_npz_from_subdir(base_dir, cfg_name, "device", source_dev)
    tgt_x, tgt_y = load_npz_from_subdir(base_dir, cfg_name, "device", target_dev)
    if src_x is None or tgt_x is None:
        print(f"❌ 数据不存在")
        return

    cfg_dir = os.path.join(save_root, cfg_name)
    os.makedirs(cfg_dir, exist_ok=True)

    # ===================== 基准实验 =====================
    print("\n📊 基准实验 Full Model (ATT+MMD=0.7)")
    mean_full, std_full = run_multi(src_x, src_y, tgt_x, tgt_y, True, True, 0.7, device)

    # ===================== 分别调用独立测试 =====================
    # ab_res = run_ablation_test(src_x, src_y, tgt_x, tgt_y, cfg_name, cfg_dir, device)
    # MMD_W, mmd_mean, mmd_std = run_mmd_weight_test(src_x, src_y, tgt_x, tgt_y, cfg_name, cfg_dir, device)

    # ===================== 保存报告 =====================
    with open(os.path.join(cfg_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"【实验配置】{cfg_name}\n")
        f.write(f"基准模型：{mean_full:.4f} ± {std_full:.4f}\n\n")
        # f.write("【模型消融实验】\n")
        # for k, (m, s) in ab_res.items():
        #     f.write(f"{k:12s} : {m:.4f} ± {s:.4f}\n")
        # f.write("\n【MMD权重消融实验】\n")
        # for w, m, s in zip(MMD_W, mmd_mean, mmd_std):
        #     f.write(f"MMD={w:.1f} : {m:.4f} ± {s:.4f}\n")

    print(f"\n✅ {cfg_name} 完成！")
    return mean_full, std_full

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    BASE = "/root/CSI_system/ablation_add_No_PCA"
    CFG_JSON = "/root/CSI_system/config/abletion_add_No_PCA.json"
    SAVE_ROOT = "/root/CSI_system/ablation_test_add_No_PCA/PE"
    LABELS = ["行走", "坐姿呼吸", "跳跃", "挥手", "跑步"]
    SOURCE_DEV = "AmazonPlug"
    TARGET_DEV = "AmazonEchoSpot"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_ROOT, exist_ok=True)

    cm = ConfigManager()
    all_configs = cm._load_from_file(CFG_JSON)
    print(f"🚀 共加载 {len(all_configs)} 个配置")

    summary = []
    for cfg in all_configs:
        mean_acc, std_acc = run_full_config(
            cfg.name, BASE, SAVE_ROOT, SOURCE_DEV, TARGET_DEV, LABELS, device
        )
        summary.append({"name": cfg.name, "mean": mean_acc, "std": std_acc})

    with open(os.path.join(SAVE_ROOT, "all_summary.txt"), "w", encoding="utf-8") as f:
        f.write("【全部预处理消融汇总】\n")
        for s in summary:
            f.write(f"{s['name']:25s} | {s['mean']:.4f} ± {s['std']:.4f}\n")

    print("\n🎉 所有实验全部 5 次平均完成！")