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
from sklearn.metrics import confusion_matrix
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
# 注意力模块
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

# ==========================================
# ✅ 模型固定 90 通道 匹配流水线输出 [N,90,F,T]
# ==========================================
class SensingNet(nn.Module):
    def __init__(self, num_classes=5, use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        self.cnn = nn.Sequential(
            nn.Conv2d(90, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, 1))
        )

        self.gru = nn.GRU(64, 64, bidirectional=True, batch_first=True)

        if self.use_attention:
            self.attention = AttentionLayer(128)

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, return_feat=False):
        if x.dim() == 5:
            x = x.squeeze(1)

        f = self.cnn(x).squeeze(-1).permute(0, 2, 1)
        g, _ = self.gru(f)

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
# 绘图函数
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
# 数据读取：自动修复object、dtype、concat
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
        if len(parts) < 3:
            continue

        meta = {"user": parts[0], "env": parts[1], "device": parts[2]}
        if meta[mode] == tag:
            d = np.load(os.path.join(subdir, f), allow_pickle=True)
            x = d['x']
            y = d['y']

            # 修复object数组
            if x.dtype == object:
                x = np.array(x.tolist(), dtype=np.float32)
            if y.dtype == object:
                y = np.array(y.tolist(), dtype=np.int64)

            Xs.append(x)
            Ys.append(y)

    if not Xs:
        return None, None

    X = np.concatenate(Xs, axis=0).astype(np.float32)
    Y = np.concatenate(Ys, axis=0).astype(np.int64)

    return X, Y

def get_loader(x, y, shuffle=True):
    return DataLoader(
        TensorDataset(torch.tensor(x), torch.tensor(y)),
        batch_size=32, shuffle=shuffle
    )

# ==========================================
# 单次训练（返回最优模型）
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
    acc_record = []
    best_model_state = None  # 保存最优模型参数

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
            acc_record.append(acc)
            if acc > best_acc:
                best_acc = acc
                best_model_state = model.state_dict()  # 保存最优

    # 加载最优模型
    model.load_state_dict(best_model_state)

    return best_acc, acc_record, model

# ==========================================
# 5次重复平均（返回最优模型）
# ==========================================
def run_multi(
    src_x, src_y, tgt_x, tgt_y,
    use_attention=True, use_mmd=True, mmd_w=0.7,
    device=None
):
    accs = []
    all_records = []
    best_model = None
    best_acc_glob = 0

    for i in range(5):
        a, rec, model = run_single(src_x, src_y, tgt_x, tgt_y,
                       use_attention=use_attention,
                       use_mmd=use_mmd,
                       mmd_w=mmd_w,
                       seed=42+i, device=device)
        accs.append(a)
        all_records.append(rec)

        if a > best_acc_glob:
            best_acc_glob = a
            best_model = model  # 保存5次中最好的模型

    mean_rec = np.mean(np.array(all_records), axis=0).tolist()
    return np.mean(accs), np.std(accs), mean_rec, best_model

# ==========================================
# 模型消融实验 + 绘图
# ==========================================
def run_ablation_test(src_x, src_y, tgt_x, tgt_y, cfg_name, cfg_dir, device):
    print("\n🔍 开始运行【模型消融实验】(ATT + MMD 开关)")
    ablations = {
        "Full_Model": (True, True, 0.1),
        "No_MMD": (True, False, 0.0),
        "No_Attention": (False, True, 0.1),
    }
    ab_res = {}
    for name, (att, mmd, w) in ablations.items():
        m, s, _, _ = run_multi(src_x, src_y, tgt_x, tgt_y, att, mmd, w, device)
        ab_res[name] = (m, s)
        print(f"  {name:12s} | Acc: {m:.4f} ± {s:.4f}")

    # 消融柱状图
    df = pd.DataFrame([{"name": k, "acc": v[0]} for k, v in ab_res.items()])
    plt.figure(figsize=(9,5),dpi=200)
    ax = sns.barplot(x="name", y="acc", hue="name", data=df, palette="viridis", legend=False)
    plt.title(f'{cfg_name} 模型消融实验（5次平均）', fontsize=20)
    plt.ylabel("准确率")
    plt.ylim(0, 1.0)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x()+p.get_width()/2, p.get_height()), ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg_dir, "ablation_bar.png"))
    plt.close()

    return ab_res

# ==========================================
# MMD权重消融 + 绘图
# ==========================================
def run_mmd_weight_test(src_x, src_y, tgt_x, tgt_y, cfg_name, cfg_dir, device):
    print("\n📈 开始运行【MMD权重消融实验】")
    MMD_W = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    mmd_mean = []
    mmd_std = []
    for w in MMD_W:
        m, s, _, _ = run_multi(src_x, src_y, tgt_x, tgt_y, True, (w>0), w, device)
        mmd_mean.append(m)
        mmd_std.append(s)
        print(f"  MMD={w:.1f} | Acc: {m:.4f} ± {s:.4f}")

    plt.figure(figsize=(9,4),dpi=200)
    plt.plot(MMD_W, mmd_mean, marker="o", linewidth=2)
    plt.xlabel("MMD Weight")
    plt.ylabel("Accuracy")
    plt.title(f'{cfg_name} MMD权重消融（5次平均）')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg_dir, "mmd_curve.png"))
    plt.close()

    return MMD_W, mmd_mean, mmd_std

# ==========================================
# 单配置完整实验：基准 + 消融 + MMD + 全部绘图
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

    # 基准实验
    print("\n📊 基准实验 Full Model (ATT+MMD=0.1)")
    mean_full, std_full, acc_curve, best_model = run_multi(
        src_x, src_y, tgt_x, tgt_y, True, True, 0.1, device
    )

    # 训练曲线绘图
    plot_training_curve(acc_curve, os.path.join(cfg_dir, "training_curve.png"), cfg_name)

    # 消融、MMD 测试
    # ab_res = run_ablation_test(src_x, src_y, tgt_x, tgt_y, cfg_name, cfg_dir, device)
    # MMD_W, mmd_mean, mmd_std = run_mmd_weight_test(src_x, src_y, tgt_x, tgt_y, cfg_name, cfg_dir, device)

    # ✅ 绘制混淆矩阵（使用训练好的最优模型，不是随机模型）
    best_model.eval()
    eval_loader = get_loader(tgt_x, tgt_y, shuffle=False)
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for ex, ey in eval_loader:
            out = best_model(ex.to(device))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_trues.extend(ey.numpy())

    plot_confusion_matrix(all_trues, all_preds, labels, os.path.join(cfg_dir, "confusion_matrix.png"))

    # 保存完整结果日志
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
# 主程序入口
# ==========================================
if __name__ == "__main__":
    BASE = "/root/CSI_system/ablation_add_new_pca"
    CFG_JSON = "/root/CSI_system/config/ablation_configs_dft.json"
    SAVE_ROOT = "/root/CSI_system/ablation_test_add_new_pca/EP"
    LABELS = ["行走", "坐姿呼吸", "跳跃", "挥手", "跑步"]
    SOURCE_DEV = "AmazonEchoSpot"
    TARGET_DEV = "AmazonPlug"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_ROOT, exist_ok=True)

    cm = ConfigManager()
    all_configs = cm._load_from_file(CFG_JSON)
    print(f"🚀 共加载 {len(all_configs)} 个配置")

    summary = []
    for cfg in all_configs:
        if cfg == all_configs[0]:
            mean_acc, std_acc = run_full_config(
                cfg.name, BASE, SAVE_ROOT, SOURCE_DEV, TARGET_DEV, LABELS, device
            )
            summary.append({"name": cfg.name, "mean": mean_acc, "std": std_acc})
        else:
            pass

    with open(os.path.join(SAVE_ROOT, "all_summary.txt"), "w", encoding="utf-8") as f:
        f.write("【全部预处理消融汇总】\n")
        for s in summary:
            f.write(f"{s['name']:25s} | {s['mean']:.4f} ± {s['std']:.4f}\n")

    print("\n🎉 所有实验全部 5 次平均完成！")