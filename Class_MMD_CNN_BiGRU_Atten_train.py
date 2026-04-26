import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
from utils.config_manager import ConfigManager 

# ==========================================
# 1. 模型架构与损失函数
# ==========================================


class AttentionLayer(nn.Module):
    """
    注意力层：用于对 BiGRU 输出的时间步进行加权求和，提取动作的核心特征。
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x):
        # x: [batch, seq_len, hidden_dim]
        weights = torch.softmax(self.attn(x), dim=1) # 计算时间维度上的权重
        return torch.sum(weights * x, dim=1)         # 加权求和，输出 [batch, hidden_dim]

class SensingNet(nn.Module):
    """
    CNN + BiGRU + Attention 网络结构：
    1. CNN 提取局部空间-频率特征。
    2. BiGRU 提取长时序运动建模。
    3. Attention 聚焦关键动作帧。
    """
    def __init__(self, num_classes=5):
        super().__init__()
        # 二维卷积部分：处理时频图输入
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, 1)) # 高度降维，保留时间轴
        )
        # 双向 GRU：128 维隐藏层 (64正向 + 64反向)
        self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(128)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, return_feat=False):
        # 处理可能的 batch 维度异常
        if x.dim() == 5: x = x.squeeze(1)
        
        x = self.cnn(x).squeeze(-1).permute(0, 2, 1) # [batch, seq_len, 64]
        gru_out, _ = self.gru(x)                     # [batch, seq_len, 128]
        feat = self.attention(gru_out)               # [batch, 128]
        logits = self.fc(feat)                       # [batch, num_classes]
        
        return (logits, feat) if return_feat else logits

class MMDLoss(nn.Module):
    """
    MMD (Maximum Mean Discrepancy) 损失：
    用于对齐源域（Source）和目标域（Target）的特征分布，减小设备间差异。
    """
    def __init__(self):
        super().__init__()
    def forward(self, source, target):
        if source.size(0) < 2: return torch.tensor(0., device=source.device)
        # 计算两个域特征均值的二范数距离
        return torch.norm(torch.mean(source, dim=0) - torch.mean(target, dim=0), 2)

# ==========================================
# 2. 可视化辅助函数
# ==========================================

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """绘制混淆矩阵并保存为图片"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix on Target Domain (Cross-Device)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_training_curve(acc_list, save_path, cfg_name):
    """绘制目标域准确率随训练周期的变化曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(acc_list, marker='o', label='Target Accuracy')
    plt.title(f'Ablation Study: {cfg_name}')
    plt.xlabel('Evaluation Step (every 5 epochs)')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# ==========================================
# 3. 数据处理逻辑
# ==========================================

def load_npz_from_subdir(base_dir, cfg_name, mode, target_tag):
    """
    从消融实验子目录加载对应的 NPZ 特征文件。
    mode: 'device' / 'user' / 'env'
    target_tag: 具体的标签值，如 'AmazonPlug'
    """
    subdir = os.path.join(base_dir, cfg_name)
    if not os.path.exists(subdir): return None, None
    
    X_list, y_list = [], []
    files = [f for f in os.listdir(subdir) if f.endswith("_feat.npz")]
    
    for f in files:
        # 解析文件名: U01_E01_AmazonPlug_feat.npz
        parts = f.replace("_feat.npz", "").split("_")
        if len(parts) < 3: continue
        meta = {"user": parts[0], "env": parts[1], "device": parts[2]}
        
        # 筛选符合当前实验设定的数据（如：只加载 AmazonPlug 的数据作为测试集）
        if meta[mode] == target_tag:
            with np.load(os.path.join(subdir, f)) as data:
                X_list.append(data['x'])
                y_list.append(data['y'])
                
    if not X_list: return None, None
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    # 特征全局标准化：对模型收敛至关重要
    X = (X - np.mean(X)) / (np.std(X) + 1e-8)
    return X, y

# ==========================================
# 4. 主运行逻辑
# ==========================================

if __name__ == "__main__":
    # 路径配置
    BASE_CACHE_DIR = "/root/CSI_system/ablation_processed_features_cross_dev"
    CONFIG_JSON = "/root/CSI_system/utils/ablation_configs_dft.json"
    # CONFIG_JSON = "/root/CSI-Bench-Real-WiFi-Sensing-Benchmark/CSI_system/utils/dft.json"
    RESULT_ROOT = "/root/CSI_system/ablation_study_dev/PE" # 结果保存根目录
    
    # 实验标签设置
    LABEL_MAP = {
        "walking": 0,
        "seated-breathing": 1,
        "jumping": 2,
        "wavinghand": 3,
        "running": 4
    }
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODE = "device"
    TRAIN_TAG = "AmazonPlug" # 源域设备
    TEST_TAG = "AmazonEchoSpot"      # 目标域设备（测试鲁棒性）

    cm_mgr = ConfigManager()
    configs = cm_mgr._load_from_file(CONFIG_JSON)
    mmd_loss_fn = MMDLoss()

    for cfg in configs:
        # 为每个消融配置创建独立的结果文件夹
        cfg_res_dir = os.path.join(RESULT_ROOT, cfg.name)
        os.makedirs(cfg_res_dir, exist_ok=True)
        
        print(f"\n" + "█"*60)
        print(f"🚀 Running Ablation: {cfg.name}")
        
        # 加载源域和目标域数据
        src_x, src_y = load_npz_from_subdir(BASE_CACHE_DIR, cfg.name, MODE, TRAIN_TAG)
        tgt_x, tgt_y = load_npz_from_subdir(BASE_CACHE_DIR, cfg.name, MODE, TEST_TAG)

        if src_x is None or tgt_x is None:
            print(f"⚠️ Skip: No data for {cfg.name}")
            continue

        # 数据 Loader 封装
        def get_loader(x, y, shuffle=True):
            tx = torch.FloatTensor(x)
            # 如果特征是单通道 [N, H, W]，则扩展为 3 通道 [N, 3, H, W] 适配 CNN
            if tx.ndim == 3: tx = tx.unsqueeze(1).repeat(1, 3, 1, 1)
            return DataLoader(TensorDataset(tx, torch.LongTensor(y)), batch_size=32, shuffle=shuffle)

        src_loader = get_loader(src_x, src_y)
        tgt_loader = get_loader(tgt_x, tgt_y)
        eval_loader = get_loader(tgt_x, tgt_y, shuffle=False) # 评估专用，不打乱顺序

        model = SensingNet(num_classes=len(LABEL_MAP)).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        acc_history = []
        best_acc = 0.0
        final_y_true, final_y_pred = [], []

        # ---------------- 训练循环 ----------------
        for epoch in range(1, 50):
            model.train()
            iter_tgt = iter(tgt_loader) # 目标域无标签迭代器
            
            for s_x, s_y in src_loader:
                # 获取一个 batch 的目标域特征（用于 MMD 域对齐）
                try: t_x, _ = next(iter_tgt)
                except StopIteration: iter_tgt = iter(tgt_loader); t_x, _ = next(iter_tgt)

                s_x, s_y, t_x = s_x.to(DEVICE), s_y.to(DEVICE), t_x.to(DEVICE)
                optimizer.zero_grad()
                
                # 前向传播：计算分类 Loss 和 MMD 迁移 Loss
                s_out, s_feat = model(s_x, return_feat=True)
                _, t_feat = model(t_x, return_feat=True)

                # 总损失 = 交叉熵(源域标签) + 0.1 * 域特征间距(MMD)
                loss = nn.CrossEntropyLoss()(s_out, s_y) + 0.25 * mmd_loss_fn(s_feat, t_feat)
                loss.backward()
                optimizer.step()

            # 每 5 轮在目标域（测试设备）上进行一次完整评估
            if epoch % 5 == 0:
                model.eval()
                temp_preds, temp_trues = [], []
                with torch.no_grad():
                    for tx, ty in eval_loader:
                        out = model(tx.to(DEVICE))
                        temp_preds.extend(out.argmax(1).cpu().numpy())
                        temp_trues.extend(ty.numpy())
                
                curr_acc = np.mean(np.array(temp_preds) == np.array(temp_trues))
                acc_history.append(curr_acc)
                print(f"Epoch {epoch:03d} | Target Acc: {curr_acc:.4f}")

                # 保存该消融配置下的最佳模型权重
                if curr_acc > best_acc:
                    best_acc = curr_acc
                    torch.save(model.state_dict(), os.path.join(cfg_res_dir, "best_model.pth"))
                    # 记录最佳性能时的预测结果用于绘制混淆矩阵
                    final_y_true, final_y_pred = temp_trues, temp_preds

        # ---------------- 结果持久化 ----------------
        # 1. 绘制准确率曲线
        plot_training_curve(acc_history, os.path.join(cfg_res_dir, "accuracy_curve.png"), cfg.name)
        # 2. 绘制混淆矩阵
        plot_confusion_matrix(final_y_true, final_y_pred, LABEL_MAP, os.path.join(cfg_res_dir, "confusion_matrix.png"))
        # 3. 保存分类报告（Precision, Recall, F1）
        with open(os.path.join(cfg_res_dir, "eval_report.txt"), "w") as f:
            f.write(f"Config: {cfg.name}\nBest Accuracy: {best_acc:.4f}\n\n")
            f.write(classification_report(final_y_true, final_y_pred, target_names=LABEL_MAP))

        print(f"✅ Finished: {cfg.name} | Best Acc: {best_acc:.4f} | Results in {cfg_res_dir}")
        
        # 每次消融实验结束，必须强制清理显存和内存，防止爆显存
        del src_loader, tgt_loader, eval_loader, model
        gc.collect()
        torch.cuda.empty_cache()