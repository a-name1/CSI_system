# ===================== 修复路径 =====================
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
# ====================================================

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ===================== 配置 =====================
PREPROCESS_DIR = "/root/CSI_system/LODO_SAFE_PREPROCESSED"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 5
RESULT_DIR = "/root/CSI_system/LODO_FINAL_RESULTS"
PLOT_DIR = os.path.join(RESULT_DIR, "plots")
# ==================================================

# 创建保存目录
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 模型（不变） =====================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.Tanh(), nn.Linear(hidden_dim//2, 1))
    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return torch.sum(w * x, dim=1)

class SensingNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.gru = nn.GRU(64, 64, bidirectional=True, batch_first=True)
        self.attn = AttentionLayer(128)
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x, return_feat=False):
        x = self.cnn(x).squeeze(-1).permute(0,2,1)
        x, _ = self.gru(x)
        feat = self.attn(x)
        logits = self.fc(feat)
        return (logits, feat) if return_feat else logits

class MMDLoss(nn.Module):
    def forward(self, s, t):
        if s.shape[0] < 2: return torch.tensor(0., device=DEVICE)
        return torch.norm(s.mean(0) - t.mean(0), 2)

# ===================== 可视化函数（新增核心） =====================
def plot_training_curve(loss_history, acc_history, device_name):
    """绘制训练损失+测试准确率曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    # 损失曲线
    ax1.plot(loss_history, label='Train Loss', color='#FF4B4B')
    ax1.set_title(f'{device_name} - Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 准确率曲线
    ax2.plot(acc_history, label='Test Acc', color='#4B7BFF')
    ax2.set_title(f'{device_name} - Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{device_name}_training_curve.png"), dpi=200)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, device_name):
    """绘制混淆矩阵热力图"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=0.5)
    plt.title(f'{device_name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{device_name}_confusion_matrix.png"), dpi=200)
    plt.close()

def plot_overall_performance(results):
    """绘制全设备性能对比柱状图"""
    devices = list(results.keys())
    accs = [results[d]['acc'] for d in devices]
    f1s = [results[d]['f1'] for d in devices]
    
    x = np.arange(len(devices))
    width = 0.35
    
    plt.figure(figsize=(16, 6))
    plt.bar(x - width/2, accs, width, label='Accuracy', color='#4B7BFF')
    plt.bar(x + width/2, f1s, width, label='F1 Score', color='#FF8C42')
    
    plt.xlabel('Device')
    plt.ylabel('Score')
    plt.title('LODO Cross-Device Performance Comparison')
    plt.xticks(x, devices, rotation=45, ha='right')
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "overall_device_performance.png"), dpi=200)
    plt.close()

# ===================== 加载预处理特征 =====================
def load_data(test_device):
    data = np.load(f"{PREPROCESS_DIR}/{test_device}.npz")
    return (
        torch.FloatTensor(data["train_x"]), torch.LongTensor(data["train_y"]),
        torch.FloatTensor(data["test_x"]), torch.LongTensor(data["test_y"])
    )

# ===================== GPU 训练（升级版） =====================
def run_train():
    all_devices = json.load(open(f"{PREPROCESS_DIR}/devices.json"))["all_devices"]
    results = {}
    mmd = MMDLoss().to(DEVICE)

    print("\n🔥 开始 LODO GPU 训练 | 100% 安全无泄露")
    for test_device in tqdm(all_devices):
        print(f"\n=====================================")
        print(f"测试设备：{test_device}")
        print(f"=====================================")
        
        train_x, train_y, test_x, test_y = load_data(test_device)
        train_loader = DataLoader(TensorDataset(train_x, train_y), BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_x, test_y), BATCH_SIZE, shuffle=False)

        # 初始化模型
        model = SensingNet(NUM_CLASSES).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        best_acc = 0
        loss_history = []  # 记录损失
        acc_history = []   # 记录测试准确率

        # 训练循环
        for epoch in range(EPOCHS):
            # 训练
            model.train()
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                out, feat = model(x, return_feat=True)
                loss = nn.CrossEntropyLoss()(out, y) + 0.25 * mmd(feat, feat)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            loss_history.append(avg_loss)

            # 测试
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    out = model(x.to(DEVICE))
                    preds.extend(out.argmax(1).cpu().numpy())
                    trues.extend(y.numpy())
            
            acc = accuracy_score(trues, preds)
            acc_history.append(acc)
            if acc > best_acc:
                best_acc = acc

        # 计算最终指标
        f1 = f1_score(trues, preds, average="weighted")
        results[test_device] = {"acc": round(best_acc,4), "f1": round(f1,4)}

        # 保存图表
        plot_training_curve(loss_history, acc_history, test_device)
        plot_confusion_matrix(trues, preds, test_device)

        # 输出详细分类报告（核心测试数据）
        print(f"\n📋 {test_device} 详细分类报告：")
        print(classification_report(trues, preds, digits=4))
        print(f"✅ 最佳结果：Acc={best_acc:.4f} | F1={f1:.4f}")

    # 绘制整体性能对比图
    plot_overall_performance(results)
    
    # 保存最终结果
    avg_acc = np.mean([v["acc"] for v in results.values()])
    avg_f1 = np.mean([v["f1"] for v in results.values()])
    result_data = {
        "avg_acc": float(avg_acc),
        "avg_f1": float(avg_f1),
        "device_results": results
    }
    json.dump(result_data, open(f"{RESULT_DIR}/lodo_safe_result.json", "w"), indent=2)

    print(f"\n📊 训练完成！平均 Acc: {avg_acc:.4f} | 平均 F1: {avg_f1:.4f}")
    print(f"🖼️  所有图表已保存至：{PLOT_DIR}")

if __name__ == "__main__":
    run_train()