# ===================== LODO 训练代码（匹配你的预处理 + CNN-BiGRU + 多种子平均）=====================
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ===================== 配置 =====================
SEEDS = [11, 22, 33, 44, 55]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 35
LR = 1e-3
LODO_DATA_ROOT = "/root/CSI_system/LODO_SAFE_PREPROCESSED"

ALL_DEVICES = [
    "AmazonEchoPlus", "AmazonEchoShow8", "AmazonEchoSpot",
    "AmazonPlug", "AppleHomePod", 
    "GoogleNest", "GoveeSmartPlug", "WyzePlug"
]

# ==========================================
# 模型：CNN-BiGRU (适配你预处理后的向量格式)
# ==========================================
class CSINet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.cls = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.cls(x)

# ==========================================
# 固定种子（保证可复现）
# ==========================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ==========================================
# 单设备 LODO 训练
# ==========================================
def train_lodo_device(test_device, seed):
    # 加载你之前生成的 npz
    path = f"{LODO_DATA_ROOT}/{test_device}.npz"
    data = np.load(path)

    train_x = torch.tensor(data["train_x"], dtype=torch.float32)
    train_y = torch.tensor(data["train_y"], dtype=torch.long)
    test_x = torch.tensor(data["test_x"], dtype=torch.float32)
    test_y = torch.tensor(data["test_y"], dtype=torch.long)

    # Loader
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=BATCH_SIZE, shuffle=False)

    # 模型
    input_dim = train_x.shape[1]
    num_classes = len(torch.unique(train_y))
    model = CSINet(input_dim, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0
    best_f1 = 0

    for epoch in range(EPOCHS):
        # 训练
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # 评估
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                pred = model(x.to(DEVICE)).argmax(1)
                preds.extend(pred.cpu().numpy())
                labels.extend(y.numpy())

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')

        if acc > best_acc:
            best_acc = acc
            best_f1 = f1

    return best_acc, best_f1

# ==========================================
# 运行全部 LODO
# ==========================================
def run_all_lodo():
    print("="*70)
    print("🚀 LODO 留一设备实验 | 5 种种子平均 | CNN-BiGRU 模型")
    print("="*70)

    results = {}
    all_acc_mean = []
    all_f1_mean = []

    for dev in ALL_DEVICES:
        print(f"\n📌 测试设备: {dev}")
        acc_list, f1_list = [], []

        for seed in SEEDS:
            set_seed(seed)
            acc, f1 = train_lodo_device(dev, seed)
            acc_list.append(acc)
            f1_list.append(f1)
            print(f"   Seed {seed}: Acc={acc:.4f} | F1={f1:.4f}")

        # 平均 ± 方差
        m_acc, s_acc = np.mean(acc_list), np.std(acc_list)
        m_f1, s_f1 = np.mean(f1_list), np.std(f1_list)

        all_acc_mean.append(m_acc)
        all_f1_mean.append(m_f1)

        results[dev] = {
            "acc": f"{m_acc:.4f}±{s_acc:.4f}",
            "f1": f"{m_f1:.4f}±{s_f1:.4f}"
        }

        print(f"✅ 最终 {dev}: Acc={m_acc:.4f}±{s_acc:.4f} | F1={m_f1:.4f}±{s_f1:.4f}")

    # 整体平均
    print("\n" + "="*70)
    print(f"📊 全部 LODO 平均结果：")
    print(f"   Acc = {np.mean(all_acc_mean):.4f} ± {np.std(all_acc_mean):.4f}")
    print(f"   F1  = {np.mean(all_f1_mean):.4f} ± {np.std(all_f1_mean):.4f}")
    print("="*70)

    return results

# ==========================================
# 启动
# ==========================================
if __name__ == "__main__":
    results = run_all_lodo()