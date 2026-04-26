import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 模型架构: CNN-BiGRU-Attention (针对 CSI 特性)
# ==========================================
class CSINet(nn.Module):
    def __init__(self, num_classes=5, input_channels=14):
        super(CSINet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(125) # 统一时域特征维度
        )
        self.rnn = nn.GRU(128, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 14, 500)
        x = self.cnn(x)              # (Batch, 128, 125)
        x = x.permute(0, 2, 1)       # (Batch, 125, 128)
        out, _ = self.rnn(x)         # (Batch, 125, 256)
        out = out[:, -1, :]          # 取最后一个时间步
        return self.fc(out)

# ==========================================
# 2. Dataset 类 (支持多 NPY 索引缓存)
# ==========================================
class CSIDataset(Dataset):
    def __init__(self, split_json, processed_dir):
        with open(split_json, 'r') as f:
            self.sample_ids = json.load(f)['sample_ids']
        self.data = {}
        # 预加载所有处理后的特征以提升 LODO 实验速度
        for f in os.listdir(processed_dir):
            if f.endswith('.npy'):
                self.data.update(np.load(os.path.join(processed_dir, f), allow_pickle=True).item())

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        item = self.data[sid]
        return torch.tensor(item['feature']).float(), torch.tensor(item['label']).long()

# ==========================================
# 3. 实验引擎: 训练、评估、绘图
# ==========================================
class ExperimentEngine:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CSINet(input_channels=config['n_subcarriers']).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.results = {}

    def train(self, train_loader, val_loader):
        print("🚀 开始模型训练...")
        best_val_acc = 0
        for epoch in range(self.config['epochs']):
            self.model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            val_acc = self.evaluate(val_loader, silent=True)['accuracy']
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
            print(f"Epoch {epoch+1}/{self.config['epochs']} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

    def evaluate(self, loader, silent=False):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                outputs = self.model(x)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())
        
        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        if not silent:
            cm = confusion_matrix(all_labels, all_preds)
            return {'accuracy': acc, 'f1': f1, 'cm': cm, 'report': classification_report(all_labels, all_preds)}
        return {'accuracy': acc}

    def run_cross_domain_test(self, split_dir, processed_dir):
        self.model.load_state_dict(torch.load('best_model.pth'))
        test_types = ['test_id', 'test_cross_device', 'test_cross_user', 'test_cross_env']
        
        final_summary = []
        for t_type in test_types:
            print(f"🔍 正在执行跨域测试: {t_type}")
            loader = DataLoader(CSIDataset(f"{split_dir}/{t_type}.json", processed_dir), batch_size=32)
            res = self.evaluate(loader)
            self.results[t_type] = res
            final_summary.append([t_type, res['accuracy'], res['f1']])
            self.plot_confusion_matrix(res['cm'], t_type)
        
        # 打印最终对比表
        df_res = pd.DataFrame(final_summary, columns=['Test Type', 'Accuracy', 'Weighted F1'])
        print("\n📊 跨域实验数据总结：")
        print(df_res.to_string(index=False))
        df_res.to_csv("experiment_summary.csv", index=False)

    def plot_confusion_matrix(self, cm, title):
        plt.figure(figsize=(8, 6))
        labels = ['Walking', 'Breathing', 'Jumping', 'Waving', 'Running']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix: {title}")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f"cm_{title}.png")
        plt.close()

# ==========================================
# 4. 主入口
# ==========================================
if __name__ == "__main__":
    config = {
        'n_subcarriers': 14,
        'lr': 0.001,
        'epochs': 30,
        'processed_dir': "/root/CSI_system/my_benchmark/processed",
        'split_dir': "/root/CSI_system/my_benchmark/splits"
    }

    # 初始化引擎
    engine = ExperimentEngine(config)

    # 加载数据
    train_loader = DataLoader(CSIDataset(f"{config['split_dir']}/train_id.json", config['processed_dir']), batch_size=32, shuffle=True)
    val_loader = DataLoader(CSIDataset(f"{config['split_dir']}/val_id.json", config['processed_dir']), batch_size=32)

    # 运行实验
    engine.train(train_loader, val_loader)
    engine.run_cross_domain_test(config['split_dir'], config['processed_dir'])