import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, vit_b_16

# --- 基础注意力层 (参考你的 SensingNet) ---
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(weights * x, dim=1)

# --- 1. MLP ---
class MLP_Benchmark(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        flat_size = input_shape[0] * input_shape[1] * input_shape[2]
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU()
        )
        self.fc = nn.Linear(256, num_classes)
    def forward(self, x, return_feat=False):
        feat = self.backbone(x)
        logits = self.fc(feat)
        return (logits, feat) if return_feat else logits

# --- 2. LSTM (Bi-GRU + Attention 风格) ---
class LSTM_Benchmark(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.hidden_dim = 128
        self.lstm = nn.LSTM(input_shape[2] * input_shape[0], 64, num_layers=2, 
                            batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(128)
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x, return_feat=False):
        batch, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, t, -1)
        lstm_out, _ = self.lstm(x)
        feat = self.attention(lstm_out)
        logits = self.fc(feat)
        return (logits, feat) if return_feat else logits

# --- 3. ResNet18 ---
class ResNet18_Benchmark(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.model = resnet18(num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=7, stride=2, padding=3)
        self.backbone = nn.Sequential(*list(self.model.children())[:-1])
        self.fc = self.model.fc
    def forward(self, x, return_feat=False):
        feat = self.backbone(x).view(x.size(0), -1)
        logits = self.fc(feat)
        return (logits, feat) if return_feat else logits

# --- 4. Transformer ---
class Transformer_Benchmark(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        d_model = 128
        self.embedding = nn.Linear(input_shape[2] * input_shape[0], d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.fc = nn.Linear(d_model, num_classes)
    def forward(self, x, return_feat=False):
        batch, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, t, -1)
        x = self.embedding(x)
        feat = self.transformer(x).mean(dim=1)
        logits = self.fc(feat)
        return (logits, feat) if return_feat else logits
    
import os
import json
import copy
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

class MMDLoss(nn.Module):
    def forward(self, source, target):
        if source.size(0) < 2: return torch.tensor(0., device=source.device)
        return torch.norm(torch.mean(source, dim=0) - torch.mean(target, dim=0), 2)

class TransferBenchmarkRunner:
    def __init__(self, output_dir, device="cuda"):
        self.output_dir = output_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.mmd_fn = MMDLoss()
        os.makedirs(output_dir, exist_ok=True)

    def train_transfer(self, model, src_loader, tgt_loader, test_loader, model_name, epochs=50):
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0
        acc_history = []

        print(f"开始训练模型: {model_name}...")

        for epoch in range(1, epochs + 1):
            model.train()
            iter_tgt = iter(tgt_loader)
            
            for s_x, s_y in src_loader:
                try: t_x, _ = next(iter_tgt)
                except StopIteration: iter_tgt = iter(tgt_loader); t_x, _ = next(iter_tgt)

                s_x, s_y, t_x = s_x.to(self.device), s_y.to(self.device), t_x.to(self.device)

                # 前向传播 (MMD 核心逻辑)
                s_logits, s_feat = model(s_x, return_feat=True)
                _, t_feat = model(t_x, return_feat=True)

                loss_cls = criterion(s_logits, s_y)
                loss_mmd = self.mmd_fn(s_feat, t_feat)
                loss = loss_cls + 0.25 * loss_mmd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 验证 (在目标域测试设备上)
            if epoch % 5 == 0 or epoch == 1:
                acc = self.evaluate(model, test_loader)
                acc_history.append(acc)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), os.path.join(self.output_dir, f"best_{model_name}.pth"))
                print(f"Epoch {epoch:03d} | Target Acc: {acc:.4f} | Loss: {loss.item():.4f}")

        return best_acc, acc_history

    @torch.no_grad()
    def evaluate(self, model, loader):
        model.eval()
        preds, trues = [], []
        for x, y in loader:
            out = model(x.to(self.device))
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(y.numpy())
        return accuracy_score(trues, preds)
    
from utils.Spliter import StandardDatasetSplitter
from utils.config_manager import ConfigManager

def get_model_instance(name, input_shape, num_classes):
    mapping = {
        'mlp': MLP_Benchmark, 'lstm': LSTM_Benchmark,
        'resnet18': ResNet18_Benchmark, 'transformer': Transformer_Benchmark
    }
    return mapping[name](input_shape, num_classes)

if __name__ == "__main__":
    # 环境准备
    splitter = StandardDatasetSplitter(seed=42)
    CONFIG_PATH = "/root/CSI_system/utils/ablation_configs.json"
    CACHE_DIR = "/root/ablation_cache"
    
    ALL_DEVICES = ['AmazonEchoSpot', 'AmazonPlug']
    MODELS = ['mlp', 'lstm', 'resnet18', 'transformer']
    
    # 获取消融实验配置
    cfg_mgr = ConfigManager()
    ablation_configs = cfg_mgr._load_from_file(CONFIG_PATH)

    for train_dev in ALL_DEVICES:
        for test_dev in ALL_DEVICES:
            if train_dev == test_dev: continue # 跨设备验证

            for cfg in ablation_configs:
                res_dir = f"./results/{cfg.name}/{train_dev}_to_{test_dev}"
                runner = TransferBenchmarkRunner(output_dir=res_dir)

                # 加载数据
                src_data = splitter.load_and_split(CACHE_DIR, cfg.name, [train_dev], [train_dev], strategy="by_devices")
                tgt_data = splitter.load_and_split(CACHE_DIR, cfg.name, [test_dev], [test_dev], strategy="by_devices")

                def wrap_loader(data_tuple, shuffle=True):
                    x, y = data_tuple
                    # 全局标准化 (参考你的逻辑)
                    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
                    tx = torch.FloatTensor(x)
                    if tx.ndim == 3: tx = tx.unsqueeze(1).repeat(1, 3, 1, 1)
                    return DataLoader(TensorDataset(tx, torch.LongTensor(y)), batch_size=32, shuffle=shuffle)

                src_loader = wrap_loader(src_data["train"])
                tgt_loader = wrap_loader(tgt_data["train"])
                test_loader = wrap_loader(tgt_data["test"], shuffle=False)

                input_shape = (3, src_data["train"][0].shape[-2], src_data["train"][0].shape[-1])
                num_classes = int(np.max(src_data["train"][1])) + 1

                # 运行所有模型
                results_summary = {}
                for m_name in MODELS:
                    model = get_model_instance(m_name, input_shape, num_classes).to("cuda")
                    best_acc, history = runner.train_transfer(model, src_loader, tgt_loader, test_loader, m_name)
                    results_summary[m_name] = best_acc
                    
                    # 显存清理 (关键！)
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()

                # 保存当前配置的汇总结果
                with open(os.path.join(res_dir, "summary.json"), "w") as f:
                    json.dump(results_summary, f, indent=4)

    print("\n✅ 所有跨设备基准测试任务已完成！")