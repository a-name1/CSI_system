import os
import json
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

# 确保引入你的数据划分器
from utils.Spliter import StandardDatasetSplitter # 根据你的实际路径修改
from utils.config_manager import ConfigManager
# ==========================================
# 1. 论文模型实现: CNN-BiGRU-Attention
# ==========================================
#------------------------- Optimized CNN + BiGRU + Attention Model (3-Channel) ------------------

class Attention(nn.Module):
    """
    针对 BiGRU 输出的时序注意力机制
    作用：自动学习动作发生的时间段（如跌倒的瞬间），赋予更高权重
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # BiGRU 双向输出，所以输入维度是 hidden_dim * 2
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x shape: [batch, seq_len, hidden_dim * 2]
        attn_weights = torch.softmax(self.attn(x), dim=1) # [batch, seq_len, 1]
        context = torch.sum(attn_weights * x, dim=1)     # [batch, hidden_dim * 2]
        return context


class CNNBiGRUClassifier(nn.Module):
    """
    针对多维 STFT 特征设计的分类模型
    输入形状: (Batch, 3, T_aligned, F_aligned) 
    - 3: PCA 后的前三个主成分 (PC1, PC2, PC3)
    """
    def __init__(self, win_len=64, feature_size=32, num_classes=6,
                 cnn_channels=[32, 64], gru_hidden=64, dropout=0.5):
        super(CNNBiGRUClassifier, self).__init__()
        self.win_len = win_len           # 对应对齐后的时间长度 (T)
        self.feature_size = feature_size # 对应对齐后的频率长度 (F)
        self.num_classes = num_classes
        self.cnn_channels = cnn_channels
        self.gru_hidden = gru_hidden
        self.dropout = dropout

        # 1. CNN: 提取时频空间特征
        # 输入形状: (B, 3, T, F) -> 将三个主成分视为 RGB 通道
        self.cnn = nn.Sequential(
            # 第一层：3通道输入
            nn.Conv2d(3, cnn_channels[0], kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),          # 降低分辨率，提取高层特征
            
            # 第二层
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(),
            # 自适应池化：将频率维度(F)压缩为1，保留时间轴(T)进行时序建模
            nn.AdaptiveAvgPool2d((None, 1))           # 输出形状: (B, C1, T', 1)
        )

        # 2. BiGRU: 捕捉动作在时间轴上的演变规律
        # 输入维度: cnn_channels[1] (即 CNN 提取的特征向量长度)
        self.gru = nn.GRU(cnn_channels[1], gru_hidden, batch_first=True, bidirectional=True)

        # 3. Attention: 聚焦动作关键帧
        self.attention = Attention(gru_hidden)

        # 4. Classifier: 最终分类
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, 3, T_aligned, F_aligned)
        """
        if x.dim() == 5 and x.size(1) == 1:
            x = x.squeeze(1)  # 变成 [32, 3, 64, 32]
        # --- 1. CNN 特征提取 ---
        x = self.cnn(x)              # (B, C1, T', 1)
        
        # 安全地移除最后的维度，防止 Batch=1 时 squeeze 掉第 0 维
        if x.shape[-1] == 1:
            x = x.squeeze(-1)        # (B, C1, T')
        
        x = x.permute(0, 2, 1)       # (B, T', C1) -> 准备进入 GRU

        # --- 2. 时序建模 ---
        gru_out, _ = self.gru(x)     # (B, T', hidden*2)

        # --- 3. 注意力加权 ---
        context = self.attention(gru_out) # (B, hidden*2)

        # --- 4. 分类预测 ---
        logits = self.fc(context)    # (B, num_classes)
        return logits

    def get_init_params(self):
        """返回初始化参数，便于模型克隆或记录实验配置"""
        return {
            'win_len': self.win_len,
            'feature_size': self.feature_size,
            'num_classes': self.num_classes,
            'cnn_channels': self.cnn_channels,
            'gru_hidden': self.gru_hidden,
            'dropout': self.dropout
        }
# ==========================================
# 2. 学习率调度器 (Warmup + Cosine)
# ==========================================
def warmup_schedule(epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(max(1, warmup_epochs))
    else:
        return 0.5 * (1.0 + np.cos(np.pi * (epoch - warmup_epochs) / 50)) # 假设总epochs为50左右

# ==========================================
# 3. 高级实验引擎
# ==========================================
class ExperimentRunner:
    def __init__(self, cache_dir="./ablation_cache", output_dir="./experiment_results"):
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📡 运行设备: {self.device}")

    def to_dataloader(self, data_tuple, batch_size=32, shuffle=False):
        X, y = data_tuple
        # 调整形状以匹配CNN输入 (Batch, 1, T, K)
        X = X[:, np.newaxis, :, :] 
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)

    def train_model(self, model, train_loader, val_loader, epochs=60, patience=15):
        """升级版训练引擎：支持早停、梯度裁剪、自适应学习率"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) # 加入 L2 正则化防过拟合
        scheduler = LambdaLR(optimizer, lr_lambda=warmup_schedule)
        
        best_val_acc = 0.0
        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # 关键：梯度裁剪，防止 No_AGC 时因为没有归一化导致的梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += torch.sum(preds == labels.data).item()
                train_total += labels.size(0)
            
            scheduler.step() # 更新学习率
            
            # 验证阶段
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += torch.sum(preds == labels.data).item()
                    val_total += labels.size(0)
            
            epoch_train_loss = train_loss / train_total
            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:e}")
            
            # Early Stopping 逻辑 (监控验证集Loss)
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_val_acc = epoch_val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                print(f"⚠️ 早停触发：连续 {patience} 轮验证集Loss未下降。")
                break
                
        print(f"🎯 训练结束，加载最佳验证集模型 (Val Acc: {best_val_acc:.4f})")
        model.load_state_dict(best_model_wts)
        return model

    def evaluate(self, model, test_loader, cfg_name):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        # 使用 weighted F1，应对可能的数据类别不均衡
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        
        # 自动保存 Classification Report 到 CSV
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(self.output_dir, f"report_{cfg_name}.csv"))
        
        return acc, f1, cm

    def run_all_configs(self, splitter, train_devs, test_devs, configs=["Full_Path_A_WLS","Full_Path_B_CONJ","Ablation_No_AGC","Ablation_No_WLS","Ablation_No_Resample","Ablation_No_Spline","Ablation_No_PCA","Baseline_Raw"]):
        final_report = {}

        for cfg in configs:
            print(f"\n" + "="*60)
            print(f"🔎 正在执行消融实验: {cfg}")
            print("="*60)

            # 1. 加载并划分数据 
            # 论文答辩强力建议：如果你想验证极限泛化能力，把 strategy="cross_device" 加上
            data_dict = splitter.load_and_split(self.cache_dir, cfg, train_devs, test_devs,strategy="by_devices")
            
            # 动态获取类别数，防止之前出现的 IndexError: Target out of bounds
            num_classes_actual = int(np.max(data_dict["train"][1])) + 1
            
            train_loader = self.to_dataloader(data_dict["train"], shuffle=True)
            val_loader = self.to_dataloader(data_dict["val"], shuffle=False)
            test_loader = self.to_dataloader(data_dict["test"], shuffle=False)

            # 2. 初始化论文中的核心模型
            sample_x, _ = data_dict["train"]
            model = CNNBiGRUClassifier(
                win_len=sample_x.shape[1], 
                feature_size=sample_x.shape[2],
                num_classes=num_classes_actual
            ).to(self.device)

            # 3. 训练与评估
            model = self.train_model(model, train_loader, val_loader, epochs=60)
            acc, f1, cm = self.evaluate(model, test_loader, cfg)

            final_report[cfg] = {"accuracy": acc, "f1_score": f1}
            
            # 4. 保存混淆矩阵图片
            self.save_cm_plot(cm, cfg)
            print(f"✅ {cfg} 测试集结果 | Acc: {acc:.4f} | F1: {f1:.4f}")

        # 5. 保存最终 JSON 汇总和柱状图
        with open(os.path.join(self.output_dir, "ablation_summary.json"), "w") as f:
            json.dump(final_report, f, indent=4)
        
        self.plot_comparison_bar(final_report)

    def run_all_configs_lodo(self, splitter, configs=["Full_Path_A_WLS","Full_Path_B_CONJ","Ablation_No_AGC","Ablation_No_WLS","Ablation_No_Resample","Ablation_No_Spline","Ablation_No_PCA","Baseline_Raw"]):
        final_report = {}

        for cfg in configs:
            print(f"\n" + "="*60)
            print(f"🔎 正在执行消融实验: {cfg}")
            print("="*60)

            # 1. 这里的变量名改为 lodo_results，避免混淆
            lodo_results = splitter.load_and_split(self.cache_dir, cfg, strategy="lodo")
            
            final_report[cfg] = {} # 为当前实验初始化一个字典

            # 这里的 lodo_results 就是你之前的 data_results
            for device_name, data_dict in lodo_results.items():
                print(f"\n>>> 正在处理设备交叉验证: {device_name}")
                
                num_classes_actual = int(np.max(data_dict["train"][1])) + 1
            
                train_loader = self.to_dataloader(data_dict["train"], shuffle=True)
                val_loader = self.to_dataloader(data_dict["val"], shuffle=False)
                test_loader = self.to_dataloader(data_dict["test"], shuffle=False)

                # 2. 初始化模型
                sample_x, _ = data_dict["train"]
                model = CNNBiGRUClassifier(
                    win_len=sample_x.shape[1], 
                    feature_size=sample_x.shape[2],
                    num_classes=num_classes_actual
                ).to(self.device)

                # 3. 训练与评估
                model = self.train_model(model, train_loader, val_loader, epochs=60)
                acc, f1, cm = self.evaluate(model, test_loader, cfg)

                # 4. 记录结果：以设备名作为 Key，防止被覆盖
                final_report[cfg][device_name] = {"accuracy": acc, "f1_score": f1}
                
                # 保存每个设备的混淆矩阵，建议文件名带上设备名
                self.save_cm_plot(cm, f"{cfg}_{device_name}")
                print(f"✅ {device_name} 测试完成 | Acc: {acc:.4f} | F1: {f1:.4f}")

        # 5. 循环结束后再统一保存
        summary_path = os.path.join(self.output_dir, "lodo_ablation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(final_report, f, indent=4)
        
        # self.plot_comparison_bar(final_report) # 注意：这里的绘图函数可能需要修改以支持 LODO 多维度数据

    def save_cm_plot(self, cm, cfg_name):
        # 标签映射：数字 → 运动名称（和你的标注完全一致）
        LABEL_MAP = {
            "walking": 0,
            "seated-breathing": 1,
            "jumping": 2,
            "wavinghand": 3,
            "running": 4
        }
        # 生成按顺序的标签列表
        class_names = sorted(LABEL_MAP.keys(), key=lambda x: LABEL_MAP[x])
        
        plt.figure(figsize=(8, 6))
        # 核心修改：添加 xticklabels 和 yticklabels
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,  # x轴：运动名
            yticklabels=class_names   # y轴：运动名
        )
        plt.title(f"Confusion Matrix: {cfg_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        # 保存图片
        plt.savefig(
            os.path.join(self.output_dir, f"cm_{cfg_name}.png"),
            dpi=300,
            bbox_inches='tight'  # 防止标签被截断
        )
        plt.close()

    def plot_comparison_bar(self, report):
        names = list(report.keys())
        accs = [v['accuracy'] for v in report.values()]
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=names, y=accs, palette="viridis")
        plt.ylim(0.0, 1.0) # 建议从 0 开始，真实反映跌幅
        plt.ylabel("Test Accuracy")
        plt.title("Ablation Study: Accuracy Comparison")
        plt.xticks(rotation=45)
        
        # 在柱子上标注具体数值
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 10),
                        textcoords='offset points')
                        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "accuracy_comparison.png"), dpi=300)
        plt.close()
        print(f"\n🎉 所有实验完成！图表及报告已保存至 {self.output_dir} 目录。")

    def train_for_learning_curve(self, model, train_loader, val_loader, epochs=50):
        """专门为学习曲线设计的轻量化训练引擎"""
        criterion = nn.CrossEntropyLoss()
        # 调低一点学习率，确保小样本训练时不会震荡太厉害
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        
        # 记录历史，用于诊断
        best_val_acc = 0.0
        final_train_acc = 0.0

        for epoch in range(epochs):
            # --- 训练阶段 ---
            model.train()
            train_correct, train_total = 0, 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, preds = torch.max(outputs, 1)
                train_correct += torch.sum(preds == labels.data).item()
                train_total += labels.size(0)
            
            curr_train_acc = train_correct / train_total

            # --- 验证阶段 ---
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    val_correct += torch.sum(preds == labels.data).item()
                    val_total += labels.size(0)
            
            curr_val_acc = val_correct / val_total
            
            # 更新最高记录
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                final_train_acc = curr_train_acc # 记录对应时刻的训练准确率

            # 打印进度，方便观察
            if (epoch + 1) % 10 == 0:
                print(f"      [Epoch {epoch+1}/{epochs}] Train Acc: {curr_train_acc:.4f} | Val Acc: {curr_val_acc:.4f}")

        return final_train_acc, best_val_acc

    def generate_learning_curve(self, splitter, ratios=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]):
        # 1. 加载全量数据
        data_dict = splitter.load_and_split(self.cache_dir, "Full_Pipeline")
        num_classes_actual = int(np.max(data_dict["train"][1])) + 1
        
        full_train_x, full_train_y = data_dict["train"] # 修正变量引用
        test_x, test_y = data_dict["test"]
        val_x, val_y = data_dict["val"]

        train_accs = []
        test_accs = []
        
        for r in ratios:
            # 2. 采样：按比例随机抽取训练集
            num_samples = int(len(full_train_x) * r)
            # 物理逻辑：确保小比例采样也能覆盖动作类别
            indices = np.random.choice(len(full_train_x), num_samples, replace=False)
            subset_x = full_train_x[indices]
            subset_y = full_train_y[indices]
            
            # 3. 关键：重新构建子集的 DataLoader (否则训练循环无法读取 subset)
            subset_train_loader = self.to_dataloader((subset_x, subset_y), shuffle=True)
            test_loader = self.to_dataloader((test_x, test_y), shuffle=False)
            val_loader = self.to_dataloader((val_x, val_y), shuffle=False)

            # 4. 初始化模型 (确保每次循环都是一个全新的模型，无权重残留)
            model = CNNBiGRUClassifier(
                win_len=subset_x.shape[2],     # 注意你的维度索引 (B, C, T, F)
                feature_size=subset_x.shape[3],
                num_classes=num_classes_actual
            ).to(self.device)
            
            # 5. 执行训练
            # 注意：你的 train_model 内部需要改为接收 loader 或是处理好的 subset 数据
            t_acc, v_acc = self.train_for_learning_curve(model, subset_train_loader, val_loader, epochs=60)
            train_accs.append(t_acc)
            test_accs.append(v_acc)
            print(f"--- Ratio {r*100}%: Samples {num_samples}, Train Acc {t_acc:.4f}, Test Acc {v_acc:.4f} ---")

        return ratios, train_accs, test_accs

def plot_learning_curve(ratios, train_accs, test_accs):
    plt.figure(figsize=(10, 6))
    plt.plot(ratios, train_accs, 'o-', color="r", label="Training score")
    plt.plot(ratios, test_accs, 'o-', color="g", label="Test score")
    
    plt.title("Learning Curve (CSI-Activity Recognition)")
    plt.xlabel("Training Examples Ratio")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    # 建议保存图片以便写论文
    plt.savefig("learning_curve_analysis.png")

if __name__ == "__main__":
    # 1. 初始化
    #     # 1. 跨设备测试 (固定用户和环境)
    # data_dict = splitter.load_and_split(
    #     cache_dir, cfg, 
    #     train_targets=['AmazonEchoSpot'], 
    #     test_targets=['AmazonPlug'], 
    #     mode="device"
    # )

    # # 2. 跨用户测试 (固定设备和环境)
    # data_dict = splitter.load_and_split(
    #     cache_dir, cfg, 
    #     train_targets=['U02', 'U03'], 
    #     test_targets=['U04'], 
    #     mode="user"
    # )

    # # 3. 跨环境测试
    # data_dict = splitter.load_and_split(
    #     cache_dir, cfg, 
    #     train_targets=['E01'], 
    #     test_targets=['E03'], 
    #     mode="env"
    # )
    splitter = StandardDatasetSplitter(seed=42)
    config_manager = ConfigManager()
    configs = config_manager._load_from_file("/root/CSI-Bench-Real-WiFi-Sensing-Benchmark/CSI_system/utils/ablation_configs.json")
    # configs = config_manager.get_all_configs()
    all_names = [cfg.name for cfg in configs]

    # ==========================================================
    # 2. 定义测试维度 (根据你的 main.pdf 需求)
    # 你可以根据需要切换测试的目标：'device', 'user', 或 'environment'
    # ==========================================================
    TEST_MODE = "device"  # 可选: "device", "user", "environment"

    # 这里的定义要和你的数据集目录/文件名中的关键字匹配
    USERS = ["U02"]
    ENVS = ["E01"]
    DEVICES = ["AmazonEchoSpot", "AmazonPlug"] 

    # 选择当前要交叉验证的列表
    if TEST_MODE == "device":
        target_list = DEVICES
    elif TEST_MODE == "user":
        target_list = USERS
    else:
        target_list = ENVS

    # ==========================================================
    # 3. 自动多维交叉验证循环
    # ==========================================================
    for train_target in target_list:
        for test_target in target_list:
            # 只有在跨目标测试时才运行 (例如：用 A 设备练，在 B 设备测)
            if train_target == test_target:
                continue
            
            print(f"\n" + "█"*80)
            print(f"🚀 模式: 跨{TEST_MODE}对齐 | 训练集: {train_target} | 测试集: {test_target}")
            print("█"*80)

            # 自动生成输出目录
            output_path = f"/root/CSI-Bench-Real-WiFi-Sensing-Benchmark/CSI_system/X_{TEST_MODE}_{train_target}_to_{test_target}"
            
            runner = ExperimentRunner(
                cache_dir="/root/CSI-Bench-Real-WiFi-Sensing-Benchmark/CSI_system/ablation_processed_features_cross_dev",
                output_dir=output_path
            )

            # 核心修改点：将目标传递给 splitter
            # 注意：这里的 load_and_split 内部逻辑需要支持按 [train_target] 过滤
            # 如果你的 splitter 是按设备过滤的，TEST_MODE="device" 时如下：
            if TEST_MODE == "device":
                # 跨设备：固定用户 U05，固定环境 E01 (举例)
                runner.run_all_configs(splitter, [train_target], [test_target], all_names)
            
            elif TEST_MODE == "user":
                # 跨用户：可能需要在 splitter 中增加 user 参数支持
                # 这里假设你的 splitter 能够识别 train_devs 里的用户 ID
                runner.run_all_configs(splitter, [train_target], [test_target], all_names)

    print("\n✅ [User/Env/Device] 多维交叉验证全部执行完成！")