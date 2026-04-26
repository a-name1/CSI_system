import os
import gc
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import silhouette_score

# ==========================================
# 1. 核心数学评估工具函数
# ==========================================

def calculate_mmd(X1, X2, gamma=1.0):
    """
    计算两个领域（Domain）之间的最大均值差异 (Maximum Mean Discrepancy)。
    MMD 越小，说明两个环境/设备下的特征分布越接近，特征越具有不变性（Robustness）。
    """
    K11 = rbf_kernel(X1, X1, gamma)
    K22 = rbf_kernel(X2, X2, gamma)
    K12 = rbf_kernel(X1, X2, gamma)
    return np.mean(K11) + np.mean(K22) - 2 * np.mean(K12)

def calculate_fisher_score(X, labels):
    """
    计算 Fisher Score。
    用于衡量动作类别之间的可分性。
    Fisher Score 越大，说明类间距离越大、类内距离越小（动作区分度越好）。
    """
    unique_labels = np.unique(labels)
    overall_mean = np.mean(X, axis=0)
    sw, sb = 0, 0  # sw: 类内散度, sb: 类间散度
    for label in unique_labels:
        cluster = X[labels == label]
        if len(cluster) < 2: continue
        m_i = np.mean(cluster, axis=0)
        sw += np.sum(np.sum((cluster - m_i)**2, axis=1))
        sb += len(cluster) * np.sum((m_i - overall_mean)**2)
    return sb / (sw + 1e-6)

# ==========================================
# 2. t-SNE 绘图函数
# ==========================================

def plot_tsne_comparison(X, y, domain_labels, title, save_path, tags, label_map):
    """
    执行 t-SNE 降维并生成对比散点图。
    """
    # 初始化 t-SNE 模型，设置随机种子保证结果可复现
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(X)

    # 预计算指标以便在图中左上角显示文本框
    mmd_val = calculate_mmd(X[domain_labels == 0], X[domain_labels == 1])
    sil_act = silhouette_score(X, y)
    fisher_act = calculate_fisher_score(X, y)

    plt.figure(figsize=(12, 8))
    markers = ['o', 'x']  # o 代表源域/环境1, x 代表目标域/环境2
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # 遍历每个动作类别进行绘图
    for act_id, act_name in label_map.items():
        color = colors[act_id % len(colors)]
        for d_idx in [0, 1]:
            # 筛选出属于该动作且属于该领域的数据点
            idx = (y == act_id) & (domain_labels == d_idx)
            if np.sum(idx) == 0: continue
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1], marker=markers[d_idx], 
                        color=color, alpha=0.6, s=45,
                        label=f"{act_name} ({tags[d_idx]})")

    # 在图上添加数值指标信息面板
    stats_box = (f"MMD: {mmd_val:.4f}\nAct Sil: {sil_act:.4f}\nAct Fisher: {fisher_act:.4f}")
    plt.gca().text(0.02, 0.98, stats_box, transform=plt.gca().transAxes, 
                   fontsize=10, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return mmd_val, sil_act, fisher_act

# ==========================================
# 3. 执行主逻辑 (Main)
# ==========================================

if __name__ == "__main__":
    # --- 实验参数配置 ---
    COMPARE_MODE = {
        # 数据根目录（消融实验提取出的特征所在位置）
        "base_dir": "/root/CSI_system/ablation_processed_features_cross_dev",
        # 结果保存根目录
        "save_root": "/root/CSI_system/tsne_ablation_plots/ablation_processed_features_cross_dev",
        
        # 想要对比的维度: "env" (跨环境对比), "user" (跨用户), "device" (跨设备)
        "dim": "device", 
        "tag_1": "AmazonEchoSpot", # 对比标签 1
        "tag_2": "AmazonPlug", # 对比标签 2
        
        # 固定变量过滤: 设为 None 表示包含所有，或者填入 "U02" 只看特定用户
        "filter_user": "U02", 
        "filter_device": None
    }

    # 动作 ID 与名称的映射表
    ACTION_LABELS = {0: "walking", 1: "seated-breathing", 2: "jumping", 3: "wavinghand", 4: "running"}

    # 自动生成当前实验的后缀名和保存路径
    exp_suffix = f"{COMPARE_MODE['dim']}_{COMPARE_MODE['tag_1']}_vs_{COMPARE_MODE['tag_2']}"
    save_dir = os.path.join(COMPARE_MODE["save_root"], f"compare_{exp_suffix}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 报表文件 CSV 的存储路径
    csv_path = os.path.join(save_dir, f"metrics_report_{exp_suffix}.csv")

    # 获取 base_dir 下所有的配置子文件夹（即每个不同的消融实验配置）
    all_subdirs = [d for d in os.listdir(COMPARE_MODE["base_dir"]) if os.path.isdir(os.path.join(COMPARE_MODE["base_dir"], d))]
    
    # 初始化 CSV 写入
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Ablation_Config", "Domain_MMD", "Action_Silhouette", "Action_Fisher_Score"])

        for cfg_name in all_subdirs:
            full_subdir_path = os.path.join(COMPARE_MODE["base_dir"], cfg_name)
            data_x, data_y, domain_labels = [], [], []
            found_domains = set()  # 用于记录是否找到了 tag_1 和 tag_2

            # 遍历子文件夹内的所有 .npz 文件
            for fname in os.listdir(full_subdir_path):
                if not fname.endswith(".npz"): continue
                
                # 解析文件名，提取 U, E, Device 信息
                parts = fname.replace("_feat.npz", "").split("_")
                meta = {"user": parts[0], "env": parts[1], "device": parts[2]}
                
                # 执行条件过滤
                if COMPARE_MODE["filter_user"] and meta["user"] != COMPARE_MODE["filter_user"]: continue
                if COMPARE_MODE["filter_device"] and meta["device"] != COMPARE_MODE["filter_device"]: continue
                
                # 确定该文件属于对比实验的哪一侧
                if meta[COMPARE_MODE["dim"]] == COMPARE_MODE["tag_1"]:
                    d_idx = 0; found_domains.add(0)
                elif meta[COMPARE_MODE["dim"]] == COMPARE_MODE["tag_2"]:
                    d_idx = 1; found_domains.add(1)
                else: continue
                
                # 加载 npz 文件并对特征进行平坦化处理 (Batch, Flatten_Feature)
                with np.load(os.path.join(full_subdir_path, fname)) as data:
                    data_x.append(data['x'].reshape(len(data['x']), -1))
                    data_y.append(data['y'])
                    domain_labels.append(np.full(len(data['y']), d_idx))

            # 只有当两个领域的数据都找齐了，才开始绘图和评估
            if len(found_domains) < 2:
                print(f"⏩ Skip {cfg_name}: 不满足对比条件（缺失数据）")
                continue

            # 合并当前配置下的所有数据
            X = np.vstack(data_x)
            y = np.concatenate(data_y)
            dom = np.concatenate(domain_labels)

            # 调用绘图函数并生成指标
            plot_path = os.path.join(save_dir, f"tsne_{cfg_name}.png")
            title = f"Config: {cfg_name} | {COMPARE_MODE['tag_1']} vs {COMPARE_MODE['tag_2']}"
            
            m_mmd, m_sil, m_fisher = plot_tsne_comparison(
                X, y, dom, title, plot_path, 
                [COMPARE_MODE["tag_1"], COMPARE_MODE["tag_2"]], ACTION_LABELS
            )
            
            # 将量化结果记录在报表中
            writer.writerow([cfg_name, f"{m_mmd:.6f}", f"{m_sil:.6f}", f"{m_fisher:.6f}"])
            print(f"✅ {cfg_name} 评估完成。")

            # 及时释放大数据内存，防止 OOM
            del X, y, dom, data_x, data_y, domain_labels
            gc.collect()

    print(f"\n✨ 分析结束！")
    print(f"🖼️ t-SNE 散点图存放在: {save_dir}")
    print(f"📊 指标报表存放在: {csv_path}")