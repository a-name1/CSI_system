import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

# 导入你的核心组件
from config_manager import ConfigManager
from pipeline_executor import PreprocessPipeline
from data_manager import LocalFileCacheManager

def run_targeted_ablation(target_users, target_envs, target_devices, samples_root, cache_root, config_path):
    """
    核心消融逻辑函数
    """
    # 1. 初始化组件
    config_manager = ConfigManager(config_path)
    pipeline = PreprocessPipeline()
    cache_manager = LocalFileCacheManager(cache_root)
    all_configs = config_manager.get_all_configs()
    
    # 解析文件名的正则: Uxx_Exx_Device.npz
    file_pattern = re.compile(r"^(U\d+)_(E\d+)_(.+)\.npz$")
    
    # 2. 筛选符合条件的样本文件
    all_files = [f for f in os.listdir(samples_root) if f.endswith(".npz")]
    selected_files = []
    for f in all_files:
        match = file_pattern.match(f)
        if match:
            u, e, dev = match.groups()
            if u in target_users and e in target_envs and dev in target_devices:
                selected_files.append((f, u, e, dev))

    if not selected_files:
        print(f"⚠️ 在 {samples_root} 未找到符合条件的样本文件，请检查白名单设置。")
        return

    print(f"🎯 筛选完成，准备处理 {len(selected_files)} 个数据...")

    # 3. 嵌套循环：消融配置 -> 样本文件 -> 单个样本
    for cfg in all_configs:
        print(f"\n" + "="*50)
        print(f"🔥 正在生成消融特征: {cfg.name}")
        print("="*50)

        # 为每个消融实验创建子目录
        abl_cfg_dir = os.path.join(cache_root, cfg.name)
        os.makedirs(abl_cfg_dir, exist_ok=True)

        for s_file, u, e, dev in tqdm(selected_files, desc=f"Config: {cfg.name}"):
            # 加载预提取的 NPZ
            data = np.load(os.path.join(samples_root, s_file), allow_pickle=True)
            samples_list = data['samples']
            
            processed_x = []
            labels = []
            
            for sample in samples_list:
                # 核心处理步骤：根据 cfg 决定是否 Resample/Filter
                feat = pipeline.process_batch(
                    [sample['raw_feature']], 
                    [sample['time']], 
                    dev, 
                    cfg, 
                    is_training=True
                )
                processed_x.append(feat)
                labels.append(sample['label'])

            # 转换为矩阵
            X = np.array(processed_x).squeeze()
            y = np.array(labels)
            
            # 保存该设备在当前配置下的最终特征
            save_name = f"{u}_{e}_{dev}_feat.npz"
            save_path = os.path.join(abl_cfg_dir, save_name)
            
            np.savez_compressed(
                save_path, 
                x=X, y=y, 
                user=u, env=e, device=dev, 
                cfg_name=cfg.name
            )

            # 及时释放内存，防止 2TiB 内存也被分片占用
            del X, y, processed_x, labels, samples_list
            gc.collect()

# ======================
# 主入口 (Main)
# ======================
if __name__ == "__main__":
    # --- 1. 路径与元数据配置 ---
    SAMPLES_ROOT = "/root/CSI_system/sample_cross_dev/raw"  # 刚才生成的 npz 存放目录
    CACHE_ROOT = "/root/CSI_system/ablation_add_new_pca" # 消融特征存放根目录
    CONFIG_PATH = "/root/CSI_system/config/ablation_configs_dft.json"
    
    # --- 2. 实验目标白名单 (在这里精确指定你要处理的对象) ---
    TARGET_USERS = ["U02"]# 可以根据需要添加，如 ["U01", "U02", "U03", "U04", "U05", "U06"] 
    TARGET_ENVS = ["E01"]   # 可以根据需要添加，如 ["E01", "E02", "E03", "E04", "E05", "E06"]
    TARGET_DEVICES = [
        'AmazonEchoSpot', 'AmazonPlug'
    ]
    # TARGET_DEVICES = [
    #     'AmazonEchoPlus', 'AmazonEchoShow8', 'AmazonEchoSpot', 'AmazonPlug',
    #     'AppleHomePod', 'EighttreePlug', 'GoogleNest', 'GoveeSmartPlug',
    #     'HealthPod1', 'HealthPod2', 'HealthPod3', 'WyzePlug'
    # ]
    # 标签映射（和你的标注一致）
    LABEL_MAP = {
        "walking": 0,
        "seated-breathing": 1,
        "jumping": 2,
        "wavinghand": 3,
        "running": 4
    }
    # --- 3. 运行环境检查 ---
    if not os.path.exists(SAMPLES_ROOT):
        print(f"❌ 样本目录 {SAMPLES_ROOT} 不存在，请先运行数据提取脚本。")
    else:
        # 执行消融实验
        run_targeted_ablation(
            target_users=TARGET_USERS,
            target_envs=TARGET_ENVS,
            target_devices=TARGET_DEVICES,
            samples_root=SAMPLES_ROOT,
            cache_root=CACHE_ROOT,
            config_path=CONFIG_PATH
        )
        
        print(f"\n🎉 指定范围的消融实验特征生成完毕！")
        print(f"📂 结果保存在: {CACHE_ROOT}")