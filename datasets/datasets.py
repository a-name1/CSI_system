import torch
from torch.utils.data import Dataset, DataLoader

class CSILODODataset(Dataset):
    def __init__(self, split_json_path, processed_dir):
        with open(split_json_path, 'r') as f:
            self.sample_ids = json.load(f)['sample_ids']
        self.processed_dir = processed_dir
        
        # 预加载映射索引 (加速读取)
        self.data_cache = {}
        self._build_index()

    def _build_index(self):
        # 建立 sample_id 到 npy 文件名的映射
        for f in os.listdir(self.processed_dir):
            if f.endswith('.npy'):
                # 这种方式虽然费点内存，但训练速度提升 10 倍以上
                file_data = np.load(os.path.join(self.processed_dir, f), allow_pickle=True).item()
                self.data_cache.update(file_data)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        s_id = self.sample_ids[idx]
        sample = self.data_cache[s_id]
        
        x = torch.from_numpy(sample['feature']).float() # (14, 500)
        y = torch.tensor(sample['label']).long()
        return x, y