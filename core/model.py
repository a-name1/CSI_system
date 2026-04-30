import torch
import torch.nn as nn
import math

class MLPClassifier(nn.Module):
    def __init__(self, win_len=500, feature_size=232, num_classes=6, in_channels=1):
        super().__init__()
        self.flatten = nn.Flatten()
        # 自动计算全连接层输入维度
        self.fc = nn.Sequential(
            nn.Linear(in_channels * win_len * feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [B, C, T, F]
        x = self.flatten(x)
        return self.fc(x)

class LSTMClassifier(nn.Module):
    def __init__(self, feature_size=232, hidden_size=128, num_layers=2, num_classes=6, in_channels=1):
        super().__init__()
        # 将 Channel 和 Feature 合并作为输入维度
        self.input_dim = feature_size * in_channels
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: [B, C, T, F] -> [B, T, C*F]
        batch, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, t, -1)
        
        out, _ = self.lstm(x)
        # 取最后一个时段的输出
        return self.fc(out[:, -1, :])

from torchvision.models import resnet18

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=6, in_channels=1):
        super().__init__()
        self.model = resnet18(weights=None)
        # 修改第一层卷积以适配 CSI 通道数 (通常为 1 或 2)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改最后全连接层
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: [B, C, T, F] 直接符合 ResNet 的 Conv2d 要求
        return self.model(x)

class PatchTST(nn.Module):
    def __init__(self, win_len=500, feature_size=232, patch_len=16, stride=8, 
                 emb_dim=128, num_heads=4, depth=3, num_classes=6, in_channels=1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        # 每个 patch 的原始维度：feature_size * patch_len
        self.patch_input_dim = feature_size * patch_len
        self.num_patches = (win_len - patch_len) // stride + 1
        
        self.proj = nn.Linear(self.patch_input_dim, emb_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # x: [B, C, T, F] -> 降维并切片
        if x.shape[1] > 1: # 如果是多通道，先融合或取单通道
            x = x.mean(dim=1) 
        else:
            x = x.squeeze(1) # [B, T, F]

        # 简单的切片实现 (Unfold)
        # x: [B, T, F] -> [B, num_patches, patch_len * F]
        patches = x.unfold(1, self.patch_len, self.stride) # [B, N, F, P_L]
        patches = patches.reshape(patches.size(0), patches.size(1), -1)
        
        x = self.proj(patches) + self.pos_embed
        x = self.transformer(x)
        return self.fc(x.mean(dim=1)) # 全局平均池化后分类
    
class ViTClassifier(nn.Module):
    def __init__(self, win_len=500, feature_size=232, patch_size=20, 
                 emb_dim=128, num_heads=4, depth=4, num_classes=6, in_channels=1):
        super().__init__()
        # 确保 patch_size 能被整除，或者在此处做 padding
        self.patch_size = patch_size
        num_patches = (win_len // patch_size) * (feature_size // patch_size)
        
        self.patch_embed = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # x: [B, C, T, F]
        x = self.patch_embed(x) # [B, E, T', F']
        x = x.flatten(2).transpose(1, 2) # [B, N, E]
        
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        x = self.transformer(x)
        return self.fc(x[:, 0]) # 取 CLS token 结果