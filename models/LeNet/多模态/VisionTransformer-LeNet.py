import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=256):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x):
        x = self.proj(x)  # B, E, H/P, W/P
        x = x.flatten(2).transpose(1, 2)  # B, N, E
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class LeNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(LeNetFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        return x

class VisionTransformerLeNet(nn.Module):
    def __init__(self, num_classes=10, embed_dim=256, num_heads=8, num_layers=6):
        super(VisionTransformerLeNet, self).__init__()
        
        # 图像分块嵌入
        self.patch_embed = PatchEmbedding()
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
        
        # Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # LeNet特征提取器
        self.lenet = LeNetFeatureExtractor()
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + 16 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # 分类头
        self.norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Transformer路径
        x_trans = self.patch_embed(x)
        x_trans = x_trans + self.pos_embed
        
        for blk in self.blocks:
            x_trans = blk(x_trans)
            
        x_trans = x_trans.mean(dim=1)  # 全局平均池化
        
        # LeNet路径
        x_lenet = self.lenet(x)
        x_lenet = x_lenet.flatten(1)
        
        # 特征融合
        x = torch.cat([x_trans, x_lenet], dim=1)
        x = self.fusion(x)
        
        # 分类
        x = self.norm(x)
        x = self.head(x)
        
        return x

def test():
    model = VisionTransformerLeNet()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.shape)

if __name__ == '__main__':
    test()
