import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageBranch(nn.Module):
    def __init__(self):
        super(ImageBranch, self).__init__()
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

class TextBranch(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=256):
        super(TextBranch, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return x.mean(dim=1)  # 时序平均池化

class FeatureFusion(nn.Module):
    def __init__(self, image_dim, text_dim, fusion_dim=256):
        super(FeatureFusion, self).__init__()
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
    def forward(self, image_feat, text_feat):
        image_feat = self.image_proj(image_feat)
        text_feat = self.text_proj(text_feat)
        
        # 特征融合
        fused = torch.cat([image_feat, text_feat], dim=1)
        fused = self.fusion(fused)
        
        return fused

class LeNetMultimodal(nn.Module):
    def __init__(self, num_classes=10, vocab_size=10000):
        super(LeNetMultimodal, self).__init__()
        
        # 图像分支
        self.image_branch = ImageBranch()
        
        # 文本分支
        self.text_branch = TextBranch(vocab_size=vocab_size)
        
        # 特征融合
        self.fusion = FeatureFusion(
            image_dim=16 * 7 * 7,  # LeNet特征维度
            text_dim=512  # 双向LSTM输出维度
        )
        
        # 分类头
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, image, text):
        # 图像特征提取
        image_feat = self.image_branch(image)
        image_feat = image_feat.flatten(1)
        
        # 文本特征提取
        text_feat = self.text_branch(text)
        
        # 特征融合
        fused = self.fusion(image_feat, text_feat)
        
        # 分类
        x = F.relu(self.fc1(fused))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def test():
    model = LeNetMultimodal()
    
    # 模拟输入
    image = torch.randn(1, 1, 28, 28)  # 图像输入
    text = torch.randint(0, 10000, (1, 50))  # 文本输入（序列长度50）
    
    y = model(image, text)
    print(y.shape)

if __name__ == '__main__':
    test()
