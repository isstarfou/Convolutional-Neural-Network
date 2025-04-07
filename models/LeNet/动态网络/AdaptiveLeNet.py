import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexityEstimator(nn.Module):
    def __init__(self, in_channels):
        super(ComplexityEstimator, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.sigmoid(x)

class AdaptiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptiveBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x, complexity):
        identity = self.shortcut(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        # 根据复杂度调整特征
        x = x * complexity.view(-1, 1, 1, 1)
        
        x += identity
        return F.relu(x)

class AdaptiveLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AdaptiveLeNet, self).__init__()
        
        # 输入分析模块
        self.complexity_estimator = ComplexityEstimator(1)
        
        # 自适应块
        self.adaptive_block1 = AdaptiveBlock(1, 6)
        self.pool1 = nn.MaxPool2d(2)
        
        self.adaptive_block2 = AdaptiveBlock(6, 16)
        self.pool2 = nn.MaxPool2d(2)
        
        # 动态特征集成
        self.adaptive_pool = nn.AdaptiveAvgPool2d(4)
        
        # 分类头
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        # 估计输入复杂度
        complexity = self.complexity_estimator(x)
        
        # 自适应处理
        x = self.adaptive_block1(x, complexity)
        x = self.pool1(x)
        
        x = self.adaptive_block2(x, complexity)
        x = self.pool2(x)
        
        # 特征集成
        x = self.adaptive_pool(x)
        
        # 分类
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def test():
    model = AdaptiveLeNet()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.shape)

if __name__ == '__main__':
    test() 