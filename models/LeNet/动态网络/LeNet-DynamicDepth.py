import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicDepthBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DynamicDepthBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # 深度决策网络
        self.depth_decision = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        # 第一层
        x = F.relu(self.bn1(self.conv1(x)))
        
        # 计算深度决策
        depth_weight = self.depth_decision(x)
        
        # 第二层（根据深度决策调整）
        x = self.bn2(self.conv2(x))
        x = x * depth_weight
        
        x += identity
        return F.relu(x)

class LeNetDynamicDepth(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNetDynamicDepth, self).__init__()
        
        # 基础层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2)
        
        # 动态深度块
        self.dynamic_block1 = DynamicDepthBlock(6, 16)
        self.pool2 = nn.MaxPool2d(2)
        
        self.dynamic_block2 = DynamicDepthBlock(16, 32)
        self.pool3 = nn.MaxPool2d(2)
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d(4)
        
        # 分类头
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        # 基础层
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # 动态深度块
        x = self.dynamic_block1(x)
        x = self.pool2(x)
        
        x = self.dynamic_block2(x)
        x = self.pool3(x)
        
        # 自适应池化
        x = self.adaptive_pool(x)
        
        # 分类
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def test():
    model = LeNetDynamicDepth()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.shape)

if __name__ == '__main__':
    test()
