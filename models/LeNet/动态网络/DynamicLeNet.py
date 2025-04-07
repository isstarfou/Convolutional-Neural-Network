import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicRouting(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DynamicRouting, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.routing = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, 2, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        routing_weights = self.routing(x)
        return x * routing_weights[:, 0:1, :, :]

class DynamicLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(DynamicLeNet, self).__init__()
        
        # 动态路由模块
        self.dynamic_conv1 = DynamicRouting(1, 6)
        self.pool1 = nn.MaxPool2d(2)
        
        self.dynamic_conv2 = DynamicRouting(6, 16)
        self.pool2 = nn.MaxPool2d(2)
        
        # 自适应特征聚合
        self.adaptive_pool = nn.AdaptiveAvgPool2d(4)
        
        # 分类头
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        # 动态路由路径
        x = self.dynamic_conv1(x)
        x = self.pool1(x)
        
        x = self.dynamic_conv2(x)
        x = self.pool2(x)
        
        # 自适应特征聚合
        x = self.adaptive_pool(x)
        
        # 分类
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def test():
    model = DynamicLeNet()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.shape)

if __name__ == '__main__':
    test() 