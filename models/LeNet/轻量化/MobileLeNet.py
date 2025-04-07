import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

"""
MobileLeNet：基于MobileNet思想的轻量化LeNet
========================================

核心特点：
----------
1. 深度可分离卷积
2. 通道注意力机制
3. 轻量化设计
4. 高效特征提取
5. 低计算复杂度

实现原理：
----------
1. 使用深度可分离卷积替代标准卷积
2. 引入SE注意力模块
3. 优化网络结构
4. 减少参数量和计算量
5. 保持模型性能

评估指标：
----------
1. 模型参数量
2. 计算复杂度(FLOPs)
3. 推理速度
4. 准确率
5. 内存占用
"""

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # 深度卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        
        # 点卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # 批归一化
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        
        return x

class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力模块"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MobileLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileLeNet, self).__init__()
        
        # 特征提取网络
        self.features = nn.Sequential(
            # 第一层使用标准卷积
            nn.Conv2d(3, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 深度可分离卷积块1
            DepthwiseSeparableConv(6, 16),
            SEBlock(16),
            nn.MaxPool2d(2),
            
            # 深度可分离卷积块2
            DepthwiseSeparableConv(16, 32),
            SEBlock(32),
            nn.MaxPool2d(2),
            
            # 深度可分离卷积块3
            DepthwiseSeparableConv(32, 64),
            SEBlock(64)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(120, num_classes)
        )
        
        # 评估指标
        self.metrics = {
            'params': self._count_parameters(),
            'flops': self._count_flops(),
            'accuracy': 0,
            'inference_time': 0,
            'memory_usage': 0
        }
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _count_parameters(self):
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters())
    
    def _count_flops(self):
        """计算模型FLOPs"""
        # 这里使用简化的计算方法
        # 实际应用中应该使用更精确的FLOPs计算工具
        flops = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                flops += m.kernel_size[0] * m.kernel_size[1] * m.in_channels * m.out_channels
            elif isinstance(m, nn.Linear):
                flops += m.in_features * m.out_features
        return flops
    
    def update_metrics(self, accuracy, inference_time, memory_usage):
        """更新评估指标"""
        self.metrics['accuracy'] = accuracy
        self.metrics['inference_time'] = inference_time
        self.metrics['memory_usage'] = memory_usage
    
    def print_metrics(self):
        """打印评估指标"""
        print("\n模型评估指标:")
        print(f"参数量: {self.metrics['params']:,}")
        print(f"FLOPs: {self.metrics['flops']:,}")
        print(f"准确率: {self.metrics['accuracy']:.2f}%")
        print(f"推理时间: {self.metrics['inference_time']:.4f}秒")
        print(f"内存占用: {self.metrics['memory_usage']:.2f}MB")

def test():
    """测试函数"""
    # 创建模型
    model = MobileLeNet(num_classes=10)
    
    # 创建随机输入
    x = torch.randn(1, 3, 32, 32)
    
    # 前向传播
    output = model(x)
    
    # 打印输出形状
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 打印模型信息
    model.print_metrics()

if __name__ == '__main__':
    test()
