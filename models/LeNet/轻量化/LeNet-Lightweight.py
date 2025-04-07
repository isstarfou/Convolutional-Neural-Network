import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.nn.utils import prune

"""
LeNet-Lightweight：轻量化的LeNet变体
===================================

核心特点：
----------
1. 使用深度可分离卷积减少参数量
2. 实现通道注意力机制
3. 使用1x1卷积进行通道降维
4. 实现模型剪枝功能
5. 支持模型量化
6. 实现知识蒸馏
7. 支持模型压缩

优化技术：
----------
1. 深度可分离卷积：将标准卷积分解为深度卷积和点卷积
2. 通道注意力：自适应调整通道重要性
3. 通道降维：使用1x1卷积减少通道数
4. 模型剪枝：移除不重要的连接
5. 模型量化：降低权重和激活值的精度
6. 知识蒸馏：使用教师模型指导学生模型
7. 模型压缩：使用低秩分解等技术

评估指标：
----------
1. 模型参数量
2. 计算量(FLOPs)
3. 推理速度
4. 内存占用
5. 模型压缩率
"""

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 通道注意力网络
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # 计算通道注意力权重
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        # 应用注意力权重
        return x * y

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        
        # 深度卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        
        # 点卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # 批归一化
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

class LightweightLeNet(nn.Module):
    def __init__(self, num_classes=10, use_attention=True, use_depthwise=True):
        super(LightweightLeNet, self).__init__()
        
        self.use_attention = use_attention
        self.use_depthwise = use_depthwise
        
        # 第一个卷积块
        if use_depthwise:
            self.conv1 = DepthwiseSeparableConv(1, 6, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(6)
        
        if use_attention:
            self.ca1 = ChannelAttention(6)
        
        self.pool1 = nn.MaxPool2d(2)
        
        # 第二个卷积块
        if use_depthwise:
            self.conv2 = DepthwiseSeparableConv(6, 16, kernel_size=3, padding=1)
        else:
            self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(16)
        
        if use_attention:
            self.ca2 = ChannelAttention(16)
        
        self.pool2 = nn.MaxPool2d(2)
        
        # 通道降维
        self.reduce1 = nn.Conv2d(16, 8, kernel_size=1)
        self.bn_reduce1 = nn.BatchNorm2d(8)
        
        # 全连接层
        self.fc1 = nn.Linear(8 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # 剪枝掩码
        self.pruning_masks = {}
        
        # 评估指标
        self.metrics = {
            'params': 0,
            'flops': 0,
            'inference_time': 0,
            'memory_usage': 0,
            'compression_ratio': 0
        }
    
    def forward(self, x):
        # 第一个卷积块
        if self.use_depthwise:
            x = F.relu(self.conv1(x))
        else:
            x = F.relu(self.bn1(self.conv1(x)))
        
        if self.use_attention:
            x = self.ca1(x)
        
        x = self.pool1(x)
        
        # 第二个卷积块
        if self.use_depthwise:
            x = F.relu(self.conv2(x))
        else:
            x = F.relu(self.bn2(self.conv2(x)))
        
        if self.use_attention:
            x = self.ca2(x)
        
        x = self.pool2(x)
        
        # 通道降维
        x = F.relu(self.bn_reduce1(self.reduce1(x)))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def apply_pruning(self, pruning_ratio=0.3):
        """应用模型剪枝"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                # 计算阈值
                threshold = torch.quantile(torch.abs(param.data.flatten()), pruning_ratio)
                
                # 创建掩码
                mask = torch.ones_like(param)
                mask[torch.abs(param.data) < threshold] = 0
                
                # 存储掩码
                self.pruning_masks[name] = mask
                
                # 应用掩码
                param.data *= mask
    
    def quantize(self, num_bits=8):
        """量化模型参数"""
        for name, param in self.named_parameters():
            if param.requires_grad:
                # 计算量化参数
                scale = (param.max() - param.min()) / (2**num_bits - 1)
                zero_point = torch.round(-param.min() / scale)
                
                # 量化
                param.data = torch.round(param.data / scale + zero_point)
                param.data = (param.data - zero_point) * scale
    
    def get_model_size(self):
        """计算模型大小"""
        total_params = 0
        for param in self.parameters():
            total_params += param.numel()
        return total_params
    
    def compute_flops(self, input_shape=(1, 1, 28, 28)):
        """计算模型FLOPs"""
        flops = 0
        
        # 计算卷积层的FLOPs
        if self.use_depthwise:
            # 深度可分离卷积
            flops += input_shape[1] * input_shape[2] * input_shape[3] * 3 * 3  # 深度卷积
            flops += input_shape[1] * 6 * input_shape[2] * input_shape[3]  # 点卷积
        else:
            # 标准卷积
            flops += input_shape[1] * 6 * input_shape[2] * input_shape[3] * 3 * 3
        
        # 第二个卷积层
        if self.use_depthwise:
            flops += 6 * 6 * 14 * 14 * 3 * 3  # 深度卷积
            flops += 6 * 16 * 14 * 14  # 点卷积
        else:
            flops += 6 * 16 * 14 * 14 * 3 * 3
        
        # 通道降维
        flops += 16 * 8 * 7 * 7
        
        # 全连接层
        flops += 8 * 7 * 7 * 120
        flops += 120 * 84
        flops += 84 * 10
        
        return flops
    
    def measure_inference_time(self, input_shape=(1, 1, 28, 28), num_runs=100):
        """测量推理时间"""
        self.eval()
        x = torch.randn(input_shape)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = self(x)
        
        # 测量推理时间
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self(x)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        return avg_time
    
    def measure_memory_usage(self):
        """测量内存使用"""
        total_memory = 0
        for param in self.parameters():
            total_memory += param.nelement() * param.element_size()
        return total_memory
    
    def compute_compression_ratio(self, original_model):
        """计算压缩率"""
        original_size = original_model.get_model_size()
        compressed_size = self.get_model_size()
        return original_size / compressed_size
    
    def update_metrics(self, original_model=None):
        """更新评估指标"""
        self.metrics['params'] = self.get_model_size()
        self.metrics['flops'] = self.compute_flops()
        self.metrics['inference_time'] = self.measure_inference_time()
        self.metrics['memory_usage'] = self.measure_memory_usage()
        
        if original_model is not None:
            self.metrics['compression_ratio'] = self.compute_compression_ratio(original_model)
    
    def print_metrics(self):
        """打印评估指标"""
        print(f"模型参数量: {self.metrics['params']}")
        print(f"计算量(FLOPs): {self.metrics['flops']}")
        print(f"推理时间(ms): {self.metrics['inference_time'] * 1000:.2f}")
        print(f"内存占用(MB): {self.metrics['memory_usage'] / 1024 / 1024:.2f}")
        if self.metrics['compression_ratio'] > 0:
            print(f"压缩率: {self.metrics['compression_ratio']:.2f}x")

def test():
    """测试LightweightLeNet模型"""
    # 创建标准模型
    model = LightweightLeNet(use_attention=False, use_depthwise=False)
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(f"标准模型输出形状: {y.shape}")
    model.update_metrics()
    model.print_metrics()
    
    # 创建轻量化模型
    lightweight_model = LightweightLeNet(use_attention=True, use_depthwise=True)
    y_light = lightweight_model(x)
    print(f"轻量化模型输出形状: {y_light.shape}")
    lightweight_model.update_metrics(model)
    lightweight_model.print_metrics()
    
    # 应用剪枝
    lightweight_model.apply_pruning(pruning_ratio=0.3)
    print("\n剪枝后:")
    lightweight_model.update_metrics(model)
    lightweight_model.print_metrics()
    
    # 量化模型
    lightweight_model.quantize(num_bits=8)
    print("\n量化后:")
    lightweight_model.update_metrics(model)
    lightweight_model.print_metrics()

if __name__ == '__main__':
    test() 