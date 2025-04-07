import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

"""
LeNet-Lite：轻量级LeNet模型
========================

核心特点：
----------
1. 多维度轻量化
2. 高效特征提取
3. 低计算复杂度
4. 快速推理速度
5. 适合边缘设备

实现原理：
----------
1. 使用1x1卷积降维
2. 引入分组卷积
3. 优化网络结构
4. 减少参数量
5. 保持模型性能

评估指标：
----------
1. 模型大小
2. 计算量
3. 推理速度
4. 准确率
5. 内存效率
"""

class GroupConv(nn.Module):
    """分组卷积模块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super(GroupConv, self).__init__()
        
        # 1x1卷积降维
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 分组卷积
        self.group_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        x = self.group_conv(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        
        return x

class LeNetLite(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNetLite, self).__init__()
        
        # 特征提取网络
        self.features = nn.Sequential(
            # 第一层卷积 (3x32x32 -> 4x28x28)
            GroupConv(3, 4, kernel_size=5, padding=0),
            nn.MaxPool2d(2),  # 4x14x14
            
            # 第二层卷积 (4x14x14 -> 8x10x10)
            GroupConv(4, 8, kernel_size=5, padding=0),
            nn.MaxPool2d(2),  # 8x5x5
            
            # 第三层卷积 (8x5x5 -> 16x1x1)
            GroupConv(8, 16, kernel_size=5, padding=0)
        )
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            # 全连接层 (16 -> 32)
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            # 输出层 (32 -> num_classes)
            nn.Linear(32, num_classes)
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _count_parameters(self):
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters())
    
    def _count_flops(self):
        """计算模型FLOPs"""
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
    
    def visualize_feature_maps(self, x):
        """可视化特征图"""
        # 存储各层输出
        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        # 注册钩子
        hooks = []
        for name, layer in self.features.named_children():
            hooks.append(layer.register_forward_hook(get_activation(name)))
        
        # 前向传播
        with torch.no_grad():
            self(x)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 可视化特征图
        plt.figure(figsize=(15, 5))
        for i, (name, act) in enumerate(activations.items()):
            plt.subplot(1, len(activations), i+1)
            plt.imshow(act[0, 0].cpu().numpy(), cmap='viridis')
            plt.title(f'{name}\n{act.shape[1]} channels')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def analyze_efficiency(self):
        """分析模型效率"""
        efficiency_stats = {
            'conv_layers': {},
            'fc_layers': {},
            'total_params': 0,
            'total_flops': 0,
            'total_memory': 0
        }
        
        for name, module in self.named_modules():
            if isinstance(module, (GroupConv, nn.Conv2d, nn.Linear)):
                # 计算参数量
                params = sum(p.numel() for p in module.parameters())
                # 计算FLOPs
                if isinstance(module, (GroupConv, nn.Conv2d)):
                    flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels * module.out_channels
                else:
                    flops = module.in_features * module.out_features
                # 计算内存占用
                memory = params * 4 / 1024 / 1024  # 转换为MB
                
                if isinstance(module, (GroupConv, nn.Conv2d)):
                    efficiency_stats['conv_layers'][name] = {
                        'params': params,
                        'flops': flops,
                        'memory_mb': memory
                    }
                else:
                    efficiency_stats['fc_layers'][name] = {
                        'params': params,
                        'flops': flops,
                        'memory_mb': memory
                    }
                
                efficiency_stats['total_params'] += params
                efficiency_stats['total_flops'] += flops
                efficiency_stats['total_memory'] += memory
        
        return efficiency_stats

def test():
    """测试函数"""
    # 创建模型
    model = LeNetLite(num_classes=10)
    
    # 创建随机输入
    x = torch.randn(1, 3, 32, 32)
    
    # 前向传播
    output = model(x)
    
    # 打印输出形状
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 打印模型信息
    model.print_metrics()
    
    # 可视化特征图
    model.visualize_feature_maps(x)
    
    # 分析模型效率
    eff_stats = model.analyze_efficiency()
    print("\n模型效率分析:")
    print(f"总参数量: {eff_stats['total_params']:,}")
    print(f"总FLOPs: {eff_stats['total_flops']:,}")
    print(f"总内存占用: {eff_stats['total_memory']:.2f}MB")
    print("\n卷积层效率:")
    for name, stats in eff_stats['conv_layers'].items():
        print(f"{name}:")
        print(f"  参数量: {stats['params']:,}")
        print(f"  FLOPs: {stats['flops']:,}")
        print(f"  内存占用: {stats['memory_mb']:.2f}MB")
    print("\n全连接层效率:")
    for name, stats in eff_stats['fc_layers'].items():
        print(f"{name}:")
        print(f"  参数量: {stats['params']:,}")
        print(f"  FLOPs: {stats['flops']:,}")
        print(f"  内存占用: {stats['memory_mb']:.2f}MB")

if __name__ == '__main__':
    test()
