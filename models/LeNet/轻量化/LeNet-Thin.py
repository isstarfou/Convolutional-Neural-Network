import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

"""
LeNet-Thin：瘦身版LeNet模型
========================

核心特点：
----------
1. 通道数优化
2. 参数量精简
3. 计算效率高
4. 内存占用低
5. 推理速度快

实现原理：
----------
1. 减少各层通道数
2. 优化网络结构
3. 保持特征提取能力
4. 平衡性能和效率
5. 适合资源受限场景

评估指标：
----------
1. 参数量
2. 计算量
3. 推理速度
4. 准确率
5. 内存效率
"""

class LeNetThin(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNetThin, self).__init__()
        
        # 特征提取网络
        self.features = nn.Sequential(
            # 第一层卷积 (3x32x32 -> 3x28x28)
            nn.Conv2d(3, 3, kernel_size=5),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 3x14x14
            
            # 第二层卷积 (3x14x14 -> 6x10x10)
            nn.Conv2d(3, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 6x5x5
            
            # 第三层卷积 (6x5x5 -> 12x1x1)
            nn.Conv2d(6, 12, kernel_size=5),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            # 全连接层 (12 -> 24)
            nn.Linear(12, 24),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            # 输出层 (24 -> num_classes)
            nn.Linear(24, num_classes)
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
            # 显示第一个通道的特征图
            plt.imshow(act[0, 0].cpu().numpy(), cmap='viridis')
            plt.title(f'{name}\n{act.shape[1]} channels')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def analyze_memory_usage(self):
        """分析内存使用情况"""
        memory_stats = {
            'conv_layers': {},
            'fc_layers': {},
            'total_params': 0,
            'total_memory': 0
        }
        
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                params = sum(p.numel() for p in module.parameters())
                memory = params * 4 / 1024 / 1024  # 转换为MB
                
                if isinstance(module, nn.Conv2d):
                    memory_stats['conv_layers'][name] = {
                        'params': params,
                        'memory_mb': memory
                    }
                else:
                    memory_stats['fc_layers'][name] = {
                        'params': params,
                        'memory_mb': memory
                    }
                
                memory_stats['total_params'] += params
                memory_stats['total_memory'] += memory
        
        return memory_stats

def test():
    """测试函数"""
    # 创建模型
    model = LeNetThin(num_classes=10)
    
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
    
    # 分析内存使用
    memory_stats = model.analyze_memory_usage()
    print("\n内存使用分析:")
    print(f"总参数量: {memory_stats['total_params']:,}")
    print(f"总内存占用: {memory_stats['total_memory']:.2f}MB")
    print("\n卷积层内存使用:")
    for name, stats in memory_stats['conv_layers'].items():
        print(f"{name}: {stats['params']:,} 参数, {stats['memory_mb']:.2f}MB")
    print("\n全连接层内存使用:")
    for name, stats in memory_stats['fc_layers'].items():
        print(f"{name}: {stats['params']:,} 参数, {stats['memory_mb']:.2f}MB")

if __name__ == '__main__':
    test() 