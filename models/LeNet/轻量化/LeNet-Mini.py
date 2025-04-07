import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

"""
LeNet-Mini：迷你版LeNet模型
=========================

核心特点：
----------
1. 极简网络结构
2. 超轻量级设计
3. 快速推理速度
4. 低内存占用
5. 适合嵌入式设备

实现原理：
----------
1. 减少卷积层通道数
2. 简化网络结构
3. 优化参数量
4. 保持基础特征提取能力
5. 适合小规模数据集

评估指标：
----------
1. 模型大小
2. 推理速度
3. 内存占用
4. 准确率
5. 计算复杂度
"""

class LeNetMini(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNetMini, self).__init__()
        
        # 特征提取网络
        self.features = nn.Sequential(
            # 第一层卷积 (3x32x32 -> 4x28x28)
            nn.Conv2d(3, 4, kernel_size=5),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x14x14
            
            # 第二层卷积 (4x14x14 -> 8x10x10)
            nn.Conv2d(4, 8, kernel_size=5),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 8x5x5
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            # 全连接层 (8*5*5 -> 32)
            nn.Linear(8*5*5, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            # 输出层 (32 -> num_classes)
            nn.Linear(32, num_classes)
        )
        
        # 评估指标
        self.metrics = {
            'params': self._count_parameters(),
            'size_mb': self._calculate_model_size(),
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
    
    def _calculate_model_size(self):
        """计算模型大小(MB)"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1024 / 1024
    
    def update_metrics(self, accuracy, inference_time, memory_usage):
        """更新评估指标"""
        self.metrics['accuracy'] = accuracy
        self.metrics['inference_time'] = inference_time
        self.metrics['memory_usage'] = memory_usage
    
    def print_metrics(self):
        """打印评估指标"""
        print("\n模型评估指标:")
        print(f"参数量: {self.metrics['params']:,}")
        print(f"模型大小: {self.metrics['size_mb']:.2f}MB")
        print(f"准确率: {self.metrics['accuracy']:.2f}%")
        print(f"推理时间: {self.metrics['inference_time']:.4f}秒")
        print(f"内存占用: {self.metrics['memory_usage']:.2f}MB")
    
    def visualize_architecture(self):
        """可视化网络架构"""
        # 创建随机输入
        x = torch.randn(1, 3, 32, 32)
        
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

def test():
    """测试函数"""
    # 创建模型
    model = LeNetMini(num_classes=10)
    
    # 创建随机输入
    x = torch.randn(1, 3, 32, 32)
    
    # 前向传播
    output = model(x)
    
    # 打印输出形状
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 打印模型信息
    model.print_metrics()
    
    # 可视化网络架构
    model.visualize_architecture()

if __name__ == '__main__':
    test() 