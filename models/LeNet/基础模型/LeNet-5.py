import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

"""
LeNet-5: 卷积神经网络的开山之作
============================================

历史背景：
----------
LeNet-5是由Yann LeCun等人在1998年提出的经典卷积神经网络架构，
是第一个成功应用于商业系统的卷积神经网络，主要用于手写数字识别。
它奠定了现代CNN的基础架构，对深度学习的发展产生了深远影响。

架构特点：
----------
1. 使用两个卷积层和三个全连接层
2. 原始设计使用Sigmoid激活函数和平均池化
3. 采用局部感受野和权值共享机制
4. 结构简洁但特征提取能力强

模型结构：
----------
1. 输入层: 32x32 灰度图像
2. 卷积层1: 6个特征图，5x5卷积核，步长1 (1x32x32 -> 6x28x28)
3. 平均池化层1: 2x2，步长2 (6x28x28 -> 6x14x14)
4. 卷积层2: 16个特征图，5x5卷积核，步长1 (6x14x14 -> 16x10x10)
5. 平均池化层2: 2x2，步长2 (16x10x10 -> 16x5x5)
6. 全连接层1: 120个神经元 (16*5*5 -> 120)
7. 全连接层2: 84个神经元 (120 -> 84)
8. 输出层: 10个类别 (84 -> 10)

学习要点：
---------
1. CNN的基本架构设计
2. 卷积与池化的作用
3. 特征提取的层次性
4. 参数共享机制
5. 激活函数的选择
"""

class OriginalLeNet5(nn.Module):
    """
    原始LeNet-5实现 (1998)
    输入: 1x32x32 的灰度图像
    """
    def __init__(self):
        super(OriginalLeNet5, self).__init__()
        # 第一个卷积层 (1x32x32 -> 6x28x28)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        # 第二个卷积层 (6x14x14 -> 16x10x10)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        # 第一个全连接层 (16*5*5 -> 120)
        self.fc1 = nn.Linear(16*5*5, 120)
        # 第二个全连接层 (120 -> 84)
        self.fc2 = nn.Linear(120, 84)
        # 输出层 (84 -> 10)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 第一个卷积层 + sigmoid激活 + 平均池化
        x = F.avg_pool2d(torch.sigmoid(self.conv1(x)), 2)
        # 第二个卷积层 + sigmoid激活 + 平均池化
        x = F.avg_pool2d(torch.sigmoid(self.conv2(x)), 2)
        # 展平
        x = x.view(-1, 16*5*5)
        # 全连接层 + sigmoid激活
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        # 输出层
        x = self.fc3(x)
        return x

class ModernLeNet5(nn.Module):
    """
    现代化的LeNet-5实现
    输入: 1x32x32 的灰度图像或 3x32x32 的彩色图像
    改进：
    1. 使用ReLU替代Sigmoid
    2. 使用最大池化替代平均池化
    3. 添加Batch Normalization
    4. 添加Dropout
    5. 支持彩色图像输入
    """
    def __init__(self, in_channels=1, num_classes=10):
        super(ModernLeNet5, self).__init__()
        
        self.features = nn.Sequential(
            # 第一个卷积块 (in_channels x 32x32 -> 6x28x28 -> 6x14x14)
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25),
            
            # 第二个卷积块 (6x14x14 -> 16x10x10 -> 16x5x5)
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25)
        )
        
        self.classifier = nn.Sequential(
            # 第一个全连接层
            nn.Linear(16*5*5, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            # 第二个全连接层
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            # 输出层
            nn.Linear(84, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16*5*5)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    """
    训练模型
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    """
    测试模型
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

def visualize_kernels(model, layer_idx=0):
    """
    可视化卷积核
    """
    if isinstance(model, ModernLeNet5):
        conv_layer = model.features[layer_idx]
    else:
        conv_layer = model.conv1 if layer_idx == 0 else model.conv2
    
    kernels = conv_layer.weight.detach().cpu()
    n_kernels = kernels.size(0)
    
    fig, axes = plt.subplots(1, n_kernels, figsize=(n_kernels*2, 2))
    for i in range(n_kernels):
        kernel = kernels[i, 0]
        axes[i].imshow(kernel, cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def visualize_feature_maps(model, image, layer_names=None):
    """
    可视化特征图
    """
    if layer_names is None:
        layer_names = ['conv1', 'conv2']
    
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # 注册钩子
    hooks = []
    for name, layer in model.named_modules():
        if name in layer_names:
            hooks.append(layer.register_forward_hook(get_activation(name)))
    
    # 前向传播
    with torch.no_grad():
        model(image.unsqueeze(0))
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 可视化
    for name, activation in activations.items():
        n_features = activation.size(1)
        fig, axes = plt.subplots(1, n_features, figsize=(n_features*2, 2))
        for i in range(n_features):
            feature_map = activation[0, i]
            axes[i].imshow(feature_map, cmap='gray')
            axes[i].axis('off')
        plt.suptitle(f'Feature maps from {name}')
        plt.tight_layout()
        plt.show()

def example_usage():
    """
    示例用法
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 创建模型
    model = ModernLeNet5().to(device)
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
    
    # 可视化
    sample_image, _ = next(iter(test_loader))
    visualize_kernels(model)
    visualize_feature_maps(model, sample_image[0])

if __name__ == '__main__':
    example_usage()

'''
这个实现包含了两个类：

OriginalLeNet5:
    完全按照1998年原始论文实现
    使用Sigmoid激活函数
    使用平均池化
    仅支持灰度图像输入
    固定的网络结构

ModernLeNet5:
    现代化改进版本
    使用ReLU激活函数提高性能
    使用最大池化代替平均池化
    添加Batch Normalization提高训练稳定性
    添加Dropout防止过拟合
    使用Kaiming初始化
    支持彩色图像输入（可配置输入通道数）
    可配置输出类别数

使用方法示例：

# 创建原始LeNet-5模型(灰度图像输入)
original_model = OriginalLeNet5()

# 创建现代LeNet-5模型(默认灰度图像输入)
modern_model = ModernLeNet5()

# 创建现代LeNet-5模型(彩色图像输入,100个类别)
modern_model_rgb = ModernLeNet5(in_channels=3, num_classes=100)
'''