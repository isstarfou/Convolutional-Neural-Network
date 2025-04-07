import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

"""
LeNet-4: 卷积神经网络早期架构的重要变体
============================================

历史背景：
----------
LeNet-4是由Yann LeCun等人在1998年提出的LeNet系列中的一个变体，是LeNet-5的前身。
在手写数字识别领域，LeNet-4相比于LeNet-1/LeNet-2等早期版本有显著改进，
但比LeNet-5结构稍简单，是理解CNN基础架构演变的重要模型。

架构特点：
----------
1. 相比LeNet-5少一个卷积层，总共使用2个卷积层
2. 使用平均池化层而非最大池化层
3. 原始设计使用Sigmoid或Tanh激活函数
4. 全连接层设计比LeNet-5更简洁

模型结构：
----------
1. 输入层: 32x32 灰度图像
2. 卷积层1: 4个特征图，5x5卷积核，步长1 (1x32x32 -> 4x28x28)
3. 平均池化层1: 2x2，步长2 (4x28x28 -> 4x14x14)
4. 卷积层2: 16个特征图，5x5卷积核，步长1 (4x14x14 -> 16x10x10)
5. 平均池化层2: 2x2，步长2 (16x10x10 -> 16x5x5)
6. 全连接层1: 120个神经元 (16*5*5 -> 120)
7. 全连接层2: 输出层，10个类别 (120 -> 10)

学习要点：
---------
1. 早期CNN设计哲学
2. 卷积与池化的基本原理
3. 参数共享与权重稀疏性
4. 激活函数选择对模型性能的影响
"""

class LeNet4(nn.Module):
    """
    LeNet-4模型原始实现
    使用了Sigmoid激活函数和平均池化，完全复现了原始论文设计
    """
    def __init__(self):
        super(LeNet4, self).__init__()
        # 第一个卷积层 (1x32x32 -> 4x28x28)
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, stride=1)
        # 第二个卷积层 (4x14x14 -> 16x10x10)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=5, stride=1)
        # 第一个全连接层 (16*5*5 -> 120)
        self.fc1 = nn.Linear(16*5*5, 120)
        # 输出层 (120 -> 10)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        # 第一个卷积层 + sigmoid激活 + 平均池化
        x = F.avg_pool2d(torch.sigmoid(self.conv1(x)), 2)
        # 第二个卷积层 + sigmoid激活 + 平均池化
        x = F.avg_pool2d(torch.sigmoid(self.conv2(x)), 2)
        # 展平
        x = x.view(-1, 16*5*5)
        # 全连接层 + sigmoid激活
        x = torch.sigmoid(self.fc1(x))
        # 输出层
        x = self.fc2(x)
        return x


class ModernLeNet4(nn.Module):
    """
    LeNet-4的现代化实现
    改进：
    1. 使用ReLU激活函数替代Sigmoid
    2. 添加批归一化(BatchNorm)提高训练稳定性
    3. 添加Dropout防止过拟合
    4. 支持彩色图像输入(可配置)
    5. 使用更现代的权重初始化方法
    """
    def __init__(self, in_channels=1, num_classes=10):
        super(ModernLeNet4, self).__init__()
        
        self.features = nn.Sequential(
            # 第一个卷积块 (in_channels x 32x32 -> 4x28x28 -> 4x14x14)
            nn.Conv2d(in_channels, 4, kernel_size=5),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.2),
            
            # 第二个卷积块 (4x14x14 -> 16x10x10 -> 16x5x5)
            nn.Conv2d(4, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.2)
        )
        
        self.classifier = nn.Sequential(
            # 全连接层
            nn.Linear(16*5*5, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            # 输出层
            nn.Linear(120, num_classes)
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


# 训练函数
def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


# 测试函数
def test(model, device, test_loader):
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
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy


# 可视化卷积核和特征图的函数
def visualize_kernels(model, layer_idx=0):
    """可视化模型的卷积核"""
    if isinstance(model, LeNet4) or isinstance(model, ModernLeNet4):
        modules = list(model.modules())
        for i, module in enumerate(modules):
            if isinstance(module, nn.Conv2d):
                if layer_idx == 0:
                    weights = module.weight.data.cpu().numpy()
                    fig, axs = plt.subplots(1, weights.shape[0], figsize=(15, 3))
                    for j in range(weights.shape[0]):
                        axs[j].imshow(weights[j, 0], cmap='gray')
                        axs[j].axis('off')
                    plt.suptitle(f'Visualizing Conv Layer {i} Kernels')
                    plt.tight_layout()
                    plt.show()
                    return
                layer_idx -= 1


def visualize_feature_maps(model, image, layer_names=None):
    """可视化模型的特征图"""
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # 注册钩子
    if isinstance(model, LeNet4):
        handles = [
            model.conv1.register_forward_hook(get_activation('conv1')),
            model.conv2.register_forward_hook(get_activation('conv2'))
        ]
    else:  # ModernLeNet4
        if layer_names is None:
            layer_names = ['features.0', 'features.6']  # 卷积层索引
        handles = []
        for name in layer_names:
            layer = model
            for part in name.split('.'):
                layer = getattr(layer, part)
            handles.append(layer.register_forward_hook(get_activation(name)))
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(image)
    
    # 移除钩子
    for handle in handles:
        handle.remove()
    
    # 可视化
    for name, feat in activation.items():
        features = feat[0].cpu().numpy()
        fig, axs = plt.subplots(1, min(8, features.shape[0]), figsize=(15, 3))
        for i in range(min(8, features.shape[0])):
            if min(8, features.shape[0]) == 1:
                ax = axs
            else:
                ax = axs[i]
            ax.imshow(features[i], cmap='viridis')
            ax.axis('off')
        plt.suptitle(f'Feature Maps of {name}')
        plt.tight_layout()
        plt.show()


# 示例：如何使用模型
def example_usage():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 准备数据集
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    # 创建模型
    original_model = LeNet4().to(device)
    modern_model = ModernLeNet4().to(device)
    
    # 训练原始模型
    print("Training original LeNet-4...")
    optimizer = optim.SGD(original_model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(1, 6):
        train(original_model, device, train_loader, optimizer, epoch)
        test(original_model, device, test_loader)
    
    # 训练现代模型
    print("\nTraining modern LeNet-4...")
    optimizer = optim.Adam(modern_model.parameters(), lr=0.001)
    for epoch in range(1, 6):
        train(modern_model, device, train_loader, optimizer, epoch)
        test(modern_model, device, test_loader)
    
    # 可视化卷积核和特征图
    print("\nVisualizing kernels and feature maps...")
    visualize_kernels(original_model)
    visualize_kernels(modern_model)
    
    # 可视化特征图
    sample_data, _ = next(iter(test_loader))
    sample_image = sample_data[0:1].to(device)
    visualize_feature_maps(original_model, sample_image)
    visualize_feature_maps(modern_model, sample_image)


"""
LeNet-4 与 LeNet-5 比较：
-----------------------
1. 结构差异：
   - LeNet-4有2个卷积层和2个全连接层
   - LeNet-5有3个卷积层和2个全连接层

2. 特征图数量：
   - LeNet-4: 4 -> 16
   - LeNet-5: 6 -> 16

3. 参数数量：
   - LeNet-4: 约35K参数
   - LeNet-5: 约60K参数

4. 性能对比：
   - LeNet-4在MNIST上达到约98.5%的准确率
   - LeNet-5在MNIST上达到约99.0%的准确率

设计理念解析：
-----------
1. 特征提取能力：
   LeNet-4设计更简洁，参数更少，在简单任务上效果接近LeNet-5，是资源受限场景下的良好选择。

2. 计算复杂度：
   LeNet-4计算复杂度显著低于LeNet-5，适合早期计算资源有限的环境，体现了早期CNN在效率与准确性之间的权衡。

3. 历史意义：
   作为LeNet-5的前身，LeNet-4是深度学习发展史上的重要里程碑，帮助研究者理解如何构建有效的卷积网络。
"""


# 如果直接运行此脚本，则执行示例使用
if __name__ == '__main__':
    example_usage()
