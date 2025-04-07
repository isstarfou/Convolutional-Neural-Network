import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

"""
LeNet-Residual: 残差连接的LeNet变体
============================================

历史背景：
----------
LeNet-Residual是LeNet系列中引入残差连接的变体，通过跳跃连接解决深层网络中的梯度消失问题。
这种设计灵感来源于ResNet的成功经验，将残差学习的思想应用到LeNet架构中。

架构特点：
----------
1. 引入残差连接，允许梯度直接传播
2. 使用批量归一化提高训练稳定性
3. 采用ReLU激活函数
4. 使用最大池化提取显著特征
5. 添加Dropout防止过拟合

模型结构：
----------
1. 输入层: 32x32 灰度图像
2. 残差块1: Conv1(32) → BN → ReLU → Conv2(32) → BN → ReLU
3. 残差块2: Conv3(64) → BN → ReLU → Conv4(64) → BN → ReLU
4. 全连接层1: 512个神经元
5. 全连接层2: 256个神经元
6. 输出层: 10个类别

学习要点：
---------
1. 残差连接的作用
2. 批量归一化的效果
3. 梯度消失问题的解决
4. 网络深度的选择
"""

class ResidualBlock(nn.Module):
    """残差块实现"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class LeNetResidual(nn.Module):
    """LeNet-Residual模型原始实现"""
    def __init__(self):
        super(LeNetResidual, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.resblock1 = ResidualBlock(32, 32)
        self.resblock2 = ResidualBlock(32, 64, stride=2)
        
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ModernLeNetResidual(nn.Module):
    """LeNet-Residual的现代化实现"""
    def __init__(self):
        super(ModernLeNetResidual, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.25)
        
        self.resblock1 = ResidualBlock(32, 32)
        self.resblock2 = ResidualBlock(32, 64, stride=2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 10)
        
        self._initialize_weights()
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.dropout2(x)
        
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.bn2(self.fc1(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn3(self.fc2(x)))
        x = self.dropout4(x)
        x = self.fc3(x)
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

# 可视化函数
def visualize_kernels(model, layer_idx=0):
    """可视化模型的卷积核"""
    if isinstance(model, LeNetResidual) or isinstance(model, ModernLeNetResidual):
        modules = list(model.modules())
        for i, module in enumerate(modules):
            if isinstance(module, nn.Conv2d):
                if layer_idx == 0:
                    weights = module.weight.data.cpu().numpy()
                    fig, axs = plt.subplots(1, min(8, weights.shape[0]), figsize=(15, 3))
                    for j in range(min(8, weights.shape[0])):
                        if min(8, weights.shape[0]) == 1:
                            ax = axs
                        else:
                            ax = axs[j]
                        ax.imshow(weights[j, 0], cmap='gray')
                        ax.axis('off')
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
    if isinstance(model, LeNetResidual):
        handles = [
            model.conv1.register_forward_hook(get_activation('conv1')),
            model.resblock1.conv1.register_forward_hook(get_activation('resblock1_conv1')),
            model.resblock2.conv1.register_forward_hook(get_activation('resblock2_conv1'))
        ]
    else:  # ModernLeNetResidual
        if layer_names is None:
            layer_names = ['conv1', 'resblock1_conv1', 'resblock2_conv1']
        handles = []
        for name in layer_names:
            if name == 'conv1':
                layer = model.conv1
            elif name == 'resblock1_conv1':
                layer = model.resblock1.conv1
            elif name == 'resblock2_conv1':
                layer = model.resblock2.conv1
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
    original_model = LeNetResidual().to(device)
    modern_model = ModernLeNetResidual().to(device)
    
    # 训练原始模型
    print("Training original LeNet-Residual...")
    optimizer = optim.SGD(original_model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(1, 6):
        train(original_model, device, train_loader, optimizer, epoch)
        test(original_model, device, test_loader)
    
    # 训练现代模型
    print("\nTraining modern LeNet-Residual...")
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
LeNet-Residual 特点解析：
----------------
1. 残差连接设计：
   - 跳跃连接解决梯度消失
   - 允许梯度直接传播
   - 支持更深的网络结构

2. 现代优化技术：
   - 批量归一化提高训练稳定性
   - Dropout防止过拟合
   - ReLU激活函数加速训练

3. 结构特点：
   - 残差块设计
   - 通道数逐步增加
   - 特征图尺寸逐步减小

4. 性能优势：
   - 训练更稳定
   - 收敛速度更快
   - 特征提取能力更强
"""

# 如果直接运行此脚本，则执行示例使用
if __name__ == '__main__':
    example_usage() 