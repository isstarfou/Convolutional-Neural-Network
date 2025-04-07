import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

"""
LeNet-7: 深度卷积神经网络
============================================

历史背景：
----------
LeNet-7是LeNet系列中的深度版本，通过增加网络层数来提高特征提取能力。
相比LeNet-5，LeNet-7具有更深的网络结构和更强的特征表达能力。

架构特点：
----------
1. 7层网络结构，包含4个卷积层和3个全连接层
2. 使用批量归一化提高训练稳定性
3. 采用ReLU激活函数替代Sigmoid
4. 使用最大池化替代平均池化
5. 添加Dropout防止过拟合

模型结构：
----------
1. 输入层: 32x32 灰度图像
2. 卷积层1: 32个特征图，3x3卷积核
3. 卷积层2: 64个特征图，3x3卷积核
4. 卷积层3: 128个特征图，3x3卷积核
5. 卷积层4: 256个特征图，3x3卷积核
6. 全连接层1: 512个神经元
7. 全连接层2: 256个神经元
8. 输出层: 10个类别

学习要点：
---------
1. 深度网络的设计原则
2. 批量归一化的作用
3. Dropout的正则化效果
4. 网络深度与性能的关系
"""

class LeNet7(nn.Module):
    """LeNet-7模型原始实现"""
    def __init__(self):
        super(LeNet7, self).__init__()
        # 第一个卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        
        # 第二个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        # 全连接层
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        # 第一个卷积块
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        # 第二个卷积块
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        # 全连接层
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ModernLeNet7(nn.Module):
    """LeNet-7的现代化实现"""
    def __init__(self):
        super(ModernLeNet7, self).__init__()
        # 第一个卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # 第二个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # 全连接层
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 10)
        
        # 权重初始化
        self._initialize_weights()
        
    def forward(self, x):
        # 第一个卷积块
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 第二个卷积块
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 全连接层
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn6(self.fc2(x)))
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
    if isinstance(model, LeNet7) or isinstance(model, ModernLeNet7):
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
    if isinstance(model, LeNet7):
        handles = [
            model.conv1.register_forward_hook(get_activation('conv1')),
            model.conv2.register_forward_hook(get_activation('conv2')),
            model.conv3.register_forward_hook(get_activation('conv3')),
            model.conv4.register_forward_hook(get_activation('conv4'))
        ]
    else:  # ModernLeNet7
        if layer_names is None:
            layer_names = ['conv1', 'conv2', 'conv3', 'conv4']
        handles = []
        for name in layer_names:
            layer = getattr(model, name)
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
    original_model = LeNet7().to(device)
    modern_model = ModernLeNet7().to(device)
    
    # 训练原始模型
    print("Training original LeNet-7...")
    optimizer = optim.SGD(original_model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(1, 6):
        train(original_model, device, train_loader, optimizer, epoch)
        test(original_model, device, test_loader)
    
    # 训练现代模型
    print("\nTraining modern LeNet-7...")
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
LeNet-7 特点解析：
----------------
1. 深度网络设计：
   - 4个卷积层提供更强的特征提取能力
   - 3个全连接层增强分类能力
   - 网络深度与性能的平衡

2. 现代优化技术：
   - 批量归一化提高训练稳定性
   - Dropout防止过拟合
   - ReLU激活函数加速训练

3. 结构特点：
   - 使用最大池化提取显著特征
   - 通道数逐步增加，特征图尺寸逐步减小
   - 全连接层神经元数量逐步减少

4. 性能优势：
   - 特征提取能力强
   - 分类准确率高
   - 训练稳定性好
"""

# 如果直接运行此脚本，则执行示例使用
if __name__ == '__main__':
    example_usage()
