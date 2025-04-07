import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

"""
DenseLeNet: 密集连接卷积神经网络
============================================

历史背景：
----------
DenseLeNet是基于DenseNet思想的LeNet变体，通过密集连接机制增强特征重用，
提高网络的信息流动效率。这种设计可以有效缓解梯度消失问题，同时减少参数量。

架构特点：
----------
1. 采用密集连接机制，每一层的输入都来自前面所有层的输出
2. 使用瓶颈层(bottleneck)减少计算量
3. 采用过渡层(transition)控制特征图数量
4. 使用批量归一化和ReLU激活函数

模型结构：
----------
1. 输入层: 32x32 灰度图像
2. 初始卷积层: 16个特征图，3x3卷积核
3. 密集块1: 4层，每层输出16个特征图
4. 过渡层1: 1x1卷积 + 2x2平均池化
5. 密集块2: 4层，每层输出16个特征图
6. 过渡层2: 1x1卷积 + 2x2平均池化
7. 全局平均池化
8. 全连接层: 输出层，10个类别

学习要点：
---------
1. 密集连接机制的优势
2. 特征重用的重要性
3. 瓶颈层和过渡层的设计
4. 网络深度与参数效率的平衡
"""

class DenseLayer(nn.Module):
    """密集块中的单层结构"""
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    """密集块结构"""
    def __init__(self, in_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i*growth_rate, growth_rate))
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    """过渡层结构"""
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)
        
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out

class DenseLeNet(nn.Module):
    """DenseLeNet模型实现"""
    def __init__(self, growth_rate=16, num_layers=4, compression=0.5):
        super(DenseLeNet, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        
        # 第一个密集块
        self.dense1 = DenseBlock(16, num_layers, growth_rate)
        in_channels = 16 + num_layers * growth_rate
        out_channels = int(in_channels * compression)
        self.trans1 = TransitionLayer(in_channels, out_channels)
        
        # 第二个密集块
        self.dense2 = DenseBlock(out_channels, num_layers, growth_rate)
        in_channels = out_channels + num_layers * growth_rate
        out_channels = int(in_channels * compression)
        self.trans2 = TransitionLayer(in_channels, out_channels)
        
        # 全局平均池化和全连接层
        self.bn = nn.BatchNorm2d(out_channels)
        self.fc = nn.Linear(out_channels, 10)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class ModernDenseLeNet(nn.Module):
    """DenseLeNet的现代化实现"""
    def __init__(self, growth_rate=16, num_layers=4, compression=0.5):
        super(ModernDenseLeNet, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 第一个密集块
        self.dense1 = DenseBlock(16, num_layers, growth_rate)
        in_channels = 16 + num_layers * growth_rate
        out_channels = int(in_channels * compression)
        self.trans1 = TransitionLayer(in_channels, out_channels)
        
        # 第二个密集块
        self.dense2 = DenseBlock(out_channels, num_layers, growth_rate)
        in_channels = out_channels + num_layers * growth_rate
        out_channels = int(in_channels * compression)
        self.trans2 = TransitionLayer(in_channels, out_channels)
        
        # 全局平均池化和全连接层
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(out_channels, 10)
        
        # 权重初始化
        self._initialize_weights()
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
    if isinstance(model, DenseLeNet) or isinstance(model, ModernDenseLeNet):
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
    if isinstance(model, DenseLeNet):
        handles = [
            model.conv1.register_forward_hook(get_activation('conv1')),
            model.dense1.register_forward_hook(get_activation('dense1')),
            model.dense2.register_forward_hook(get_activation('dense2'))
        ]
    else:  # ModernDenseLeNet
        if layer_names is None:
            layer_names = ['conv1', 'dense1', 'dense2']
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
    original_model = DenseLeNet().to(device)
    modern_model = ModernDenseLeNet().to(device)
    
    # 训练原始模型
    print("Training original DenseLeNet...")
    optimizer = optim.SGD(original_model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(1, 6):
        train(original_model, device, train_loader, optimizer, epoch)
        test(original_model, device, test_loader)
    
    # 训练现代模型
    print("\nTraining modern DenseLeNet...")
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
DenseLeNet 特点解析：
-------------------
1. 密集连接机制：
   - 每一层的输入都来自前面所有层的输出
   - 增强了特征重用，提高了参数效率
   - 缓解了梯度消失问题

2. 瓶颈层设计：
   - 使用1x1卷积减少特征图数量
   - 降低计算复杂度
   - 提高模型效率

3. 过渡层设计：
   - 控制特征图数量
   - 通过压缩因子减少参数量
   - 使用平均池化降低特征图尺寸

4. 性能优势：
   - 参数量少，计算效率高
   - 特征重用能力强
   - 训练稳定性好
"""

# 如果直接运行此脚本，则执行示例使用
if __name__ == '__main__':
    example_usage()
