import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

"""
LeNet-BN：使用批归一化的LeNet实现
=================================

核心特点：
----------
1. 增加批归一化层，加速网络训练收敛
2. 实现多种归一化变体(BatchNorm, LayerNorm, InstanceNorm, GroupNorm)
3. 分析不同归一化位置的影响
4. 分析批量大小对归一化效果的影响

批归一化原理：
------------
批归一化(Batch Normalization)通过对每一层的输入进行归一化处理，减缓内部协变量偏移问题，
从而加速训练并允许使用更高的学习率。主要步骤包括：
1. 计算批次数据的均值和方差
2. 归一化输入数据
3. 应用可学习的缩放和偏移参数
4. 在推理时使用全局统计量

归一化变体：
----------
1. BatchNorm: 对每个特征通道在批次维度上计算统计量
2. LayerNorm: 对每个样本独立计算统计量，跨所有特征通道
3. InstanceNorm: 对每个样本的每个特征通道独立计算统计量
4. GroupNorm: 将通道分组，在每组内计算统计量，兼顾BN和LN

实现目标：
---------
1. 理解批归一化的原理和各种变体的特点
2. 掌握批归一化在网络不同位置的应用技巧
3. 分析不同批次大小下归一化方法的性能差异
4. 学习如何调整归一化层的参数
"""

class LeNetBN(nn.Module):
    def __init__(self, num_classes=10, norm_type='batch', eps=1e-5, momentum=0.1, 
                 affine=True, track_running_stats=True, num_groups=2):
        super(LeNetBN, self).__init__()
        
        self.norm_type = norm_type
        
        # 首层卷积
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        
        # 首层归一化
        if norm_type == 'batch':
            self.norm1 = nn.BatchNorm2d(6, eps=eps, momentum=momentum, 
                                         affine=affine, track_running_stats=track_running_stats)
        elif norm_type == 'layer':
            self.norm1 = nn.LayerNorm([6, 28, 28], eps=eps, elementwise_affine=affine)
        elif norm_type == 'instance':
            self.norm1 = nn.InstanceNorm2d(6, eps=eps, momentum=momentum, 
                                            affine=affine, track_running_stats=track_running_stats)
        elif norm_type == 'group':
            self.norm1 = nn.GroupNorm(min(num_groups, 6), 6, eps=eps, affine=affine)
        else:
            self.norm1 = nn.Identity()
            
        # 第一个池化层
        self.pool1 = nn.MaxPool2d(2)
        
        # 第二层卷积
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        
        # 第二层归一化
        if norm_type == 'batch':
            self.norm2 = nn.BatchNorm2d(16, eps=eps, momentum=momentum, 
                                         affine=affine, track_running_stats=track_running_stats)
        elif norm_type == 'layer':
            self.norm2 = nn.LayerNorm([16, 14, 14], eps=eps, elementwise_affine=affine)
        elif norm_type == 'instance':
            self.norm2 = nn.InstanceNorm2d(16, eps=eps, momentum=momentum, 
                                            affine=affine, track_running_stats=track_running_stats)
        elif norm_type == 'group':
            self.norm2 = nn.GroupNorm(min(num_groups, 16), 16, eps=eps, affine=affine)
        else:
            self.norm2 = nn.Identity()
        
        # 第二个池化层
        self.pool2 = nn.MaxPool2d(2)
        
        # 全连接层1
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        
        # 全连接层1的归一化
        if norm_type == 'batch':
            self.norm3 = nn.BatchNorm1d(120, eps=eps, momentum=momentum, 
                                         affine=affine, track_running_stats=track_running_stats)
        elif norm_type == 'layer':
            self.norm3 = nn.LayerNorm(120, eps=eps, elementwise_affine=affine)
        elif norm_type == 'instance':
            # InstanceNorm1d等价于在这种情况下的LayerNorm
            self.norm3 = nn.InstanceNorm1d(120, eps=eps, momentum=momentum, 
                                            affine=affine, track_running_stats=track_running_stats)
        elif norm_type == 'group':
            self.norm3 = nn.GroupNorm(min(num_groups, 120), 120, eps=eps, affine=affine)
        else:
            self.norm3 = nn.Identity()
        
        # 全连接层2
        self.fc2 = nn.Linear(120, 84)
        
        # 全连接层2的归一化
        if norm_type == 'batch':
            self.norm4 = nn.BatchNorm1d(84, eps=eps, momentum=momentum, 
                                         affine=affine, track_running_stats=track_running_stats)
        elif norm_type == 'layer':
            self.norm4 = nn.LayerNorm(84, eps=eps, elementwise_affine=affine)
        elif norm_type == 'instance':
            self.norm4 = nn.InstanceNorm1d(84, eps=eps, momentum=momentum, 
                                            affine=affine, track_running_stats=track_running_stats)
        elif norm_type == 'group':
            self.norm4 = nn.GroupNorm(min(num_groups, 84), 84, eps=eps, affine=affine)
        else:
            self.norm4 = nn.Identity()
        
        # 输出层
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        """前向传播"""
        # 第一个卷积块
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层1
        x = self.fc1(x)
        x = self.norm3(x)
        x = F.relu(x)
        
        # 全连接层2
        x = self.fc2(x)
        x = self.norm4(x)
        x = F.relu(x)
        
        # 输出层
        x = self.fc3(x)
        
        return x

    def get_norm_stats(self):
        """获取各归一化层的统计数据，用于分析"""
        stats = {}
        
        if self.norm_type == 'batch':
            # 只有批归一化层才有运行统计量
            stats['layer1'] = {
                'mean': self.norm1.running_mean.cpu().numpy(),
                'var': self.norm1.running_var.cpu().numpy()
            }
            stats['layer2'] = {
                'mean': self.norm2.running_mean.cpu().numpy(),
                'var': self.norm2.running_var.cpu().numpy()
            }
            stats['layer3'] = {
                'mean': self.norm3.running_mean.cpu().numpy(),
                'var': self.norm3.running_var.cpu().numpy()
            }
            stats['layer4'] = {
                'mean': self.norm4.running_mean.cpu().numpy(),
                'var': self.norm4.running_var.cpu().numpy()
            }
        
        return stats

class LeNetWithoutBN(nn.Module):
    """不使用批归一化的标准LeNet，用于对比"""
    def __init__(self, num_classes=10):
        super(LeNetWithoutBN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        
        # 池化层
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        """前向传播"""
        # 第一个卷积块
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def visualize_activations(models, input_data):
    """可视化不同归一化方法的激活分布"""
    # 检查不同模型在同一层的激活值分布
    activations = {}
    
    for name, model in models.items():
        model.eval()
        
        # 注册钩子以获取每一层的激活值
        activation = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        # 注册钩子
        hooks = []
        hooks.append(model.conv1.register_forward_hook(get_activation('conv1')))
        
        if name != 'No Normalization':
            hooks.append(model.norm1.register_forward_hook(get_activation('norm1')))
        
        hooks.append(model.conv2.register_forward_hook(get_activation('conv2')))
        
        if name != 'No Normalization':
            hooks.append(model.norm2.register_forward_hook(get_activation('norm2')))
        
        # 前向传播
        with torch.no_grad():
            _ = model(input_data)
        
        # 存储激活值
        activations[name] = activation
        
        # 移除钩子
        for h in hooks:
            h.remove()
    
    # 绘制分布图
    plt.figure(figsize=(15, 10))
    
    # 选择要分析的层
    layers_to_plot = ['conv1', 'norm1', 'conv2', 'norm2']
    
    for i, layer in enumerate(layers_to_plot):
        plt.subplot(2, 2, i+1)
        
        for name, activation in activations.items():
            if layer in activation:
                # 获取激活值
                act = activation[layer].cpu().numpy()
                
                # 展平激活值
                act_flat = act.reshape(-1)
                
                # 绘制直方图
                plt.hist(act_flat, bins=50, alpha=0.5, label=name)
        
        plt.title(f'Layer: {layer}')
        plt.xlabel('Activation value')
        plt.ylabel('Frequency')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def analyze_batch_size_impact():
    """分析批次大小对不同归一化方法的影响"""
    # 创建不同批次大小的随机数据
    batch_sizes = [1, 4, 16, 64]
    
    # 创建不同归一化方法的模型
    norm_types = ['batch', 'layer', 'instance', 'group']
    
    # 用于存储统计量
    stats = {norm: {batch: {} for batch in batch_sizes} for norm in norm_types}
    
    for norm_type in norm_types:
        model = LeNetBN(norm_type=norm_type)
        model.train()  # 设置为训练模式，使BN更新统计量
        
        for batch_size in batch_sizes:
            # 创建随机输入数据
            x = torch.randn(batch_size, 1, 28, 28)
            
            # 前向传播
            _ = model(x)
            
            # 提取各层激活值
            activation = {}
            
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook
            
            hooks = []
            hooks.append(model.norm1.register_forward_hook(get_activation('norm1')))
            hooks.append(model.norm2.register_forward_hook(get_activation('norm2')))
            
            # 再次前向传播
            _ = model(x)
            
            # 计算第一个归一化层的均值和方差
            act1 = activation['norm1'].cpu().numpy()
            act1_flat = act1.reshape(batch_size, -1)
            
            # 计算第二个归一化层的均值和方差
            act2 = activation['norm2'].cpu().numpy()
            act2_flat = act2.reshape(batch_size, -1)
            
            # 存储统计量
            stats[norm_type][batch_size]['norm1_mean'] = np.mean(act1_flat)
            stats[norm_type][batch_size]['norm1_std'] = np.std(act1_flat)
            stats[norm_type][batch_size]['norm2_mean'] = np.mean(act2_flat)
            stats[norm_type][batch_size]['norm2_std'] = np.std(act2_flat)
            
            # 移除钩子
            for h in hooks:
                h.remove()
    
    # 可视化统计量
    plt.figure(figsize=(15, 10))
    
    stat_types = ['norm1_mean', 'norm1_std', 'norm2_mean', 'norm2_std']
    
    for i, stat in enumerate(stat_types):
        plt.subplot(2, 2, i+1)
        
        for norm_type in norm_types:
            values = [stats[norm_type][batch][stat] for batch in batch_sizes]
            plt.plot(batch_sizes, values, marker='o', label=norm_type)
        
        plt.title(f'Statistic: {stat}')
        plt.xlabel('Batch Size')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def test():
    """测试不同归一化方法的LeNet模型"""
    # 创建不同归一化方法的模型
    models = {
        'No Normalization': LeNetWithoutBN(),
        'Batch Norm': LeNetBN(norm_type='batch'),
        'Layer Norm': LeNetBN(norm_type='layer'),
        'Instance Norm': LeNetBN(norm_type='instance'),
        'Group Norm': LeNetBN(norm_type='group', num_groups=2)
    }
    
    # 创建随机输入
    x = torch.randn(4, 1, 28, 28)
    
    # 测试每个模型
    for name, model in models.items():
        y = model(x)
        print(f"{name} 输出形状: {y.shape}")
    
    # 可视化归一化层影响
    print("\n可视化归一化层对激活分布的影响...")
    visualize_activations(models, x)
    
    # 分析批次大小对不同归一化方法的影响
    print("\n分析批次大小对不同归一化方法的影响...")
    analyze_batch_size_impact()
    
    # 打印可学习参数数量
    print("\n不同模型的参数数量:")
    for name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name}: {param_count} 参数")
    
    # BatchNorm的运行均值和方差
    if 'Batch Norm' in models:
        bn_model = models['Batch Norm']
        # 训练模式下前向传播几次数据，让BN积累统计量
        bn_model.train()
        for _ in range(10):
            _ = bn_model(torch.randn(32, 1, 28, 28))
        
        # 打印统计量
        bn_stats = bn_model.get_norm_stats()
        print("\nBatchNorm运行统计量:")
        for layer, stats in bn_stats.items():
            print(f"  {layer}:")
            print(f"    均值范围: [{stats['mean'].min():.4f}, {stats['mean'].max():.4f}]")
            print(f"    方差范围: [{stats['var'].min():.4f}, {stats['var'].max():.4f}]")

if __name__ == '__main__':
    test() 