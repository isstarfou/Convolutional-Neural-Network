import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

"""
LeNet-CycleGAN：循环生成对抗网络
=============================

核心特点：
----------
1. 循环一致性生成对抗网络
2. 基于LeNet结构
3. 双向图像转换
4. 循环一致性损失
5. 特征提取能力

实现原理：
----------
1. 两个生成器实现双向转换
2. 两个判别器分别判别两个域
3. 循环一致性约束
4. 对抗训练机制
5. 特征提取和重构

评估指标：
----------
1. 生成图像质量
2. 循环一致性损失
3. 对抗损失
4. 特征提取能力
5. 转换效果评估
"""

class Generator(nn.Module):
    """生成器网络"""
    def __init__(self):
        super(Generator, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            # 3x32x32 -> 4x28x28
            nn.Conv2d(3, 4, kernel_size=5),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x14x14
            
            # 4x14x14 -> 8x10x10
            nn.Conv2d(4, 8, kernel_size=5),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 8x5x5
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            # 8x5x5 -> 4x10x10
            nn.ConvTranspose2d(8, 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            
            # 4x10x10 -> 3x20x20
            nn.ConvTranspose2d(4, 3, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            
            # 3x20x20 -> 3x32x32
            nn.ConvTranspose2d(3, 3, kernel_size=5, stride=2, padding=2),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Discriminator(nn.Module):
    """判别器网络"""
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # 特征提取网络
        self.features = nn.Sequential(
            # 3x32x32 -> 4x28x28
            nn.Conv2d(3, 4, kernel_size=5),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),  # 4x14x14
            
            # 4x14x14 -> 8x10x10
            nn.Conv2d(4, 8, kernel_size=5),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2)  # 8x5x5
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(8 * 5 * 5, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class LeNetCycleGAN(nn.Module):
    def __init__(self):
        super(LeNetCycleGAN, self).__init__()
        
        # 生成器和判别器
        self.G_AB = Generator()  # A域到B域的生成器
        self.G_BA = Generator()  # B域到A域的生成器
        self.D_A = Discriminator()  # A域判别器
        self.D_B = Discriminator()  # B域判别器
        
        # 评估指标
        self.metrics = {
            'g_loss': [],
            'd_loss': [],
            'cycle_loss': [],
            'identity_loss': [],
            'total_loss': []
        }
    
    def forward(self, x_A, x_B):
        """前向传播"""
        # 生成图像
        fake_B = self.G_AB(x_A)
        fake_A = self.G_BA(x_B)
        
        # 循环重构
        cycle_A = self.G_BA(fake_B)
        cycle_B = self.G_AB(fake_A)
        
        # 身份映射
        identity_A = self.G_BA(x_A)
        identity_B = self.G_AB(x_B)
        
        return {
            'fake_B': fake_B,
            'fake_A': fake_A,
            'cycle_A': cycle_A,
            'cycle_B': cycle_B,
            'identity_A': identity_A,
            'identity_B': identity_B
        }
    
    def update_metrics(self, g_loss, d_loss, cycle_loss, identity_loss, total_loss):
        """更新评估指标"""
        self.metrics['g_loss'].append(g_loss)
        self.metrics['d_loss'].append(d_loss)
        self.metrics['cycle_loss'].append(cycle_loss)
        self.metrics['identity_loss'].append(identity_loss)
        self.metrics['total_loss'].append(total_loss)
    
    def print_metrics(self):
        """打印评估指标"""
        print("\n模型评估指标:")
        print(f"生成器损失: {self.metrics['g_loss'][-1]:.4f}")
        print(f"判别器损失: {self.metrics['d_loss'][-1]:.4f}")
        print(f"循环一致性损失: {self.metrics['cycle_loss'][-1]:.4f}")
        print(f"身份映射损失: {self.metrics['identity_loss'][-1]:.4f}")
        print(f"总损失: {self.metrics['total_loss'][-1]:.4f}")
    
    def visualize_training(self):
        """可视化训练过程"""
        plt.figure(figsize=(15, 10))
        
        # 绘制损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics['g_loss'], label='Generator Loss')
        plt.plot(self.metrics['d_loss'], label='Discriminator Loss')
        plt.title('Adversarial Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制循环一致性损失
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics['cycle_loss'], label='Cycle Loss')
        plt.title('Cycle Consistency Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制身份映射损失
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics['identity_loss'], label='Identity Loss')
        plt.title('Identity Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制总损失
        plt.subplot(2, 2, 4)
        plt.plot(self.metrics['total_loss'], label='Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def visualize_translation(self, x_A, x_B):
        """可视化图像转换结果"""
        with torch.no_grad():
            outputs = self(x_A, x_B)
        
        plt.figure(figsize=(15, 10))
        
        # 显示A域图像
        plt.subplot(2, 3, 1)
        plt.imshow(x_A[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        plt.title('Domain A')
        plt.axis('off')
        
        # 显示A到B的转换
        plt.subplot(2, 3, 2)
        plt.imshow(outputs['fake_B'][0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        plt.title('A -> B')
        plt.axis('off')
        
        # 显示循环重构的A
        plt.subplot(2, 3, 3)
        plt.imshow(outputs['cycle_A'][0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        plt.title('Cycle A')
        plt.axis('off')
        
        # 显示B域图像
        plt.subplot(2, 3, 4)
        plt.imshow(x_B[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        plt.title('Domain B')
        plt.axis('off')
        
        # 显示B到A的转换
        plt.subplot(2, 3, 5)
        plt.imshow(outputs['fake_A'][0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        plt.title('B -> A')
        plt.axis('off')
        
        # 显示循环重构的B
        plt.subplot(2, 3, 6)
        plt.imshow(outputs['cycle_B'][0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        plt.title('Cycle B')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def test():
    """测试函数"""
    # 创建模型
    model = LeNetCycleGAN()
    
    # 创建随机输入
    x_A = torch.randn(1, 3, 32, 32)
    x_B = torch.randn(1, 3, 32, 32)
    
    # 前向传播
    outputs = model(x_A, x_B)
    
    # 打印输出形状
    print(f"A域输入形状: {x_A.shape}")
    print(f"B域输入形状: {x_B.shape}")
    print(f"A到B转换形状: {outputs['fake_B'].shape}")
    print(f"B到A转换形状: {outputs['fake_A'].shape}")
    print(f"循环重构A形状: {outputs['cycle_A'].shape}")
    print(f"循环重构B形状: {outputs['cycle_B'].shape}")
    
    # 更新评估指标
    model.update_metrics(0.5, 0.5, 0.3, 0.2, 1.5)
    model.print_metrics()
    
    # 可视化转换结果
    model.visualize_translation(x_A, x_B)

if __name__ == '__main__':
    test() 