import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

"""
LeNet-GAN：基于LeNet的生成对抗网络
==============================

核心特点：
----------
1. 生成器基于LeNet结构
2. 判别器采用LeNet变体
3. 对抗训练机制
4. 图像生成能力
5. 特征提取能力

实现原理：
----------
1. 生成器将随机噪声转换为图像
2. 判别器区分真实和生成图像
3. 对抗训练优化生成器和判别器
4. 使用LeNet结构进行特征提取
5. 结合GAN和LeNet的优势

评估指标：
----------
1. 生成图像质量
2. 判别器准确率
3. 训练稳定性
4. 特征提取能力
5. 模型收敛性
"""

class Generator(nn.Module):
    """生成器网络"""
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        
        # 全连接层将噪声转换为特征图
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 16 * 5 * 5),
            nn.BatchNorm1d(16 * 5 * 5),
            nn.ReLU(inplace=True)
        )
        
        # 反卷积层生成图像
        self.deconv = nn.Sequential(
            # 16x5x5 -> 8x10x10
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            
            # 8x10x10 -> 4x20x20
            nn.ConvTranspose2d(8, 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            
            # 4x20x20 -> 3x32x32
            nn.ConvTranspose2d(4, 3, kernel_size=5, stride=2, padding=2),
            nn.Tanh()
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 16, 5, 5)
        x = self.deconv(x)
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

class LeNetGAN(nn.Module):
    def __init__(self, latent_dim=100):
        super(LeNetGAN, self).__init__()
        
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()
        
        # 评估指标
        self.metrics = {
            'g_loss': [],
            'd_loss': [],
            'd_real_acc': [],
            'd_fake_acc': [],
            'generation_quality': 0
        }
    
    def generate_images(self, num_images=16):
        """生成图像"""
        z = torch.randn(num_images, self.generator.fc[0].in_features)
        with torch.no_grad():
            fake_images = self.generator(z)
        return fake_images
    
    def update_metrics(self, g_loss, d_loss, d_real_acc, d_fake_acc):
        """更新评估指标"""
        self.metrics['g_loss'].append(g_loss)
        self.metrics['d_loss'].append(d_loss)
        self.metrics['d_real_acc'].append(d_real_acc)
        self.metrics['d_fake_acc'].append(d_fake_acc)
    
    def print_metrics(self):
        """打印评估指标"""
        print("\n模型评估指标:")
        print(f"生成器损失: {self.metrics['g_loss'][-1]:.4f}")
        print(f"判别器损失: {self.metrics['d_loss'][-1]:.4f}")
        print(f"判别器真实图像准确率: {self.metrics['d_real_acc'][-1]:.2f}%")
        print(f"判别器生成图像准确率: {self.metrics['d_fake_acc'][-1]:.2f}%")
    
    def visualize_training(self):
        """可视化训练过程"""
        plt.figure(figsize=(15, 5))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['g_loss'], label='Generator Loss')
        plt.plot(self.metrics['d_loss'], label='Discriminator Loss')
        plt.title('Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics['d_real_acc'], label='Real Accuracy')
        plt.plot(self.metrics['d_fake_acc'], label='Fake Accuracy')
        plt.title('Discriminator Accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def visualize_generated_images(self, num_images=16):
        """可视化生成的图像"""
        fake_images = self.generate_images(num_images)
        
        plt.figure(figsize=(10, 10))
        for i in range(num_images):
            plt.subplot(4, 4, i+1)
            plt.imshow(fake_images[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

def test():
    """测试函数"""
    # 创建模型
    model = LeNetGAN()
    
    # 创建随机输入
    z = torch.randn(1, 100)
    real_images = torch.randn(1, 3, 32, 32)
    
    # 生成图像
    fake_images = model.generator(z)
    
    # 判别器预测
    real_pred = model.discriminator(real_images)
    fake_pred = model.discriminator(fake_images)
    
    # 打印输出形状
    print(f"噪声输入形状: {z.shape}")
    print(f"生成图像形状: {fake_images.shape}")
    print(f"真实图像判别结果: {real_pred.shape}")
    print(f"生成图像判别结果: {fake_pred.shape}")
    
    # 更新评估指标
    model.update_metrics(0.5, 0.5, 0.8, 0.7)
    model.print_metrics()
    
    # 可视化生成的图像
    model.visualize_generated_images()

if __name__ == '__main__':
    test() 