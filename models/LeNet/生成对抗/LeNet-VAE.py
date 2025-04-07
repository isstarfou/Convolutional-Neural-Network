import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

"""
LeNet-VAE：变分自编码器
=====================

核心特点：
----------
1. 变分自编码器架构
2. 基于LeNet结构
3. 潜在空间建模
4. 图像重构能力
5. 特征提取能力

实现原理：
----------
1. 编码器将图像映射到潜在空间
2. 解码器从潜在空间重构图像
3. 变分推理优化
4. 重参数化技巧
5. 潜在空间采样

评估指标：
----------
1. 重构损失
2. KL散度
3. 潜在空间质量
4. 生成图像质量
5. 特征提取能力
"""

class Encoder(nn.Module):
    """编码器网络"""
    def __init__(self, latent_dim=20):
        super(Encoder, self).__init__()
        
        # 特征提取网络
        self.features = nn.Sequential(
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
        
        # 潜在空间映射
        self.fc_mu = nn.Linear(8 * 5 * 5, latent_dim)
        self.fc_var = nn.Linear(8 * 5 * 5, latent_dim)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

class Decoder(nn.Module):
    """解码器网络"""
    def __init__(self, latent_dim=20):
        super(Decoder, self).__init__()
        
        # 全连接层将潜在变量转换为特征图
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 8 * 5 * 5),
            nn.BatchNorm1d(8 * 5 * 5),
            nn.ReLU(inplace=True)
        )
        
        # 反卷积层重构图像
        self.deconv = nn.Sequential(
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
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 8, 5, 5)
        x = self.deconv(x)
        return x

class LeNetVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(LeNetVAE, self).__init__()
        
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
        # 评估指标
        self.metrics = {
            'reconstruction_loss': [],
            'kl_loss': [],
            'total_loss': [],
            'latent_space_quality': 0
        }
    
    def reparameterize(self, mu, log_var):
        """重参数化技巧"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # 编码
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        
        # 解码
        x_recon = self.decoder(z)
        
        return x_recon, mu, log_var
    
    def generate_images(self, num_images=16):
        """从潜在空间生成图像"""
        z = torch.randn(num_images, self.encoder.fc_mu.out_features)
        with torch.no_grad():
            fake_images = self.decoder(z)
        return fake_images
    
    def update_metrics(self, recon_loss, kl_loss, total_loss):
        """更新评估指标"""
        self.metrics['reconstruction_loss'].append(recon_loss)
        self.metrics['kl_loss'].append(kl_loss)
        self.metrics['total_loss'].append(total_loss)
    
    def print_metrics(self):
        """打印评估指标"""
        print("\n模型评估指标:")
        print(f"重构损失: {self.metrics['reconstruction_loss'][-1]:.4f}")
        print(f"KL散度: {self.metrics['kl_loss'][-1]:.4f}")
        print(f"总损失: {self.metrics['total_loss'][-1]:.4f}")
    
    def visualize_training(self):
        """可视化训练过程"""
        plt.figure(figsize=(15, 5))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['reconstruction_loss'], label='Reconstruction Loss')
        plt.plot(self.metrics['kl_loss'], label='KL Loss')
        plt.title('Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制总损失曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics['total_loss'], label='Total Loss')
        plt.title('Total Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def visualize_reconstruction(self, x):
        """可视化重构结果"""
        with torch.no_grad():
            x_recon, _, _ = self(x)
        
        plt.figure(figsize=(10, 5))
        
        # 显示原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(x[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        plt.title('Original Image')
        plt.axis('off')
        
        # 显示重构图像
        plt.subplot(1, 2, 2)
        plt.imshow(x_recon[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        plt.title('Reconstructed Image')
        plt.axis('off')
        
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
    model = LeNetVAE()
    
    # 创建随机输入
    x = torch.randn(1, 3, 32, 32)
    
    # 前向传播
    x_recon, mu, log_var = model(x)
    
    # 打印输出形状
    print(f"输入形状: {x.shape}")
    print(f"重构图像形状: {x_recon.shape}")
    print(f"均值形状: {mu.shape}")
    print(f"方差形状: {log_var.shape}")
    
    # 更新评估指标
    model.update_metrics(0.1, 0.05, 0.15)
    model.print_metrics()
    
    # 可视化重构结果
    model.visualize_reconstruction(x)
    
    # 可视化生成的图像
    model.visualize_generated_images()

if __name__ == '__main__':
    test() 