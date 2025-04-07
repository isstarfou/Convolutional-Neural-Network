import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import random

"""
增强版LeNet模型
=============

核心特点：
----------
1. 内置数据增强功能
2. 多种增强策略
3. 自适应增强强度
4. 在线增强训练
5. 特征提取能力

实现原理：
----------
1. 随机裁剪和翻转
2. 颜色抖动
3. 随机擦除
4. 混合增强
5. 自适应增强强度

评估指标：
----------
1. 训练准确率
2. 验证准确率
3. 增强效果评估
4. 模型鲁棒性
5. 泛化能力
"""

class RandomErasing:
    """随机擦除增强"""
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, img):
        if random.random() < self.p:
            # 获取图像尺寸
            img_h, img_w = img.shape[1], img.shape[2]
            
            # 计算擦除区域面积
            area = img_h * img_w
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            # 计算擦除区域尺寸
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if h < img_h and w < img_w:
                # 随机选择擦除位置
                top = random.randint(0, img_h - h)
                left = random.randint(0, img_w - w)
                
                # 执行擦除
                img[:, top:top+h, left:left+w] = torch.randn(3, h, w)
        
        return img

class AugmentedLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AugmentedLeNet, self).__init__()
        
        # 数据增强变换
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            RandomErasing()
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
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
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(8 * 5 * 5, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
        # 评估指标
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'augmentation_strength': []
        }
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def augment_image(self, x, is_training=True):
        """增强图像"""
        if is_training:
            return self.train_transform(x)
        else:
            return self.test_transform(x)
    
    def update_metrics(self, train_loss, train_acc, val_loss, val_acc, aug_strength):
        """更新评估指标"""
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['augmentation_strength'].append(aug_strength)
    
    def print_metrics(self):
        """打印评估指标"""
        print("\n模型评估指标:")
        print(f"训练损失: {self.metrics['train_loss'][-1]:.4f}")
        print(f"训练准确率: {self.metrics['train_acc'][-1]:.4f}")
        print(f"验证损失: {self.metrics['val_loss'][-1]:.4f}")
        print(f"验证准确率: {self.metrics['val_acc'][-1]:.4f}")
        print(f"增强强度: {self.metrics['augmentation_strength'][-1]:.4f}")
    
    def visualize_training(self):
        """可视化训练过程"""
        plt.figure(figsize=(15, 10))
        
        # 绘制损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics['train_loss'], label='Train Loss')
        plt.plot(self.metrics['val_loss'], label='Val Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制准确率曲线
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics['train_acc'], label='Train Acc')
        plt.plot(self.metrics['val_acc'], label='Val Acc')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # 绘制增强强度曲线
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics['augmentation_strength'])
        plt.title('Augmentation Strength')
        plt.xlabel('Epoch')
        plt.ylabel('Strength')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_augmentation(self, x):
        """可视化增强效果"""
        # 原始图像
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(x.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        plt.title('Original Image')
        plt.axis('off')
        
        # 增强后的图像1
        plt.subplot(1, 3, 2)
        aug_x1 = self.augment_image(x)
        plt.imshow(aug_x1.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        plt.title('Augmented Image 1')
        plt.axis('off')
        
        # 增强后的图像2
        plt.subplot(1, 3, 3)
        aug_x2 = self.augment_image(x)
        plt.imshow(aug_x2.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        plt.title('Augmented Image 2')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def test():
    """测试函数"""
    # 创建模型
    model = AugmentedLeNet()
    
    # 创建随机输入
    x = torch.randn(3, 32, 32)
    
    # 增强图像
    aug_x = model.augment_image(x)
    print(f"输入形状: {x.shape}")
    print(f"增强后形状: {aug_x.shape}")
    
    # 前向传播
    output = model(aug_x.unsqueeze(0))
    print(f"输出形状: {output.shape}")
    
    # 更新评估指标
    model.update_metrics(0.5, 0.8, 0.4, 0.85, 0.3)
    model.print_metrics()
    
    # 可视化增强效果
    model.visualize_augmentation(x)
    
    # 可视化训练过程
    model.visualize_training()

if __name__ == '__main__':
    test() 