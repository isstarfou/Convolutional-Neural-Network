import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import random

"""
CutMix增强版LeNet模型
==================

核心特点：
----------
1. CutMix数据增强
2. 区域裁剪混合
3. 标签平滑
4. 特征提取能力
5. 模型鲁棒性

实现原理：
----------
1. 随机区域裁剪
2. 图像块交换
3. 标签比例调整
4. 在线增强训练
5. 损失函数调整

评估指标：
----------
1. 训练准确率
2. 验证准确率
3. 混合效果评估
4. 模型鲁棒性
5. 泛化能力
"""

class CutMixLeNet(nn.Module):
    def __init__(self, num_classes=10, beta=1.0):
        super(CutMixLeNet, self).__init__()
        
        # CutMix参数
        self.beta = beta
        
        # 数据预处理
        self.transform = transforms.Compose([
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
            'cutmix_ratio': []
        }
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def rand_bbox(self, size, lam):
        """生成随机边界框"""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        
        # 随机选择中心点
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # 计算边界框坐标
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def cutmix_data(self, x, y):
        """执行CutMix增强"""
        # 生成混合比例
        lam = np.random.beta(self.beta, self.beta)
        
        # 随机打乱数据
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # 生成边界框
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        
        # 混合图像
        mixed_x = x.clone()
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # 调整混合比例
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        
        # 混合标签
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def cutmix_criterion(self, criterion, pred, y_a, y_b, lam):
        """计算CutMix损失"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def update_metrics(self, train_loss, train_acc, val_loss, val_acc, cutmix_ratio):
        """更新评估指标"""
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['cutmix_ratio'].append(cutmix_ratio)
    
    def print_metrics(self):
        """打印评估指标"""
        print("\n模型评估指标:")
        print(f"训练损失: {self.metrics['train_loss'][-1]:.4f}")
        print(f"训练准确率: {self.metrics['train_acc'][-1]:.4f}")
        print(f"验证损失: {self.metrics['val_loss'][-1]:.4f}")
        print(f"验证准确率: {self.metrics['val_acc'][-1]:.4f}")
        print(f"平均混合比例: {np.mean(self.metrics['cutmix_ratio']):.4f}")
    
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
        
        # 绘制混合比例分布
        plt.subplot(2, 2, 3)
        plt.hist(self.metrics['cutmix_ratio'], bins=20)
        plt.title('CutMix Ratio Distribution')
        plt.xlabel('Ratio')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_cutmix(self, x1, x2):
        """可视化CutMix效果"""
        # 转换为张量
        x1 = self.transform(x1)
        x2 = self.transform(x2)
        
        # 生成混合比例
        lam = np.random.beta(self.beta, self.beta)
        
        # 生成边界框
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x1.unsqueeze(0).size(), lam)
        
        # 混合图像
        mixed_x = x1.clone()
        mixed_x[:, bbx1:bbx2, bby1:bby2] = x2[:, bbx1:bbx2, bby1:bby2]
        
        plt.figure(figsize=(15, 5))
        
        # 显示原始图像1
        plt.subplot(1, 3, 1)
        plt.imshow(x1.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        plt.title('Image 1')
        plt.axis('off')
        
        # 显示混合图像
        plt.subplot(1, 3, 2)
        plt.imshow(mixed_x.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        plt.title(f'CutMix Image (λ={lam:.2f})')
        plt.axis('off')
        
        # 显示原始图像2
        plt.subplot(1, 3, 3)
        plt.imshow(x2.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        plt.title('Image 2')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def test():
    """测试函数"""
    # 创建模型
    model = CutMixLeNet()
    
    # 创建随机输入
    x = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, 10, (2,))
    
    # 执行CutMix
    mixed_x, y_a, y_b, lam = model.cutmix_data(x, y)
    print(f"输入形状: {x.shape}")
    print(f"混合后形状: {mixed_x.shape}")
    print(f"混合比例: {lam:.4f}")
    
    # 前向传播
    output = model(mixed_x)
    print(f"输出形状: {output.shape}")
    
    # 计算损失
    criterion = nn.CrossEntropyLoss()
    loss = model.cutmix_criterion(criterion, output, y_a, y_b, lam)
    print(f"混合损失: {loss.item():.4f}")
    
    # 更新评估指标
    model.update_metrics(0.5, 0.8, 0.4, 0.85, lam)
    model.print_metrics()
    
    # 可视化训练过程
    model.visualize_training()
    
    # 可视化CutMix效果
    x1 = torch.randn(3, 32, 32)
    x2 = torch.randn(3, 32, 32)
    model.visualize_cutmix(x1, x2)

if __name__ == '__main__':
    test() 