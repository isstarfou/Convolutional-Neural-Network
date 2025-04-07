import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

"""
基于元学习的小样本学习LeNet模型
===========================

核心特点：
----------
1. 支持小样本学习
2. 基于元学习的优化
3. 特征提取能力
4. 快速适应能力
5. 端到端训练

实现原理：
----------
1. 特征提取网络
2. 元学习优化器
3. 支持集和查询集处理
4. 梯度更新策略
5. 分类预测

评估指标：
----------
1. 支持集准确率
2. 查询集准确率
3. 适应速度
4. 模型鲁棒性
5. 泛化能力
"""

class MetaLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MetaLeNet, self).__init__()
        
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
        
        # 特征嵌入
        self.embedding = nn.Sequential(
            nn.Linear(8 * 5 * 5, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # 分类器
        self.classifier = nn.Linear(64, num_classes)
        
        # 评估指标
        self.metrics = {
            'support_acc': [],
            'query_acc': [],
            'adaptation_speed': [],
            'loss': []
        }
    
    def forward(self, support_x, support_y, query_x, num_steps=5):
        """
        前向传播
        Args:
            support_x: 支持集图像 [n_way * k_shot, C, H, W]
            support_y: 支持集标签 [n_way * k_shot]
            query_x: 查询集图像 [n_query, C, H, W]
            num_steps: 适应步数
        Returns:
            query_pred: 查询集预测 [n_query, n_way]
        """
        # 提取支持集特征
        support_features = self.features(support_x)
        support_features = support_features.view(support_features.size(0), -1)
        support_features = self.embedding(support_features)
        
        # 提取查询集特征
        query_features = self.features(query_x)
        query_features = query_features.view(query_features.size(0), -1)
        query_features = self.embedding(query_features)
        
        # 创建分类器副本用于元学习
        classifier = self.classifier
        fast_weights = list(classifier.parameters())
        
        # 元学习适应
        for _ in range(num_steps):
            # 计算支持集预测
            support_logits = classifier(support_features)
            support_loss = F.cross_entropy(support_logits, support_y)
            
            # 计算梯度
            grads = torch.autograd.grad(support_loss, fast_weights, create_graph=True)
            
            # 更新快速权重
            fast_weights = [w - 0.01 * g for w, g in zip(fast_weights, grads)]
            
            # 创建新的分类器
            classifier = nn.Linear(64, len(torch.unique(support_y)))
            classifier.load_state_dict({
                'weight': fast_weights[0],
                'bias': fast_weights[1]
            })
        
        # 计算查询集预测
        query_logits = classifier(query_features)
        
        return query_logits
    
    def update_metrics(self, support_acc, query_acc, adaptation_speed, loss):
        """更新评估指标"""
        self.metrics['support_acc'].append(support_acc)
        self.metrics['query_acc'].append(query_acc)
        self.metrics['adaptation_speed'].append(adaptation_speed)
        self.metrics['loss'].append(loss)
    
    def print_metrics(self):
        """打印评估指标"""
        print("\n模型评估指标:")
        print(f"支持集准确率: {self.metrics['support_acc'][-1]:.4f}")
        print(f"查询集准确率: {self.metrics['query_acc'][-1]:.4f}")
        print(f"适应速度: {self.metrics['adaptation_speed'][-1]:.4f}")
        print(f"损失: {self.metrics['loss'][-1]:.4f}")
    
    def visualize_training(self):
        """可视化训练过程"""
        plt.figure(figsize=(15, 10))
        
        # 绘制准确率曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics['support_acc'], label='Support Acc')
        plt.plot(self.metrics['query_acc'], label='Query Acc')
        plt.title('Accuracy Curves')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # 绘制适应速度曲线
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics['adaptation_speed'])
        plt.title('Adaptation Speed')
        plt.xlabel('Episode')
        plt.ylabel('Speed')
        
        # 绘制损失曲线
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics['loss'])
        plt.title('Loss Curve')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_adaptation(self, support_x, support_y, query_x, num_steps=5):
        """可视化适应过程"""
        # 提取特征
        support_features = self.features(support_x)
        support_features = support_features.view(support_features.size(0), -1)
        support_features = self.embedding(support_features)
        
        query_features = self.features(query_x)
        query_features = query_features.view(query_features.size(0), -1)
        query_features = self.embedding(query_features)
        
        # 创建分类器副本
        classifier = self.classifier
        fast_weights = list(classifier.parameters())
        
        # 记录适应过程中的准确率
        support_accs = []
        query_accs = []
        
        # 适应过程
        for step in range(num_steps):
            # 计算支持集预测
            support_logits = classifier(support_features)
            support_loss = F.cross_entropy(support_logits, support_y)
            
            # 计算梯度
            grads = torch.autograd.grad(support_loss, fast_weights, create_graph=True)
            
            # 更新快速权重
            fast_weights = [w - 0.01 * g for w, g in zip(fast_weights, grads)]
            
            # 创建新的分类器
            classifier = nn.Linear(64, len(torch.unique(support_y)))
            classifier.load_state_dict({
                'weight': fast_weights[0],
                'bias': fast_weights[1]
            })
            
            # 计算准确率
            with torch.no_grad():
                support_pred = classifier(support_features).argmax(dim=1)
                query_pred = classifier(query_features).argmax(dim=1)
                
                support_acc = (support_pred == support_y).float().mean()
                query_acc = (query_pred == support_y[:len(query_pred)]).float().mean()
                
                support_accs.append(support_acc.item())
                query_accs.append(query_acc.item())
        
        # 绘制适应过程
        plt.figure(figsize=(10, 6))
        plt.plot(range(num_steps), support_accs, label='Support Acc')
        plt.plot(range(num_steps), query_accs, label='Query Acc')
        plt.title('Adaptation Process')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

def test():
    """测试函数"""
    # 创建模型
    model = MetaLeNet()
    
    # 创建随机输入
    n_way = 5
    k_shot = 5
    n_query = 10
    
    support_x = torch.randn(n_way * k_shot, 3, 32, 32)
    support_y = torch.arange(n_way).repeat(k_shot)
    query_x = torch.randn(n_query, 3, 32, 32)
    
    # 前向传播
    output = model(support_x, support_y, query_x)
    print(f"支持集形状: {support_x.shape}")
    print(f"查询集形状: {query_x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 更新评估指标
    model.update_metrics(0.8, 0.7, 0.6, 0.5)
    model.print_metrics()
    
    # 可视化训练过程
    model.visualize_training()
    
    # 可视化适应过程
    model.visualize_adaptation(support_x, support_y, query_x)

if __name__ == '__main__':
    test() 