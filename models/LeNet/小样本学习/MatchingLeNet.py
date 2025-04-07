import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

"""
基于匹配网络的小样本学习LeNet模型
============================

核心特点：
----------
1. 支持小样本学习
2. 基于匹配的度量学习
3. 特征提取能力
4. 自适应度量
5. 端到端训练

实现原理：
----------
1. 特征提取网络
2. 支持集和查询集处理
3. 余弦相似度计算
4. 注意力机制
5. 分类预测

评估指标：
----------
1. 支持集准确率
2. 查询集准确率
3. 特征相似度
4. 模型鲁棒性
5. 泛化能力
"""

class MatchingLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MatchingLeNet, self).__init__()
        
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
        
        # 评估指标
        self.metrics = {
            'support_acc': [],
            'query_acc': [],
            'feature_similarity': [],
            'loss': []
        }
    
    def forward(self, support_x, support_y, query_x):
        """
        前向传播
        Args:
            support_x: 支持集图像 [n_way * k_shot, C, H, W]
            support_y: 支持集标签 [n_way * k_shot]
            query_x: 查询集图像 [n_query, C, H, W]
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
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(
            query_features.unsqueeze(1),
            support_features.unsqueeze(0),
            dim=2
        )
        
        # 计算注意力权重
        attention = F.softmax(similarity, dim=1)
        
        # 计算类别原型
        n_way = len(torch.unique(support_y))
        prototypes = torch.zeros(n_way, support_features.size(1)).to(support_features.device)
        for i in range(n_way):
            mask = (support_y == i)
            prototypes[i] = support_features[mask].mean(0)
        
        # 计算查询集与类原型的相似度
        query_similarity = F.cosine_similarity(
            query_features.unsqueeze(1),
            prototypes.unsqueeze(0),
            dim=2
        )
        
        return query_similarity
    
    def update_metrics(self, support_acc, query_acc, feature_similarity, loss):
        """更新评估指标"""
        self.metrics['support_acc'].append(support_acc)
        self.metrics['query_acc'].append(query_acc)
        self.metrics['feature_similarity'].append(feature_similarity)
        self.metrics['loss'].append(loss)
    
    def print_metrics(self):
        """打印评估指标"""
        print("\n模型评估指标:")
        print(f"支持集准确率: {self.metrics['support_acc'][-1]:.4f}")
        print(f"查询集准确率: {self.metrics['query_acc'][-1]:.4f}")
        print(f"特征相似度: {self.metrics['feature_similarity'][-1]:.4f}")
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
        
        # 绘制特征相似度曲线
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics['feature_similarity'])
        plt.title('Feature Similarity')
        plt.xlabel('Episode')
        plt.ylabel('Similarity')
        
        # 绘制损失曲线
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics['loss'])
        plt.title('Loss Curve')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_features(self, support_x, support_y, query_x):
        """可视化特征分布"""
        # 提取特征
        support_features = self.features(support_x)
        support_features = support_features.view(support_features.size(0), -1)
        support_features = self.embedding(support_features)
        
        query_features = self.features(query_x)
        query_features = query_features.view(query_features.size(0), -1)
        query_features = self.embedding(query_features)
        
        # 使用PCA降维
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        all_features = torch.cat([support_features, query_features], dim=0)
        reduced_features = pca.fit_transform(all_features.cpu().detach().numpy())
        
        # 绘制特征分布
        plt.figure(figsize=(10, 8))
        
        # 绘制支持集特征
        n_way = len(torch.unique(support_y))
        for i in range(n_way):
            mask = (support_y == i)
            plt.scatter(
                reduced_features[mask, 0],
                reduced_features[mask, 1],
                label=f'Support Class {i}',
                marker='o'
            )
        
        # 绘制查询集特征
        plt.scatter(
            reduced_features[support_features.size(0):, 0],
            reduced_features[support_features.size(0):, 1],
            label='Query',
            marker='x'
        )
        
        plt.title('Feature Distribution')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.legend()
        plt.show()

def test():
    """测试函数"""
    # 创建模型
    model = MatchingLeNet()
    
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
    
    # 可视化特征分布
    model.visualize_features(support_x, support_y, query_x)

if __name__ == '__main__':
    test() 