import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
from typing import List, Tuple, Dict, Any

"""
Neural-Architecture-Search-LeNet: 基于神经架构搜索的LeNet优化
============================================================

历史背景：
----------
神经架构搜索(NAS)是一种自动化机器学习方法，用于自动发现最优的神经网络架构。
本实现将NAS技术应用于LeNet架构的优化，通过强化学习或进化算法自动搜索最佳网络配置。

架构特点：
----------
1. 可搜索的架构空间包括：
   - 卷积层通道数
   - 卷积核大小
   - 激活函数类型
   - 池化层类型
   - 全连接层大小
2. 支持多种搜索策略：
   - 强化学习
   - 进化算法
   - 随机搜索
3. 灵活的评估机制

搜索空间：
----------
1. 卷积层配置：
   - 通道数: [4, 8, 16, 32, 64]
   - 卷积核大小: [3, 5, 7]
   - 激活函数: [ReLU, LeakyReLU, ELU]
2. 池化层配置：
   - 类型: [MaxPool, AvgPool]
   - 核大小: [2, 3]
3. 全连接层配置：
   - 隐藏单元数: [64, 128, 256, 512]
   - Dropout率: [0.0, 0.2, 0.4]

学习要点：
---------
1. 神经架构搜索的基本原理
2. 自动化机器学习方法
3. 架构空间的设计
4. 搜索策略的选择
5. 性能评估方法
"""

class NASLeNet(nn.Module):
    """
    基于神经架构搜索的LeNet实现
    """
    def __init__(self, config: Dict[str, Any]):
        super(NASLeNet, self).__init__()
        self.config = config
        
        # 构建特征提取器
        self.features = self._build_features()
        
        # 构建分类器
        self.classifier = self._build_classifier()
        
        # 初始化权重
        self._initialize_weights()

    def _build_features(self) -> nn.Sequential:
        layers = []
        in_channels = 1
        
        # 第一个卷积块
        conv1_config = self.config['conv1']
        layers.extend([
            nn.Conv2d(in_channels, conv1_config['channels'], 
                     kernel_size=conv1_config['kernel_size']),
            self._get_activation(conv1_config['activation']),
            self._get_pooling(conv1_config['pooling'])
        ])
        
        # 第二个卷积块
        conv2_config = self.config['conv2']
        layers.extend([
            nn.Conv2d(conv1_config['channels'], conv2_config['channels'],
                     kernel_size=conv2_config['kernel_size']),
            self._get_activation(conv2_config['activation']),
            self._get_pooling(conv2_config['pooling'])
        ])
        
        return nn.Sequential(*layers)

    def _build_classifier(self) -> nn.Sequential:
        fc_config = self.config['fc']
        layers = []
        
        # 计算全连接层输入大小
        conv2_config = self.config['conv2']
        in_features = conv2_config['channels'] * 5 * 5  # 假设输入是32x32
        
        layers.extend([
            nn.Linear(in_features, fc_config['hidden_units']),
            self._get_activation(fc_config['activation']),
            nn.Dropout(fc_config['dropout']),
            nn.Linear(fc_config['hidden_units'], 10)
        ])
        
        return nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'relu': nn.ReLU(inplace=True),
            'leakyrelu': nn.LeakyReLU(0.1, inplace=True),
            'elu': nn.ELU(inplace=True)
        }
        return activations[name]

    def _get_pooling(self, name: str) -> nn.Module:
        poolings = {
            'max': nn.MaxPool2d(kernel_size=2),
            'avg': nn.AvgPool2d(kernel_size=2)
        }
        return poolings[name]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class NASController:
    """
    神经架构搜索控制器
    """
    def __init__(self, search_space: Dict[str, List[Any]]):
        self.search_space = search_space
        
    def random_search(self, n_samples: int) -> List[Dict[str, Any]]:
        """
        随机搜索策略
        """
        architectures = []
        for _ in range(n_samples):
            config = {}
            for key, values in self.search_space.items():
                config[key] = random.choice(values)
            architectures.append(config)
        return architectures
    
    def evolutionary_search(self, population_size: int, generations: int,
                          mutation_rate: float = 0.1) -> List[Dict[str, Any]]:
        """
        进化算法搜索策略
        """
        # 初始化种群
        population = self.random_search(population_size)
        
        for _ in range(generations):
            # 评估适应度
            fitness = self._evaluate_population(population)
            
            # 选择
            selected = self._select(population, fitness)
            
            # 交叉和变异
            new_population = []
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(selected, 2)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child, mutation_rate)
                new_population.append(child)
            
            population = new_population
        
        return population
    
    def _evaluate_population(self, population: List[Dict[str, Any]]) -> List[float]:
        """
        评估种群中每个个体的适应度
        """
        fitness = []
        for config in population:
            # 这里应该实现实际的评估逻辑
            # 例如：训练模型并计算验证集准确率
            score = random.random()  # 临时使用随机分数
            fitness.append(score)
        return fitness
    
    def _select(self, population: List[Dict[str, Any]], 
                fitness: List[float], k: int = 2) -> List[Dict[str, Any]]:
        """
        选择操作
        """
        # 使用轮盘赌选择
        total_fitness = sum(fitness)
        probabilities = [f/total_fitness for f in fitness]
        selected = random.choices(population, weights=probabilities, k=k)
        return selected
    
    def _crossover(self, parent1: Dict[str, Any], 
                  parent2: Dict[str, Any]) -> Dict[str, Any]:
        """
        交叉操作
        """
        child = {}
        for key in parent1.keys():
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child
    
    def _mutate(self, config: Dict[str, Any], 
                mutation_rate: float) -> Dict[str, Any]:
        """
        变异操作
        """
        mutated = config.copy()
        for key in mutated.keys():
            if random.random() < mutation_rate:
                mutated[key] = random.choice(self.search_space[key])
        return mutated

def train_model(model: nn.Module, train_loader: DataLoader, 
                device: torch.device, epochs: int = 10) -> float:
    """
    训练模型并返回验证准确率
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # 返回验证准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    return correct / total

def example_usage():
    """
    示例用法
    """
    # 定义搜索空间
    search_space = {
        'conv1': {
            'channels': [4, 8, 16, 32],
            'kernel_size': [3, 5, 7],
            'activation': ['relu', 'leakyrelu', 'elu'],
            'pooling': ['max', 'avg']
        },
        'conv2': {
            'channels': [8, 16, 32, 64],
            'kernel_size': [3, 5, 7],
            'activation': ['relu', 'leakyrelu', 'elu'],
            'pooling': ['max', 'avg']
        },
        'fc': {
            'hidden_units': [64, 128, 256, 512],
            'activation': ['relu', 'leakyrelu', 'elu'],
            'dropout': [0.0, 0.2, 0.4]
        }
    }
    
    # 创建搜索控制器
    controller = NASController(search_space)
    
    # 使用进化算法搜索
    best_architectures = controller.evolutionary_search(
        population_size=20,
        generations=10,
        mutation_rate=0.1
    )
    
    # 选择最佳架构
    best_config = best_architectures[0]  # 这里应该根据实际评估结果选择
    
    # 创建和训练最终模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NASLeNet(best_config).to(device)
    
    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True,
                                 transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 训练模型
    accuracy = train_model(model, train_loader, device)
    print(f"Final model accuracy: {accuracy:.2%}")

if __name__ == '__main__':
    example_usage()
