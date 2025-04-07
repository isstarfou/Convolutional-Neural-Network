import torch
import torch.nn as nn
import torch.nn.functional as F

"""
LeNet-AdamW：使用AdamW优化器的LeNet变体
======================================

核心特点：
----------
1. 集成了AdamW优化算法，结合了Adam的自适应学习率与权重衰减的优势
2. 权重衰减正确解耦，避免了Adam优化器中权重衰减实现的问题
3. 有效缓解过拟合，提高泛化性能
4. 适用于大多数深度学习任务，特别是参数量大的模型

优化器原理：
----------
AdamW解决了Adam优化器中权重衰减实现不当的问题。在传统Adam中，
权重衰减通过L2正则化实现，会受到自适应学习率的影响。
AdamW将权重衰减直接应用于权重更新步骤，与动量和自适应学习率解耦。

算法步骤：
----------
1. 计算梯度及其一阶和二阶矩估计
2. 计算偏差修正后的矩估计
3. 计算自适应学习率
4. 直接在权重上应用权重衰减
5. 更新参数

主要超参数：
----------
- 学习率(lr)：控制参数更新步长
- Beta1(β1)：一阶矩估计的指数衰减率
- Beta2(β2)：二阶矩估计的指数衰减率
- 权重衰减(weight_decay)：控制正则化强度
- Epsilon(ε)：防止除零错误的小常数
"""

class AdamWLeNet(nn.Module):
    def __init__(self, num_classes=10, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        super(AdamWLeNet, self).__init__()
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        
        # 池化层
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # AdamW优化器状态
        self.m = {}  # 一阶矩估计
        self.v = {}  # 二阶矩估计
        self.t = 0   # 时间步
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.m[name] = torch.zeros_like(param)
                self.v[name] = torch.zeros_like(param)
                
    def update_weights(self, gradients, lr):
        """使用AdamW更新权重"""
        self.t += 1
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    grad = gradients[name]
                    
                    # 更新偏置修正的一阶和二阶矩估计
                    self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
                    self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad * grad
                    
                    # 计算偏差修正
                    m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[name] / (1 - self.beta2 ** self.t)
                    
                    # AdamW：先应用L2正则化/权重衰减，再更新参数
                    param.data = param.data - lr * self.weight_decay * param.data
                    param.data = param.data - lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        
    def forward(self, x):
        # 第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 第二层
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def test():
    model = AdamWLeNet()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(f"输出形状: {y.shape}")
    
    # 模拟梯度更新
    gradients = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradients[name] = torch.randn_like(param)
    
    # 更新权重
    model.update_weights(gradients, lr=0.001)
    print("已使用AdamW优化器更新权重")

if __name__ == '__main__':
    test()
