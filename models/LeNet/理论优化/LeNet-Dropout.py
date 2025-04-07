import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

"""
LeNet-Dropout：使用Dropout正则化技术的LeNet实现
=============================================

核心特点：
----------
1. 添加Dropout层防止过拟合
2. 实现多种Dropout变体（标准、空间、Alpha）
3. 提供Dropout率的自动调整机制
4. 支持蒙特卡洛Dropout用于不确定性估计

Dropout工作原理：
---------------
1. 训练时：随机"丢弃"一部分神经元，打破神经元间的共适应关系
2. 推理时：保留所有神经元，但权重需要缩放以保持激活值的期望一致
3. 类似于集成多个子网络的效果，提高模型泛化能力

Dropout变体：
-----------
1. 标准Dropout：随机将激活值置零
2. 空间Dropout（Spatial Dropout）：丢弃整个特征图，适用于CNN
3. Alpha Dropout：保持均值和方差，适用于SELU激活函数
4. MC Dropout：推理时保持Dropout开启，进行多次采样得到不确定性估计

实现目标：
---------
1. 理解Dropout的原理和各种变体
2. 掌握不同位置添加Dropout的效果差异
3. 学习如何调整Dropout率以获得最佳性能
4. 使用MC Dropout进行模型不确定性估计
"""

class DropoutLeNet(nn.Module):
    def __init__(self, num_classes=10, dropout_type='standard', dropout_rate=0.5, 
                 feature_dropout=0.2, spatial_dropout=False, alpha_dropout=False):
        super(DropoutLeNet, self).__init__()
        
        self.dropout_type = dropout_type
        self.dropout_rate = dropout_rate
        self.feature_dropout = feature_dropout
        self.spatial_dropout = spatial_dropout
        self.alpha_dropout = alpha_dropout
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        
        # 池化层
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        
        # Dropout层 - 标准或Alpha
        if self.alpha_dropout:
            self.dropout1 = nn.AlphaDropout(self.feature_dropout)
            self.dropout2 = nn.AlphaDropout(self.feature_dropout)
            self.dropout3 = nn.AlphaDropout(self.dropout_rate)
            self.dropout4 = nn.AlphaDropout(self.dropout_rate)
        else:
            self.dropout1 = nn.Dropout2d(self.feature_dropout) if self.spatial_dropout else nn.Dropout(self.feature_dropout)
            self.dropout2 = nn.Dropout2d(self.feature_dropout) if self.spatial_dropout else nn.Dropout(self.feature_dropout)
            self.dropout3 = nn.Dropout(self.dropout_rate)
            self.dropout4 = nn.Dropout(self.dropout_rate)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # MC Dropout模式
        self.mc_dropout = False
    
    def set_mc_dropout(self, enable=True):
        """开启/关闭MC Dropout模式"""
        self.mc_dropout = enable
    
    def adjust_dropout_rate(self, new_rate):
        """动态调整Dropout率"""
        self.dropout_rate = new_rate
        
        if self.alpha_dropout:
            self.dropout3 = nn.AlphaDropout(new_rate)
            self.dropout4 = nn.AlphaDropout(new_rate)
        else:
            self.dropout3 = nn.Dropout(new_rate)
            self.dropout4 = nn.Dropout(new_rate)
        
        print(f"Dropout率已调整为 {new_rate}")
    
    def forward(self, x):
        """前向传播"""
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # 应用第一个Dropout
        if self.training or self.mc_dropout:
            x = self.dropout1(x)
        
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # 应用第二个Dropout
        if self.training or self.mc_dropout:
            x = self.dropout2(x)
        
        x = self.pool2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 第一个全连接层
        x = self.fc1(x)
        x = F.relu(x)
        
        # 应用第三个Dropout（较高的dropout率）
        if self.training or self.mc_dropout:
            x = self.dropout3(x)
        
        # 第二个全连接层
        x = self.fc2(x)
        x = F.relu(x)
        
        # 应用第四个Dropout
        if self.training or self.mc_dropout:
            x = self.dropout4(x)
        
        # 输出层
        x = self.fc3(x)
        
        return x
    
    def monte_carlo_predict(self, x, n_samples=10):
        """使用MC Dropout进行多次采样预测"""
        self.eval()  # 切换到评估模式，但保持dropout开启
        self.set_mc_dropout(True)
        
        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                output = self(x)
                probs = F.softmax(output, dim=1)
                samples.append(probs)
        
        # 重置MC Dropout模式
        self.set_mc_dropout(False)
        
        # 堆叠所有样本
        samples = torch.stack(samples)
        
        # 计算平均概率和不确定性
        mean_probs = samples.mean(dim=0)
        uncertainty = samples.std(dim=0)
        
        return mean_probs, uncertainty

def visualize_dropout_effects():
    """可视化不同Dropout策略的效果"""
    # 创建不同Dropout类型的模型
    models = {
        'No Dropout': DropoutLeNet(dropout_rate=0.0, feature_dropout=0.0),
        'Standard Dropout': DropoutLeNet(dropout_type='standard'),
        'Spatial Dropout': DropoutLeNet(dropout_type='standard', spatial_dropout=True),
        'Alpha Dropout': DropoutLeNet(dropout_type='standard', alpha_dropout=True),
        'High Dropout': DropoutLeNet(dropout_rate=0.7, feature_dropout=0.4),
        'Low Dropout': DropoutLeNet(dropout_rate=0.3, feature_dropout=0.1)
    }
    
    # 创建随机输入和目标
    x = torch.randn(100, 1, 28, 28)
    targets = torch.randint(0, 10, (100,))
    
    # 创建损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 评估模式下的损失
    print("不同Dropout配置的训练/测试模式效果:")
    print("---------------------------------")
    
    for name, model in models.items():
        # 训练模式
        model.train()
        outputs_train = model(x)
        loss_train = criterion(outputs_train, targets)
        
        # 评估模式
        model.eval()
        with torch.no_grad():
            outputs_eval = model(x)
            loss_eval = criterion(outputs_eval, targets)
        
        # 差异
        diff = torch.abs(outputs_train - outputs_eval).mean().item()
        
        print(f"{name}:")
        print(f"  训练损失: {loss_train.item():.4f}")
        print(f"  评估损失: {loss_eval.item():.4f}")
        print(f"  输出差异: {diff:.4f}")
    
    # MC Dropout示例
    print("\nMC Dropout不确定性估计示例:")
    x_single = torch.randn(1, 1, 28, 28)
    model = models['Standard Dropout']
    probs, uncertainty = model.monte_carlo_predict(x_single)
    
    top_probs, top_classes = torch.topk(probs[0], 3)
    top_uncertainty = uncertainty[0, top_classes]
    
    print(f"Top-3预测类别: {top_classes.tolist()}")
    print(f"Top-3预测概率: {top_probs.tolist()}")
    print(f"Top-3不确定性: {top_uncertainty.tolist()}")

def test():
    """测试DropoutLeNet模型"""
    # 创建模型
    model = DropoutLeNet(num_classes=10)
    
    # 创建随机输入
    x = torch.randn(1, 1, 28, 28)
    
    # 训练模式测试
    model.train()
    y_train = model(x)
    print(f"训练模式输出形状: {y_train.shape}")
    
    # 评估模式测试
    model.eval()
    y_eval = model(x)
    print(f"评估模式输出形状: {y_eval.shape}")
    
    # 测试不同的Dropout配置
    dropout_configs = [
        {'name': '低Dropout', 'rate': 0.2},
        {'name': '中Dropout', 'rate': 0.5},
        {'name': '高Dropout', 'rate': 0.8},
    ]
    
    for config in dropout_configs:
        model.adjust_dropout_rate(config['rate'])
        model.train()
        y = model(x)
        print(f"{config['name']}训练输出形状: {y.shape}")
    
    # MC Dropout测试
    print("\nMC Dropout测试:")
    probs, uncertainty = model.monte_carlo_predict(x, n_samples=15)
    print(f"平均概率形状: {probs.shape}")
    print(f"不确定性形状: {uncertainty.shape}")
    print(f"最大不确定性: {uncertainty.max().item():.4f}")
    
    # 可视化Dropout效果
    print("\nDropout效果可视化:")
    visualize_dropout_effects()

if __name__ == '__main__':
    test() 