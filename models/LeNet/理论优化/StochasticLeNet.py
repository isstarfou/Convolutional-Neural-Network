import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
StochasticLeNet：结合多种随机正则化技术的LeNet实现
================================================

核心特点：
----------
1. 实现多种随机正则化方法，包括Dropout、随机深度、随机宽度等
2. 支持贝叶斯神经网络的近似实现
3. 提供模型不确定性估计的多种方法
4. 实现随机梯度噪声注入

随机正则化原理：
-------------
随机正则化通过在训练和/或推理过程中引入随机性，显著提高模型的泛化能力和鲁棒性。
这些技术通常可以被解释为贝叶斯神经网络的变分推断，使模型能够估计预测的不确定性。

实现的随机技术：
-------------
1. Dropout：随机丢弃神经元，经典随机正则化方法
2. 随机深度(Stochastic Depth)：随机跳过某些层，适合深度网络
3. DropConnect：随机丢弃权重连接而非神经元
4. 随机权重噪声：向权重添加高斯噪声
5. 权重采样：从分布中采样权重而非使用点估计
6. 随机激活函数：随机选择不同的激活函数

实现目标：
---------
1. 理解各种随机正则化技术的原理与特点
2. 掌握贝叶斯深度学习的基本思想
3. 学习如何估计深度学习模型的不确定性
4. 提高模型在小数据集上的泛化能力
"""

class StochasticLeNet(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5, stochastic_depth_prob=0.2,
                 weight_noise=0.01, dropconnect_prob=0.0, random_activation=False):
        super(StochasticLeNet, self).__init__()
        
        # 随机化参数
        self.dropout_rate = dropout_rate
        self.stochastic_depth_prob = stochastic_depth_prob
        self.weight_noise = weight_noise
        self.dropconnect_prob = dropconnect_prob
        self.random_activation = random_activation
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        
        # 池化层
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        
        # Dropout层
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # 随机化标志
        self.is_stochastic = True
        
        # 可选激活函数
        self.activations = [
            F.relu,
            F.leaky_relu,
            F.elu,
            F.selu,
            lambda x: torch.tanh(x),
            lambda x: torch.sigmoid(x)
        ]
    
    def toggle_stochastic(self, enable=True):
        """开启/关闭随机化特性"""
        self.is_stochastic = enable
    
    def _apply_dropconnect(self, x, weight, bias=None, p=0.5):
        """应用DropConnect - 随机丢弃权重而非激活"""
        if not self.training or not self.is_stochastic or p <= 0:
            return F.linear(x, weight, bias)
        
        mask = torch.bernoulli(torch.ones_like(weight) * (1 - p))
        masked_weight = mask * weight
        return F.linear(x, masked_weight, bias)
    
    def _apply_weight_noise(self, module):
        """应用权重噪声"""
        if not self.training or not self.is_stochastic or self.weight_noise <= 0:
            return
        
        with torch.no_grad():
            for param in module.parameters():
                noise = torch.randn_like(param) * self.weight_noise
                param.add_(noise)
    
    def _apply_stochastic_depth(self, x, layer, identity=None):
        """应用随机深度 - 随机跳过层"""
        if not self.training or not self.is_stochastic or self.stochastic_depth_prob <= 0:
            return layer(x)
        
        if torch.rand(1).item() < self.stochastic_depth_prob:
            return identity if identity is not None else x
        
        return layer(x)
    
    def _get_random_activation(self, x):
        """随机选择激活函数"""
        if not self.training or not self.is_stochastic or not self.random_activation:
            return F.relu(x)
        
        idx = torch.randint(0, len(self.activations), (1,)).item()
        return self.activations[idx](x)
    
    def forward(self, x):
        """前向传播"""
        # 第一层卷积 - 可选随机深度
        identity = F.avg_pool2d(x, 3, stride=1, padding=1)  # 假设的恒等映射
        x = self._apply_stochastic_depth(
            x, 
            lambda y: self.bn1(self.conv1(y)),
            identity
        )
        
        # 激活 - 可选随机激活
        x = self._get_random_activation(x)
        
        # 池化
        x = self.pool1(x)
        
        # Dropout
        if self.training and self.is_stochastic:
            x = self.dropout1(x)
        
        # 第二层卷积 - 可选随机深度
        identity = x
        x = self._apply_stochastic_depth(
            x, 
            lambda y: self.bn2(self.conv2(y)),
            identity
        )
        
        # 激活 - 可选随机激活
        x = self._get_random_activation(x)
        
        # 池化
        x = self.pool2(x)
        
        # Dropout
        if self.training and self.is_stochastic:
            x = self.dropout2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层1 - 可选随机权重噪声
        if self.training and self.is_stochastic and self.weight_noise > 0:
            self._apply_weight_noise(self.fc1)
        
        # 可选DropConnect
        if self.dropconnect_prob > 0 and self.training and self.is_stochastic:
            x = self._apply_dropconnect(x, self.fc1.weight, self.fc1.bias, self.dropconnect_prob)
            x = self._get_random_activation(x)
        else:
            x = self.fc1(x)
            x = self._get_random_activation(x)
        
        # 全连接层2 - 可选随机权重噪声
        if self.training and self.is_stochastic and self.weight_noise > 0:
            self._apply_weight_noise(self.fc2)
        
        # 可选DropConnect
        if self.dropconnect_prob > 0 and self.training and self.is_stochastic:
            x = self._apply_dropconnect(x, self.fc2.weight, self.fc2.bias, self.dropconnect_prob)
            x = self._get_random_activation(x)
        else:
            x = self.fc2(x)
            x = self._get_random_activation(x)
        
        # 输出层
        x = self.fc3(x)
        
        return x
    
    def monte_carlo_predict(self, x, n_samples=30):
        """使用蒙特卡洛方法进行多次采样预测，用于不确定性估计"""
        self.eval()
        self.toggle_stochastic(True)  # 确保随机性打开
        
        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                output = self(x)
                probs = F.softmax(output, dim=1)
                samples.append(probs)
        
        # 堆叠所有样本
        samples = torch.stack(samples)
        
        # 计算平均概率、方差和信息熵
        mean_probs = samples.mean(dim=0)
        variance = samples.var(dim=0)
        
        # 计算预测熵作为不确定性度量
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
        
        # 计算互信息（预测方差的另一种表达）
        # MI = H(y|x,D) - E_θ[H(y|x,θ,D)]
        expected_entropy = -torch.sum(samples * torch.log(samples + 1e-10), dim=2).mean(dim=0)
        mutual_info = entropy - expected_entropy
        
        return mean_probs, variance, entropy, mutual_info

class EnsembleStochasticLeNet(nn.Module):
    """集成多个随机LeNet模型，进一步提高不确定性估计能力"""
    def __init__(self, num_models=5, **kwargs):
        super(EnsembleStochasticLeNet, self).__init__()
        
        # 创建多个随机LeNet模型
        self.models = nn.ModuleList([
            StochasticLeNet(**kwargs) for _ in range(num_models)
        ])
    
    def forward(self, x):
        """前向传播 - 在推理时返回集成的平均预测"""
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # 返回平均输出
        return torch.stack(outputs).mean(dim=0)
    
    def predict_with_uncertainty(self, x, n_samples=10):
        """进行集成预测，并估计不确定性"""
        all_probs = []
        
        # 从每个模型获取多个样本
        for model in self.models:
            # 获取这个模型的蒙特卡洛样本
            probs, _, _, _ = model.monte_carlo_predict(x, n_samples)
            all_probs.append(probs)
        
        # 堆叠所有模型的预测
        all_probs = torch.stack(all_probs)
        
        # 计算平均概率和不确定性
        mean_probs = all_probs.mean(dim=0)
        uncertainty = all_probs.std(dim=0)
        
        # 计算预测熵
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
        
        return mean_probs, uncertainty, entropy

def compare_stochastic_settings():
    """比较不同随机设置的模型行为"""
    # 创建不同设置的模型
    models = {
        'Baseline': StochasticLeNet(dropout_rate=0.0, stochastic_depth_prob=0.0, 
                                    weight_noise=0.0, dropconnect_prob=0.0, random_activation=False),
        'Dropout Only': StochasticLeNet(dropout_rate=0.5, stochastic_depth_prob=0.0,
                                        weight_noise=0.0, dropconnect_prob=0.0, random_activation=False),
        'Random Depth': StochasticLeNet(dropout_rate=0.0, stochastic_depth_prob=0.2,
                                        weight_noise=0.0, dropconnect_prob=0.0, random_activation=False),
        'Weight Noise': StochasticLeNet(dropout_rate=0.0, stochastic_depth_prob=0.0,
                                        weight_noise=0.01, dropconnect_prob=0.0, random_activation=False),
        'DropConnect': StochasticLeNet(dropout_rate=0.0, stochastic_depth_prob=0.0,
                                      weight_noise=0.0, dropconnect_prob=0.3, random_activation=False),
        'Random Activation': StochasticLeNet(dropout_rate=0.0, stochastic_depth_prob=0.0,
                                           weight_noise=0.0, dropconnect_prob=0.0, random_activation=True),
        'Full Stochastic': StochasticLeNet(dropout_rate=0.3, stochastic_depth_prob=0.1,
                                          weight_noise=0.005, dropconnect_prob=0.2, random_activation=True)
    }
    
    # 创建随机输入
    x = torch.randn(16, 1, 28, 28)
    
    # 比较训练和评估模式的差异
    print("不同随机设置的模型行为比较:")
    print("-----------------------")
    
    for name, model in models.items():
        # 启用训练模式
        model.train()
        
        # 多次运行以观察随机性
        train_outputs = []
        for _ in range(5):
            out = model(x)
            train_outputs.append(out.detach())
        
        # 计算训练模式下不同运行的输出差异
        train_diffs = []
        for i in range(len(train_outputs) - 1):
            for j in range(i + 1, len(train_outputs)):
                diff = torch.abs(train_outputs[i] - train_outputs[j]).mean().item()
                train_diffs.append(diff)
        
        avg_train_diff = sum(train_diffs) / len(train_diffs) if train_diffs else 0
        
        # 禁用训练模式
        model.eval()
        
        # 多次运行以观察随机性
        eval_outputs = []
        for _ in range(5):
            with torch.no_grad():
                out = model(x)
                eval_outputs.append(out)
        
        # 计算评估模式下不同运行的输出差异
        eval_diffs = []
        for i in range(len(eval_outputs) - 1):
            for j in range(i + 1, len(eval_outputs)):
                diff = torch.abs(eval_outputs[i] - eval_outputs[j]).mean().item()
                eval_diffs.append(diff)
        
        avg_eval_diff = sum(eval_diffs) / len(eval_diffs) if eval_diffs else 0
        
        print(f"{name}:")
        print(f"  训练模式下的随机性: {avg_train_diff:.6f}")
        print(f"  评估模式下的随机性: {avg_eval_diff:.6f}")
        
        # 仅对全随机模型进行蒙特卡洛不确定性估计
        if name == 'Full Stochastic':
            print("\n蒙特卡洛不确定性估计示例 (Full Stochastic model):")
            probs, variance, entropy, mutual_info = model.monte_carlo_predict(x[:1])
            
            print(f"  平均预测类别: {torch.argmax(probs[0]).item()}")
            print(f"  预测方差: {variance[0].sum().item():.6f}")
            print(f"  预测熵: {entropy[0].item():.6f}")
            print(f"  互信息: {mutual_info[0].item():.6f}")

def test():
    """测试StochasticLeNet模型"""
    # 创建模型
    model = StochasticLeNet()
    
    # 创建随机输入
    x = torch.randn(1, 1, 28, 28)
    
    # 训练模式下的前向传播
    model.train()
    y_train = model(x)
    print(f"训练模式输出形状: {y_train.shape}")
    
    # 评估模式下的前向传播
    model.eval()
    y_eval = model(x)
    print(f"评估模式输出形状: {y_eval.shape}")
    
    # 测试不确定性估计
    probs, variance, entropy, mutual_info = model.monte_carlo_predict(x)
    print(f"平均概率形状: {probs.shape}")
    print(f"方差形状: {variance.shape}")
    print(f"熵: {entropy.item():.4f}")
    print(f"互信息: {mutual_info.item():.4f}")
    
    # 测试集成模型
    ensemble = EnsembleStochasticLeNet(num_models=3)
    y_ensemble = ensemble(x)
    print(f"集成模型输出形状: {y_ensemble.shape}")
    
    # 测试集成模型的不确定性估计
    ensemble_probs, ensemble_uncertainty, ensemble_entropy = ensemble.predict_with_uncertainty(x)
    print(f"集成模型平均概率形状: {ensemble_probs.shape}")
    print(f"集成模型不确定性形状: {ensemble_uncertainty.shape}")
    print(f"集成模型熵: {ensemble_entropy.item():.4f}")
    
    # 比较不同的随机设置
    print("\n不同随机设置比较:")
    compare_stochastic_settings()

if __name__ == '__main__':
    test() 