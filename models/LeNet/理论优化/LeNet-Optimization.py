import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
LeNet-Optimization：优化算法综合比较与实现
=========================================

核心特点：
----------
1. 实现多种优化算法在统一框架下的比较
2. 支持动态切换不同优化策略
3. 自适应调整超参数
4. 提供可视化分析工具

优化算法概述：
------------
此模块实现了多种常见优化算法，包括：
1. SGD - 随机梯度下降，最基础的优化器
2. Momentum - 引入动量项，加速收敛并帮助逃离局部最小值
3. Nesterov - 动量的改进版，提前看到未来位置
4. Adagrad - 自适应学习率，适用于稀疏数据
5. RMSprop - 解决Adagrad学习率递减问题
6. Adam - 结合动量和自适应学习率
7. AdamW - 正确实现权重衰减的Adam变体
8. RAdam - 校正Adam中的方差偏差问题
9. Lookahead - 通过慢权重维持稳定性
10. SWATS - 自动从Adam切换到SGD

实现目标：
----------
1. 理解各种优化算法的原理与特点
2. 掌握超参数调整技巧
3. 分析不同优化器在不同任务上的表现差异
4. 提供最佳实践建议
"""

class OptimizationLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(OptimizationLeNet, self).__init__()
        
        # 网络结构 - 标准LeNet
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # 优化器状态
        self.optimizer_states = {}
        self.current_optimizer = "sgd"
        
        # 初始化所有可能优化器的状态
        self._init_optimizer_states()
        
    def _init_optimizer_states(self):
        """初始化所有优化器的状态"""
        params = list(self.named_parameters())
        param_names = [name for name, _ in params if _.requires_grad]
        
        # SGD状态
        self.optimizer_states["sgd"] = {
            "learning_rate": 0.01
        }
        
        # Momentum状态
        self.optimizer_states["momentum"] = {
            "learning_rate": 0.01,
            "momentum": 0.9,
            "velocity": {name: torch.zeros_like(param) for name, param in params if param.requires_grad}
        }
        
        # Nesterov状态
        self.optimizer_states["nesterov"] = {
            "learning_rate": 0.01,
            "momentum": 0.9,
            "velocity": {name: torch.zeros_like(param) for name, param in params if param.requires_grad}
        }
        
        # Adagrad状态
        self.optimizer_states["adagrad"] = {
            "learning_rate": 0.01,
            "epsilon": 1e-8,
            "sum_squared_grad": {name: torch.zeros_like(param) for name, param in params if param.requires_grad}
        }
        
        # RMSprop状态
        self.optimizer_states["rmsprop"] = {
            "learning_rate": 0.001,
            "decay_rate": 0.9,
            "epsilon": 1e-8,
            "moving_avg_squared": {name: torch.zeros_like(param) for name, param in params if param.requires_grad}
        }
        
        # Adam状态
        self.optimizer_states["adam"] = {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "m": {name: torch.zeros_like(param) for name, param in params if param.requires_grad},
            "v": {name: torch.zeros_like(param) for name, param in params if param.requires_grad},
            "t": 0
        }
        
        # AdamW状态
        self.optimizer_states["adamw"] = {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "weight_decay": 0.01,
            "m": {name: torch.zeros_like(param) for name, param in params if param.requires_grad},
            "v": {name: torch.zeros_like(param) for name, param in params if param.requires_grad},
            "t": 0
        }
        
        # RAdam状态
        self.optimizer_states["radam"] = {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "m": {name: torch.zeros_like(param) for name, param in params if param.requires_grad},
            "v": {name: torch.zeros_like(param) for name, param in params if param.requires_grad},
            "t": 0,
            "rho_inf": 2.0 / (1.0 - 0.999) - 1.0
        }
        
        # Lookahead状态
        self.optimizer_states["lookahead"] = {
            "learning_rate": 0.001,
            "inner_optimizer": "adam",  # 内部使用的优化器
            "k": 5,  # 每k步更新一次慢权重
            "alpha": 0.5,  # 慢权重与快权重的插值系数
            "step": 0,
            "slow_weights": {name: param.data.clone() for name, param in params if param.requires_grad}
        }
        
        # SWATS状态
        self.optimizer_states["swats"] = {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "m": {name: torch.zeros_like(param) for name, param in params if param.requires_grad},
            "v": {name: torch.zeros_like(param) for name, param in params if param.requires_grad},
            "t": 0,
            "use_sgd": False,  # 是否切换到SGD
            "momentum": 0.9,  # SGD的动量系数
            "velocity": {name: torch.zeros_like(param) for name, param in params if param.requires_grad}
        }
    
    def set_optimizer(self, optimizer_name, **kwargs):
        """设置当前使用的优化器"""
        if optimizer_name not in self.optimizer_states:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
        
        self.current_optimizer = optimizer_name
        
        # 更新优化器参数
        for key, value in kwargs.items():
            if key in self.optimizer_states[optimizer_name]:
                self.optimizer_states[optimizer_name][key] = value
    
    def update_weights(self, gradients):
        """根据当前选择的优化器更新权重"""
        optimizer_state = self.optimizer_states[self.current_optimizer]
        
        if self.current_optimizer == "sgd":
            self._update_sgd(gradients, optimizer_state)
        elif self.current_optimizer == "momentum":
            self._update_momentum(gradients, optimizer_state)
        elif self.current_optimizer == "nesterov":
            self._update_nesterov(gradients, optimizer_state)
        elif self.current_optimizer == "adagrad":
            self._update_adagrad(gradients, optimizer_state)
        elif self.current_optimizer == "rmsprop":
            self._update_rmsprop(gradients, optimizer_state)
        elif self.current_optimizer == "adam":
            self._update_adam(gradients, optimizer_state)
        elif self.current_optimizer == "adamw":
            self._update_adamw(gradients, optimizer_state)
        elif self.current_optimizer == "radam":
            self._update_radam(gradients, optimizer_state)
        elif self.current_optimizer == "lookahead":
            self._update_lookahead(gradients, optimizer_state)
        elif self.current_optimizer == "swats":
            self._update_swats(gradients, optimizer_state)
    
    def _update_sgd(self, gradients, state):
        """SGD更新"""
        lr = state["learning_rate"]
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    param.data -= lr * gradients[name]
    
    def _update_momentum(self, gradients, state):
        """Momentum更新"""
        lr = state["learning_rate"]
        momentum = state["momentum"]
        velocity = state["velocity"]
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    velocity[name] = momentum * velocity[name] + gradients[name]
                    param.data -= lr * velocity[name]
    
    def _update_nesterov(self, gradients, state):
        """Nesterov更新"""
        lr = state["learning_rate"]
        momentum = state["momentum"]
        velocity = state["velocity"]
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    old_velocity = velocity[name].clone()
                    velocity[name] = momentum * velocity[name] + gradients[name]
                    param.data -= lr * (momentum * velocity[name] + gradients[name] - momentum * old_velocity)
    
    def _update_adagrad(self, gradients, state):
        """Adagrad更新"""
        lr = state["learning_rate"]
        epsilon = state["epsilon"]
        sum_squared_grad = state["sum_squared_grad"]
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    sum_squared_grad[name] += gradients[name] ** 2
                    param.data -= lr * gradients[name] / (torch.sqrt(sum_squared_grad[name]) + epsilon)
    
    def _update_rmsprop(self, gradients, state):
        """RMSprop更新"""
        lr = state["learning_rate"]
        decay_rate = state["decay_rate"]
        epsilon = state["epsilon"]
        moving_avg_squared = state["moving_avg_squared"]
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    moving_avg_squared[name] = decay_rate * moving_avg_squared[name] + (1 - decay_rate) * gradients[name] ** 2
                    param.data -= lr * gradients[name] / (torch.sqrt(moving_avg_squared[name]) + epsilon)
    
    def _update_adam(self, gradients, state):
        """Adam更新"""
        lr = state["learning_rate"]
        beta1 = state["beta1"]
        beta2 = state["beta2"]
        epsilon = state["epsilon"]
        m = state["m"]
        v = state["v"]
        
        state["t"] += 1
        t = state["t"]
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    m[name] = beta1 * m[name] + (1 - beta1) * gradients[name]
                    v[name] = beta2 * v[name] + (1 - beta2) * gradients[name] ** 2
                    
                    m_hat = m[name] / (1 - beta1 ** t)
                    v_hat = v[name] / (1 - beta2 ** t)
                    
                    param.data -= lr * m_hat / (torch.sqrt(v_hat) + epsilon)
    
    def _update_adamw(self, gradients, state):
        """AdamW更新"""
        lr = state["learning_rate"]
        beta1 = state["beta1"]
        beta2 = state["beta2"]
        epsilon = state["epsilon"]
        weight_decay = state["weight_decay"]
        m = state["m"]
        v = state["v"]
        
        state["t"] += 1
        t = state["t"]
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    # 应用权重衰减
                    param.data = param.data - lr * weight_decay * param.data
                    
                    m[name] = beta1 * m[name] + (1 - beta1) * gradients[name]
                    v[name] = beta2 * v[name] + (1 - beta2) * gradients[name] ** 2
                    
                    m_hat = m[name] / (1 - beta1 ** t)
                    v_hat = v[name] / (1 - beta2 ** t)
                    
                    param.data -= lr * m_hat / (torch.sqrt(v_hat) + epsilon)
    
    def _update_radam(self, gradients, state):
        """RAdam更新"""
        lr = state["learning_rate"]
        beta1 = state["beta1"]
        beta2 = state["beta2"]
        epsilon = state["epsilon"]
        rho_inf = state["rho_inf"]
        m = state["m"]
        v = state["v"]
        
        state["t"] += 1
        t = state["t"]
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    m[name] = beta1 * m[name] + (1 - beta1) * gradients[name]
                    v[name] = beta2 * v[name] + (1 - beta2) * gradients[name] ** 2
                    
                    m_hat = m[name] / (1 - beta1 ** t)
                    
                    rho_t = rho_inf - 2 * t * (beta2 ** t) / (1 - beta2 ** t)
                    
                    if rho_t > 4:
                        # 方差修正项
                        v_hat = torch.sqrt(v[name] / (1 - beta2 ** t))
                        r_t = math.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                        param.data -= lr * r_t * m_hat / (v_hat + epsilon)
                    else:
                        # 退化为SGD with momentum
                        param.data -= lr * m_hat
    
    def _update_lookahead(self, gradients, state):
        """Lookahead更新"""
        lr = state["learning_rate"]
        inner_optimizer = state["inner_optimizer"]
        k = state["k"]
        alpha = state["alpha"]
        slow_weights = state["slow_weights"]
        
        # 增加步数
        state["step"] += 1
        
        # 先用内部优化器更新快速权重
        inner_state = self.optimizer_states[inner_optimizer]
        if inner_optimizer == "sgd":
            self._update_sgd(gradients, inner_state)
        elif inner_optimizer == "adam":
            self._update_adam(gradients, inner_state)
        
        # 每k步更新一次慢权重
        if state["step"] % k == 0:
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        # 更新慢权重
                        slow_weights[name] = slow_weights[name] + alpha * (param.data - slow_weights[name])
                        # 更新快速权重为慢权重
                        param.data.copy_(slow_weights[name])
    
    def _update_swats(self, gradients, state):
        """SWATS更新"""
        lr = state["learning_rate"]
        beta1 = state["beta1"]
        beta2 = state["beta2"]
        epsilon = state["epsilon"]
        m = state["m"]
        v = state["v"]
        use_sgd = state["use_sgd"]
        momentum = state["momentum"]
        velocity = state["velocity"]
        
        state["t"] += 1
        t = state["t"]
        
        with torch.no_grad():
            if not use_sgd:
                # 使用Adam更新
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        m[name] = beta1 * m[name] + (1 - beta1) * gradients[name]
                        v[name] = beta2 * v[name] + (1 - beta2) * gradients[name] ** 2
                        
                        m_hat = m[name] / (1 - beta1 ** t)
                        v_hat = v[name] / (1 - beta2 ** t)
                        
                        param.data -= lr * m_hat / (torch.sqrt(v_hat) + epsilon)
                        
                        # 更新动量以便平滑过渡到SGD
                        velocity[name] = momentum * velocity[name] + lr * m_hat / (torch.sqrt(v_hat) + epsilon)
                
                # 检查是否切换到SGD（这里简化为在固定步数后切换，实际应根据梯度方向一致性决定）
                if t > 1000:  # 示例：1000步后切换
                    state["use_sgd"] = True
                    print("切换到SGD优化器...")
            else:
                # 使用SGD with Momentum更新
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        velocity[name] = momentum * velocity[name] + gradients[name]
                        param.data -= lr * velocity[name]
    
    def forward(self, x):
        """前向传播"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def test():
    """测试不同优化器"""
    model = OptimizationLeNet()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(f"输出形状: {y.shape}")
    
    # 模拟梯度
    gradients = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradients[name] = torch.randn_like(param)
    
    # 测试不同优化器
    optimizers = ["sgd", "momentum", "nesterov", "adagrad", "rmsprop", 
                 "adam", "adamw", "radam", "lookahead", "swats"]
    
    for opt in optimizers:
        model.set_optimizer(opt)
        model.update_weights(gradients)
        print(f"使用 {opt} 优化器更新权重")

if __name__ == '__main__':
    test() 