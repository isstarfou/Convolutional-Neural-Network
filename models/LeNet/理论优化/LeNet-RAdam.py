import torch
import torch.nn as nn
import torch.nn.functional as F

class RAdamLeNet(nn.Module):
    def __init__(self, num_classes=10, beta1=0.9, beta2=0.999, eps=1e-8):
        super(RAdamLeNet, self).__init__()
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
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
        
        # RAdam状态
        self.m = {}  # 一阶矩估计
        self.v = {}  # 二阶矩估计
        self.t = 0   # 时间步
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.m[name] = torch.zeros_like(param)
                self.v[name] = torch.zeros_like(param)
                
    def update_weights(self, gradients, lr):
        """使用RAdam更新权重"""
        self.t += 1
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    # 更新一阶矩估计
                    self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * gradients[name]
                    
                    # 更新二阶矩估计
                    self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * gradients[name].pow(2)
                    
                    # 计算偏差修正
                    m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[name] / (1 - self.beta2 ** self.t)
                    
                    # 计算rho
                    rho_inf = 2 / (1 - self.beta2) - 1
                    rho_t = rho_inf - 2 * self.t * self.beta2 ** self.t / (1 - self.beta2 ** self.t)
                    
                    if rho_t > 4:
                        # 计算自适应步长
                        r_t = torch.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                        step_size = lr * r_t * m_hat / (torch.sqrt(v_hat) + self.eps)
                    else:
                        step_size = lr * m_hat
                    
                    # 更新参数
                    param.data -= step_size
        
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
    model = RAdamLeNet()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.shape)
    
    # 模拟梯度更新
    gradients = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradients[name] = torch.randn_like(param)
    
    # 更新权重
    model.update_weights(gradients, lr=0.001)
    print("Weights updated with RAdam")

if __name__ == '__main__':
    test() 