import torch
import torch.nn as nn
import torch.nn.functional as F

class SWATSLeNet(nn.Module):
    def __init__(self, num_classes=10, beta1=0.9, beta2=0.999, eps=1e-8):
        super(SWATSLeNet, self).__init__()
        
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
        
        # SWATS状态
        self.m = {}  # 一阶矩估计
        self.v = {}  # 二阶矩估计
        self.t = 0   # 时间步
        self.switched = False  # 是否已切换到SGD
        self.avg_grad = {}  # 平均梯度
        self.avg_grad_sq = {}  # 平均梯度平方
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.m[name] = torch.zeros_like(param)
                self.v[name] = torch.zeros_like(param)
                self.avg_grad[name] = torch.zeros_like(param)
                self.avg_grad_sq[name] = torch.zeros_like(param)
                
    def update_weights(self, gradients, lr):
        """使用SWATS更新权重"""
        self.t += 1
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if not self.switched:
                        # Adam阶段
                        self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * gradients[name]
                        self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * gradients[name].pow(2)
                        
                        m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                        v_hat = self.v[name] / (1 - self.beta2 ** self.t)
                        
                        # 更新参数
                        param.data -= lr * m_hat / (torch.sqrt(v_hat) + self.eps)
                        
                        # 更新平均梯度
                        self.avg_grad[name] = self.beta1 * self.avg_grad[name] + (1 - self.beta1) * gradients[name]
                        self.avg_grad_sq[name] = self.beta2 * self.avg_grad_sq[name] + (1 - self.beta2) * gradients[name].pow(2)
                        
                        # 检查是否切换到SGD
                        if self.t > 1:
                            avg_grad_hat = self.avg_grad[name] / (1 - self.beta1 ** self.t)
                            avg_grad_sq_hat = self.avg_grad_sq[name] / (1 - self.beta2 ** self.t)
                            
                            # 如果梯度方差足够小，切换到SGD
                            if torch.mean(avg_grad_sq_hat - avg_grad_hat.pow(2)) < 1e-8:
                                self.switched = True
                                print("Switching to SGD")
                    else:
                        # SGD阶段
                        param.data -= lr * gradients[name]
        
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
    model = SWATSLeNet()
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
    print("Weights updated with SWATS")

if __name__ == '__main__':
    test() 