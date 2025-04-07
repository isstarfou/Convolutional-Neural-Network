import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSpropLeNet(nn.Module):
    def __init__(self, num_classes=10, alpha=0.99, eps=1e-8):
        super(RMSpropLeNet, self).__init__()
        
        self.alpha = alpha
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
        
        # RMSprop状态
        self.square_avg = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.square_avg[name] = torch.zeros_like(param)
                
    def update_weights(self, gradients, lr):
        """使用RMSprop更新权重"""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    # 更新平方梯度移动平均
                    self.square_avg[name] = self.alpha * self.square_avg[name] + \
                                          (1 - self.alpha) * gradients[name].pow(2)
                    
                    # 更新参数
                    param.data -= lr * gradients[name] / (torch.sqrt(self.square_avg[name]) + self.eps)
        
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
    model = RMSpropLeNet()
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
    print("Weights updated with RMSprop")

if __name__ == '__main__':
    test() 