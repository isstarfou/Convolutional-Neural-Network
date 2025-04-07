import torch
import torch.nn as nn
import torch.nn.functional as F

class MomentumLeNet(nn.Module):
    def __init__(self, num_classes=10, momentum=0.9):
        super(MomentumLeNet, self).__init__()
        
        self.momentum = momentum
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        
        # 批归一化层（使用动量）
        self.bn1 = nn.BatchNorm2d(6, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(16, momentum=momentum)
        
        # 池化层
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # 动量缓冲区
        self.velocity = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.velocity[name] = torch.zeros_like(param)
                
    def update_weights(self, gradients, lr):
        """使用动量更新权重"""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    # 更新速度
                    self.velocity[name] = self.momentum * self.velocity[name] + gradients[name]
                    # 更新参数
                    param.data -= lr * self.velocity[name]
        
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
    model = MomentumLeNet()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.shape)
    
    # 模拟梯度更新
    gradients = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradients[name] = torch.randn_like(param)
    
    # 更新权重
    model.update_weights(gradients, lr=0.01)
    print("Weights updated with momentum")

if __name__ == '__main__':
    test() 