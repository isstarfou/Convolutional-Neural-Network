import torch
import torch.nn as nn
import torch.nn.functional as F

class LookaheadLeNet(nn.Module):
    def __init__(self, num_classes=10, k=5, alpha=0.5):
        super(LookaheadLeNet, self).__init__()
        
        self.k = k  # 更新频率
        self.alpha = alpha  # 插值系数
        self.step = 0  # 当前步数
        
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
        
        # 慢权重
        self.slow_weights = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.slow_weights[name] = param.data.clone()
                
    def update_weights(self, gradients, lr):
        """使用Lookahead更新权重"""
        self.step += 1
        
        with torch.no_grad():
            # 更新快速权重
            for name, param in self.named_parameters():
                if param.requires_grad:
                    param.data -= lr * gradients[name]
            
            # 每k步更新一次慢权重
            if self.step % self.k == 0:
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        # 更新慢权重
                        self.slow_weights[name] = self.slow_weights[name] + self.alpha * (param.data - self.slow_weights[name])
                        # 更新快速权重
                        param.data.copy_(self.slow_weights[name])
        
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
    model = LookaheadLeNet()
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
    print("Weights updated with Lookahead")

if __name__ == '__main__':
    test() 