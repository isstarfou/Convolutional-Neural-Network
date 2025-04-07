import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNetELU(nn.Module):
    def __init__(self, num_classes=10, alpha=1.0):
        super(LeNetELU, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        
        # 池化层
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # ELU参数
        self.alpha = alpha
        
    def forward(self, x):
        # 第一层
        x = self.conv1(x)
        x = F.elu(x, alpha=self.alpha)  # ELU激活
        x = self.pool1(x)
        
        # 第二层
        x = self.conv2(x)
        x = F.elu(x, alpha=self.alpha)  # ELU激活
        x = self.pool2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.elu(self.fc1(x), alpha=self.alpha)  # ELU激活
        x = F.elu(self.fc2(x), alpha=self.alpha)  # ELU激活
        x = self.fc3(x)
        
        return x

def test():
    model = LeNetELU()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.shape)

if __name__ == '__main__':
    test() 