import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNetPReLU(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNetPReLU, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        
        # PReLU激活函数
        self.prelu1 = nn.PReLU(6)  # 6个通道
        self.prelu2 = nn.PReLU(16)  # 16个通道
        self.prelu3 = nn.PReLU(120)  # 120个神经元
        self.prelu4 = nn.PReLU(84)  # 84个神经元
        
        # 池化层
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        # 第一层
        x = self.conv1(x)
        x = self.prelu1(x)  # PReLU激活
        x = self.pool1(x)
        
        # 第二层
        x = self.conv2(x)
        x = self.prelu2(x)  # PReLU激活
        x = self.pool2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.prelu3(x)  # PReLU激活
        x = self.fc2(x)
        x = self.prelu4(x)  # PReLU激活
        x = self.fc3(x)
        
        return x

def test():
    model = LeNetPReLU()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.shape)

if __name__ == '__main__':
    test() 