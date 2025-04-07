import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNetLeakyReLU(nn.Module):
    def __init__(self, num_classes=10, negative_slope=0.01):
        super(LeNetLeakyReLU, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        
        # LeakyReLU参数
        self.negative_slope = negative_slope
        
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
        x = F.leaky_relu(x, negative_slope=self.negative_slope)  # LeakyReLU激活
        x = self.pool1(x)
        
        # 第二层
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=self.negative_slope)  # LeakyReLU激活
        x = self.pool2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.leaky_relu(self.fc1(x), negative_slope=self.negative_slope)  # LeakyReLU激活
        x = F.leaky_relu(self.fc2(x), negative_slope=self.negative_slope)  # LeakyReLU激活
        x = self.fc3(x)
        
        return x

def test():
    model = LeNetLeakyReLU()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.shape)

if __name__ == '__main__':
    test()
