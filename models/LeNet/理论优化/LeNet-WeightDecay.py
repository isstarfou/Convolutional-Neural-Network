import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightDecayLeNet(nn.Module):
    def __init__(self, num_classes=10, weight_decay=0.01):
        super(WeightDecayLeNet, self).__init__()
        
        self.weight_decay = weight_decay
        
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
        
    def get_weight_decay_loss(self):
        """计算权重衰减损失"""
        l2_reg = torch.tensor(0., requires_grad=True)
        for param in self.parameters():
            if param.requires_grad:
                l2_reg = l2_reg + torch.norm(param, p=2)
        return self.weight_decay * l2_reg

def test():
    model = WeightDecayLeNet()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.shape)
    
    # 计算权重衰减损失
    weight_decay_loss = model.get_weight_decay_loss()
    print("Weight decay loss:", weight_decay_loss.item())

if __name__ == '__main__':
    test() 