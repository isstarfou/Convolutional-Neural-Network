import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainAdaptationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DomainAdaptationModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        attention = self.attention(x)
        return x * attention

class DomainAdaptLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(DomainAdaptLeNet, self).__init__()
        
        # 基础特征提取器
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        
        # 域自适应模块
        self.domain_adapt1 = DomainAdaptationModule(6, 6)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        # 域自适应模块
        self.domain_adapt2 = DomainAdaptationModule(16, 16)
        
        # 分类器
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # 域分类器
        self.domain_classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )
        
    def forward(self, x, alpha=1.0, mode='train'):
        # 特征提取
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.domain_adapt1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.domain_adapt2(x)
        
        # 展平
        features = x.view(x.size(0), -1)
        
        if mode == 'train':
            # 梯度反转层
            reverse_features = GradientReversal.apply(features, alpha)
            domain_output = self.domain_classifier(reverse_features)
            class_output = self.classify(features)
            return class_output, domain_output
        else:
            return self.classify(features)
            
    def classify(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def get_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.domain_adapt1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.domain_adapt2(x)
        
        return x.view(x.size(0), -1)

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def test():
    model = DomainAdaptLeNet()
    
    # 模拟源域和目标域数据
    source_data = torch.randn(32, 1, 28, 28)
    target_data = torch.randn(32, 1, 28, 28)
    
    # 训练模式
    source_output, source_domain = model(source_data, mode='train')
    target_output, target_domain = model(target_data, mode='train')
    
    print("Source output shape:", source_output.shape)
    print("Source domain shape:", source_domain.shape)
    print("Target output shape:", target_output.shape)
    print("Target domain shape:", target_domain.shape)
    
    # 测试模式
    test_output = model(source_data, mode='test')
    print("Test output shape:", test_output.shape)

if __name__ == '__main__':
    test() 