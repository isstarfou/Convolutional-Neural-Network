import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=256):
        super(DomainDiscriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class CrossDomainLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CrossDomainLeNet, self).__init__()
        
        # 特征提取器
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        # 特征投影
        self.feature_projection = nn.Sequential(
            nn.Linear(16 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
        
        # 域判别器
        self.domain_discriminator = DomainDiscriminator()
        
    def forward(self, x, alpha=1.0, mode='train'):
        # 特征提取
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 特征投影
        features = self.feature_projection(x)
        
        if mode == 'train':
            # 梯度反转层
            reverse_features = GradientReversal.apply(features, alpha)
            domain_output = self.domain_discriminator(reverse_features)
            class_output = self.classifier(features)
            return class_output, domain_output
        else:
            # 测试模式只返回分类结果
            return self.classifier(features)
        
    def get_features(self, x):
        # 获取特征表示
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        return self.feature_projection(x)

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def test():
    model = CrossDomainLeNet()
    
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
