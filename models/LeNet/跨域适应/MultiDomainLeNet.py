import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainSpecificBatchNorm(nn.Module):
    def __init__(self, num_features, num_domains):
        super(DomainSpecificBatchNorm, self).__init__()
        self.num_domains = num_domains
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(num_features) for _ in range(num_domains)
        ])
        
    def forward(self, x, domain_id):
        out = torch.zeros_like(x)
        for i in range(self.num_domains):
            mask = (domain_id == i)
            if mask.any():
                out[mask] = self.bns[i](x[mask])
        return out

class MultiDomainLeNet(nn.Module):
    def __init__(self, num_classes=10, num_domains=3):
        super(MultiDomainLeNet, self).__init__()
        
        self.num_domains = num_domains
        
        # 特征提取器
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.bn1 = DomainSpecificBatchNorm(6, num_domains)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.bn2 = DomainSpecificBatchNorm(16, num_domains)
        self.pool2 = nn.MaxPool2d(2)
        
        # 域特定分类器
        self.domain_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(16 * 7 * 7, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, num_classes)
            ) for _ in range(num_domains)
        ])
        
        # 域判别器
        self.domain_discriminator = nn.Sequential(
            nn.Linear(16 * 7 * 7, 100),
            nn.ReLU(),
            nn.Linear(100, num_domains)
        )
        
    def forward(self, x, domain_id=None, mode='train'):
        # 特征提取
        x = self.conv1(x)
        if domain_id is not None:
            x = self.bn1(x, domain_id)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        if domain_id is not None:
            x = self.bn2(x, domain_id)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 展平
        features = x.view(x.size(0), -1)
        
        if mode == 'train':
            if domain_id is None:
                raise ValueError("Domain ID must be provided in training mode")
                
            # 使用域特定分类器
            outputs = []
            for i in range(self.num_domains):
                mask = (domain_id == i)
                if mask.any():
                    outputs.append(self.domain_classifiers[i](features[mask]))
                else:
                    outputs.append(None)
                    
            # 域判别
            domain_output = self.domain_discriminator(features)
            
            return outputs, domain_output
            
        else:
            # 测试模式：使用所有域分类器的平均
            outputs = []
            for classifier in self.domain_classifiers:
                outputs.append(classifier(features))
            return torch.stack(outputs).mean(dim=0)
            
    def get_features(self, x, domain_id=None):
        """获取特征表示"""
        x = self.conv1(x)
        if domain_id is not None:
            x = self.bn1(x, domain_id)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        if domain_id is not None:
            x = self.bn2(x, domain_id)
        x = F.relu(x)
        x = self.pool2(x)
        
        return x.view(x.size(0), -1)

def test():
    model = MultiDomainLeNet(num_domains=3)
    
    # 模拟输入
    x = torch.randn(32, 1, 28, 28)
    domain_ids = torch.randint(0, 3, (32,))
    
    # 训练模式
    outputs, domain_output = model(x, domain_ids, mode='train')
    print("Number of domain outputs:", len(outputs))
    print("Domain output shape:", domain_output.shape)
    
    # 测试模式
    test_output = model(x, mode='test')
    print("Test output shape:", test_output.shape)

if __name__ == '__main__':
    test() 