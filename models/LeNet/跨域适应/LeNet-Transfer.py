import torch
import torch.nn as nn
import torch.nn.functional as F

class TransferLeNet(nn.Module):
    def __init__(self, num_classes=10, freeze_features=True):
        super(TransferLeNet, self).__init__()
        
        # 特征提取器
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
        
        # 是否冻结特征提取器
        if freeze_features:
            for param in self.features.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        
    def unfreeze_features(self):
        """解冻特征提取器参数"""
        for param in self.features.parameters():
            param.requires_grad = True
            
    def freeze_features(self):
        """冻结特征提取器参数"""
        for param in self.features.parameters():
            param.requires_grad = False
            
    def get_features(self, x):
        """获取特征表示"""
        x = self.features(x)
        return x.view(x.size(0), -1)
        
    def fine_tune(self, new_num_classes):
        """微调模型以适应新的类别数"""
        # 保存原始分类器的权重
        old_classifier = self.classifier
        
        # 创建新的分类器
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, new_num_classes)
        )
        
        # 如果可能，迁移部分权重
        if new_num_classes > old_classifier[-1].out_features:
            with torch.no_grad():
                # 迁移前几层的权重
                for i in range(len(self.classifier) - 1):
                    if isinstance(self.classifier[i], nn.Linear):
                        self.classifier[i].weight.copy_(old_classifier[i].weight)
                        self.classifier[i].bias.copy_(old_classifier[i].bias)

def test():
    # 创建预训练模型
    pretrained_model = TransferLeNet(num_classes=10, freeze_features=True)
    
    # 模拟输入
    x = torch.randn(32, 1, 28, 28)
    
    # 测试前向传播
    y = pretrained_model(x)
    print("Pretrained output shape:", y.shape)
    
    # 微调模型
    pretrained_model.fine_tune(new_num_classes=20)
    y = pretrained_model(x)
    print("Fine-tuned output shape:", y.shape)
    
    # 解冻特征提取器
    pretrained_model.unfreeze_features()
    y = pretrained_model(x)
    print("Unfrozen output shape:", y.shape)

if __name__ == '__main__':
    test() 