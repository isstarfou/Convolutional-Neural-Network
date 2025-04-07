import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(LeNetFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        return x

class LeNetCNNRNN(nn.Module):
    def __init__(self, num_classes=10, hidden_size=128, num_layers=2):
        super(LeNetCNNRNN, self).__init__()
        
        # CNN特征提取器
        self.feature_extractor = LeNetFeatureExtractor()
        
        # RNN层
        self.rnn = nn.LSTM(
            input_size=16 * 7 * 7,  # LeNet特征图大小
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 时序池化
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头
        self.fc1 = nn.Linear(hidden_size * 2, 84)  # 双向LSTM
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 重塑输入以处理序列
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        
        # 提取特征
        features = self.feature_extractor(x)
        
        # 重塑特征以输入RNN
        features = features.view(batch_size, seq_len, -1)
        
        # RNN处理
        rnn_out, _ = self.rnn(features)
        
        # 时序池化
        pooled = self.temporal_pool(rnn_out.transpose(1, 2))
        pooled = pooled.squeeze(-1)
        
        # 分类
        x = F.relu(self.fc1(pooled))
        x = self.fc2(x)
        
        return x

def test():
    model = LeNetCNNRNN()
    # 模拟输入：batch_size=2, sequence_length=5, channels=1, height=28, width=28
    x = torch.randn(2, 5, 1, 28, 28)
    y = model(x)
    print(y.shape)

if __name__ == '__main__':
    test() 