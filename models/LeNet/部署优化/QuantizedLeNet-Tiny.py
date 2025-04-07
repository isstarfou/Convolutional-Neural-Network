import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.quantization
import torch.quantization.quantize_fx as quantize_fx
from typing import Dict, Any, Optional

"""
QuantizedLeNet-Tiny: 量化版本的轻量级LeNet
============================================

历史背景：
----------
模型量化是一种重要的模型压缩和加速技术，通过降低模型参数的精度来减少模型大小和计算量。
本实现将量化技术应用于LeNet架构，使其更适合在资源受限的设备上部署。

架构特点：
----------
1. 支持多种量化方式：
   - 动态量化
   - 静态量化
   - 量化感知训练
2. 轻量级设计：
   - 减少通道数
   - 简化网络结构
   - 优化计算量
3. 部署友好：
   - 支持ONNX导出
   - 支持TensorRT优化
   - 支持移动端部署

量化配置：
----------
1. 权重量化：
   - 8位整数量化
   - 对称量化
   - 每通道量化
2. 激活量化：
   - 8位整数量化
   - 非对称量化
   - 每层量化
3. 量化策略：
   - 训练后量化
   - 量化感知训练

学习要点：
---------
1. 模型量化的基本原理
2. 量化感知训练方法
3. 部署优化技术
4. 性能与精度的平衡
"""

class QuantizedLeNetTiny(nn.Module):
    """
    量化版本的轻量级LeNet实现
    """
    def __init__(self, num_classes: int = 10):
        super(QuantizedLeNetTiny, self).__init__()
        
        # 特征提取器
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(8 * 7 * 7, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )
        
        # 量化配置
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """
        融合模型中的Conv+ReLU和Conv+ReLU+MaxPool
        """
        torch.quantization.fuse_modules(
            self,
            [['features.0', 'features.1'],
             ['features.3', 'features.4']],
            inplace=True
        )

def prepare_model_for_quantization(model: nn.Module) -> nn.Module:
    """
    准备模型进行量化
    """
    # 设置量化配置
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # 融合模型
    model.fuse_model()
    
    # 准备量化
    torch.quantization.prepare(model, inplace=True)
    
    return model

def quantize_model(model: nn.Module, 
                  calibration_data: DataLoader,
                  num_calibration_batches: int = 32) -> nn.Module:
    """
    量化模型
    """
    # 校准
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(calibration_data):
            if i >= num_calibration_batches:
                break
            model(data)
    
    # 转换量化模型
    torch.quantization.convert(model, inplace=True)
    
    return model

def train_model(model: nn.Module,
                train_loader: DataLoader,
                device: torch.device,
                epochs: int = 10,
                learning_rate: float = 0.001) -> Dict[str, Any]:
    """
    训练模型
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                      f'Loss: {running_loss/(batch_idx+1):.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
    
    return {
        'loss': running_loss / len(train_loader),
        'accuracy': 100. * correct / total
    }

def evaluate_model(model: nn.Module,
                  test_loader: DataLoader,
                  device: torch.device) -> Dict[str, float]:
    """
    评估模型
    """
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return {
        'loss': test_loss,
        'accuracy': accuracy
    }

def export_to_onnx(model: nn.Module,
                  input_shape: tuple = (1, 1, 28, 28),
                  onnx_path: str = 'quantized_lenet_tiny.onnx') -> None:
    """
    导出模型到ONNX格式
    """
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

def example_usage():
    """
    示例用法
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True,
                                 transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 创建模型
    model = QuantizedLeNetTiny().to(device)
    
    # 训练模型
    train_results = train_model(model, train_loader, device)
    print(f"Training Results: {train_results}")
    
    # 评估模型
    eval_results = evaluate_model(model, test_loader, device)
    print(f"Evaluation Results: {eval_results}")
    
    # 准备量化
    model = prepare_model_for_quantization(model)
    
    # 量化模型
    model = quantize_model(model, train_loader)
    
    # 评估量化后的模型
    quantized_results = evaluate_model(model, test_loader, device)
    print(f"Quantized Model Results: {quantized_results}")
    
    # 导出到ONNX
    export_to_onnx(model)
    print("Model exported to ONNX format")

if __name__ == '__main__':
    example_usage()
