import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import onnx
import onnxruntime as ort
from torch.cuda.amp import autocast, GradScaler

"""
LeNet-FP16orFP64: 支持FP16和FP64精度的LeNet
===========================================

历史背景：
----------
随着深度学习模型规模的增大，计算精度和效率之间的平衡变得越来越重要。
本实现支持FP16和FP64两种精度，可以根据需求选择最适合的精度级别。

架构特点：
----------
1. 精度支持：
   - FP16：半精度浮点数，节省内存和计算资源
   - FP64：双精度浮点数，提供更高的计算精度
2. 混合精度训练：
   - 自动混合精度(AMP)
   - 梯度缩放
   - 精度损失控制
3. 部署优化：
   - 支持多种精度级别
   - 支持多种推理框架
   - 支持模型转换

精度选择：
----------
1. FP16：
   - 优点：内存占用小，计算速度快
   - 缺点：精度较低，可能出现数值不稳定
   - 适用场景：移动设备、边缘计算
2. FP64：
   - 优点：精度高，数值稳定
   - 缺点：内存占用大，计算速度慢
   - 适用场景：科学计算、高精度需求

学习要点：
---------
1. 浮点数精度的基本原理
2. 混合精度训练方法
3. 精度与性能的平衡
4. 部署优化技术
"""

class LeNetPrecision(nn.Module):
    """
    支持FP16和FP64精度的LeNet实现
    """
    def __init__(self, num_classes: int = 10, precision: str = 'fp32'):
        super(LeNetPrecision, self).__init__()
        self.precision = precision
        
        # 特征提取器
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积块
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
        
        # 设置精度
        self._set_precision()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _set_precision(self):
        """
        设置模型精度
        """
        if self.precision == 'fp16':
            self.half()
        elif self.precision == 'fp64':
            self.double()
        else:  # fp32
            self.float()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model(model: nn.Module,
                train_loader: DataLoader,
                device: torch.device,
                epochs: int = 10,
                learning_rate: float = 0.001,
                use_amp: bool = True) -> Dict[str, Any]:
    """
    训练模型，支持混合精度训练
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 创建梯度缩放器
    scaler = GradScaler(enabled=use_amp)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # 使用自动混合精度
            with autocast(enabled=use_amp):
                output = model(data)
                loss = criterion(output, target)
            
            # 使用梯度缩放器
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
                  device: torch.device,
                  use_amp: bool = True) -> Dict[str, float]:
    """
    评估模型，支持混合精度推理
    """
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # 使用自动混合精度
            with autocast(enabled=use_amp):
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
                  onnx_path: str = 'lenet_precision.onnx',
                  precision: str = 'fp32') -> None:
    """
    导出模型到ONNX格式，支持不同精度
    """
    # 设置模型精度
    if precision == 'fp16':
        model.half()
        dummy_input = torch.randn(input_shape, dtype=torch.float16)
    elif precision == 'fp64':
        model.double()
        dummy_input = torch.randn(input_shape, dtype=torch.float64)
    else:  # fp32
        model.float()
        dummy_input = torch.randn(input_shape, dtype=torch.float32)
    
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
    
    # 测试不同精度
    precisions = ['fp16', 'fp32', 'fp64']
    for precision in precisions:
        print(f"\nTesting {precision} precision...")
        
        # 创建模型
        model = LeNetPrecision(precision=precision).to(device)
        
        # 训练模型
        train_results = train_model(model, train_loader, device, use_amp=(precision == 'fp16'))
        print(f"Training Results ({precision}): {train_results}")
        
        # 评估模型
        eval_results = evaluate_model(model, test_loader, device, use_amp=(precision == 'fp16'))
        print(f"Evaluation Results ({precision}): {eval_results}")
        
        # 导出到ONNX
        export_to_onnx(model, precision=precision)
        print(f"Model exported to ONNX format ({precision})")

if __name__ == '__main__':
    example_usage() 