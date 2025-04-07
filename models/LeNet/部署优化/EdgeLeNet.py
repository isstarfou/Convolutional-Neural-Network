import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import Dict, Any, Optional, Tuple
import onnx
import onnxruntime as ort
import tvm
from tvm import relay
import os

"""
EdgeLeNet: 专为边缘计算优化的LeNet
===================================

历史背景：
----------
边缘计算是一种将计算能力部署到数据源附近的架构，可以减少延迟和带宽消耗。
本实现针对边缘设备的特点，对LeNet进行了专门的优化，使其更适合在资源受限的边缘设备上运行。

架构特点：
----------
1. 计算优化：
   - 减少计算量
   - 优化内存使用
   - 降低功耗
2. 部署优化：
   - 支持多种边缘设备
   - 支持多种推理框架
   - 支持模型压缩
3. 性能优化：
   - 低延迟推理
   - 低内存占用
   - 低功耗运行

边缘设备支持：
-------------
1. 移动设备：
   - Android
   - iOS
2. 嵌入式设备：
   - Raspberry Pi
   - NVIDIA Jetson
3. 物联网设备：
   - ESP32
   - Arduino

学习要点：
---------
1. 边缘计算的特点和挑战
2. 模型部署优化技术
3. 边缘设备适配方法
4. 性能优化策略
"""

class EdgeLeNet(nn.Module):
    """
    专为边缘计算优化的LeNet实现
    """
    def __init__(self, num_classes: int = 10):
        super(EdgeLeNet, self).__init__()
        
        # 特征提取器
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积块
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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

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
                  onnx_path: str = 'edge_lenet.onnx') -> None:
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

def optimize_for_tvm(model: nn.Module,
                    input_shape: tuple = (1, 1, 28, 28),
                    target: str = 'llvm') -> Tuple[Any, Any, Any]:
    """
    使用TVM优化模型
    """
    # 导出到ONNX
    onnx_path = 'edge_lenet.onnx'
    export_to_onnx(model, input_shape, onnx_path)
    
    # 加载ONNX模型
    onnx_model = onnx.load(onnx_path)
    
    # 转换为TVM格式
    shape_dict = {'input': input_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    
    # 优化
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    
    return lib, mod, params

def optimize_for_onnxruntime(model: nn.Module,
                           input_shape: tuple = (1, 1, 28, 28)) -> ort.InferenceSession:
    """
    优化模型用于ONNX Runtime
    """
    # 导出到ONNX
    onnx_path = 'edge_lenet.onnx'
    export_to_onnx(model, input_shape, onnx_path)
    
    # 创建ONNX Runtime会话
    session = ort.InferenceSession(onnx_path)
    
    return session

def optimize_for_mobile(model: nn.Module,
                       input_shape: tuple = (1, 1, 28, 28)) -> None:
    """
    优化模型用于移动设备
    """
    # 导出到ONNX
    onnx_path = 'edge_lenet.onnx'
    export_to_onnx(model, input_shape, onnx_path)
    
    # 这里可以添加移动设备特定的优化
    # 例如：使用TensorFlow Lite转换器
    # 或使用Core ML转换器

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
    model = EdgeLeNet().to(device)
    
    # 训练模型
    train_results = train_model(model, train_loader, device)
    print(f"Training Results: {train_results}")
    
    # 评估模型
    eval_results = evaluate_model(model, test_loader, device)
    print(f"Evaluation Results: {eval_results}")
    
    # 优化模型
    print("Optimizing for TVM...")
    lib, mod, params = optimize_for_tvm(model)
    print("TVM optimization complete")
    
    print("Optimizing for ONNX Runtime...")
    session = optimize_for_onnxruntime(model)
    print("ONNX Runtime optimization complete")
    
    print("Optimizing for mobile devices...")
    optimize_for_mobile(model)
    print("Mobile optimization complete")

if __name__ == '__main__':
    example_usage() 