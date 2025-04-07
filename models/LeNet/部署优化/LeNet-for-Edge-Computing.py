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
import tvm
from tvm import relay
import os

"""
LeNet-for-Edge-Computing: 专为边缘计算优化的LeNet
================================================

历史背景：
----------
随着边缘计算的发展，越来越多的深度学习模型需要在资源受限的设备上运行。
本实现针对边缘计算场景进行了专门优化，在保持模型性能的同时降低计算和存储需求。

架构特点：
----------
1. 轻量化设计：
   - 减少参数量
   - 优化计算量
   - 降低内存占用
2. 部署优化：
   - 支持多种边缘设备
   - 支持多种推理框架
   - 支持模型压缩
3. 性能优化：
   - 低延迟推理
   - 低功耗设计
   - 资源高效利用

边缘设备支持：
-------------
1. 移动设备：
   - 智能手机
   - 平板电脑
2. 嵌入式设备：
   - 树莓派
   - Jetson系列
3. 边缘服务器：
   - 边缘网关
   - 边缘计算节点

学习要点：
---------
1. 边缘计算的特点和挑战
2. 模型轻量化技术
3. 部署优化方法
4. 性能评估指标
"""

class EdgeLeNet(nn.Module):
    """
    专为边缘计算优化的LeNet实现
    """
    def __init__(self, num_classes: int = 10):
        super(EdgeLeNet, self).__init__()
        
        # 轻量化特征提取器
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 4, kernel_size=3, padding=1),  # 减少通道数
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积块
            nn.Conv2d(4, 8, kernel_size=3, padding=1),  # 减少通道数
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 轻量化分类器
        self.classifier = nn.Sequential(
            nn.Linear(8 * 7 * 7, 32),  # 减少神经元数量
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
    训练模型，针对边缘计算场景优化
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
    评估模型性能
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
                    target: str = 'llvm') -> Any:
    """
    使用TVM优化模型
    """
    # 转换为TVM格式
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data)
    
    # 转换为Relay IR
    shape_dict = {'input': input_shape}
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_dict)
    
    # 优化
    target = tvm.target.Target(target)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    
    return lib

def optimize_for_onnxruntime(model: nn.Module,
                           input_shape: tuple = (1, 1, 28, 28)) -> ort.InferenceSession:
    """
    优化模型用于ONNX Runtime
    """
    # 导出到ONNX
    onnx_path = 'edge_lenet_optimized.onnx'
    export_to_onnx(model, input_shape, onnx_path)
    
    # 创建优化选项
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    
    # 创建推理会话
    session = ort.InferenceSession(onnx_path, sess_options)
    
    return session

def optimize_for_mobile(model: nn.Module,
                       input_shape: tuple = (1, 1, 28, 28)) -> None:
    """
    优化模型用于移动设备
    """
    # 量化模型
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model)
    model_quantized = torch.quantization.convert(model_prepared)
    
    # 导出量化模型
    torch.save(model_quantized.state_dict(), 'edge_lenet_quantized.pth')
    
    # 导出到TorchScript
    scripted_model = torch.jit.script(model_quantized)
    scripted_model.save('edge_lenet_scripted.pt')

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
    print("\nOptimizing model for different platforms...")
    
    # TVM优化
    tvm_lib = optimize_for_tvm(model)
    print("TVM optimization completed")
    
    # ONNX Runtime优化
    ort_session = optimize_for_onnxruntime(model)
    print("ONNX Runtime optimization completed")
    
    # 移动设备优化
    optimize_for_mobile(model)
    print("Mobile optimization completed")

if __name__ == '__main__':
    example_usage()
