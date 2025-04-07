import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
import time
import numpy as np
import matplotlib.pyplot as plt

"""
QuantizedLeNet：LeNet的量化版本实现
==================================

核心特点：
----------
1. 支持动态量化(Dynamic Quantization)
2. 支持静态量化(Static Quantization)
3. 支持量化感知训练(Quantization-Aware Training)
4. 实现不同位宽的量化(8-bit, 4-bit)
5. 分析量化对模型精度和性能的影响

量化原理：
---------
量化是将浮点数模型转换为低精度整数模型的过程，
主要目的是减小模型大小和提高推理速度，尤其在边缘设备上。
量化过程包括：
1. 确定量化范围(校准)
2. 将浮点值映射到整数值
3. 在整数域进行计算
4. 将结果反量化回浮点域

量化方法：
---------
1. 动态量化：推理时对权重进行量化，激活值保持浮点
2. 静态量化：对权重和激活值进行预先量化，需要校准数据
3. 量化感知训练：训练过程中模拟量化效果，减少精度损失

实现目标：
---------
1. 理解不同量化技术的原理与应用场景
2. 掌握量化对模型大小和推理速度的影响
3. 分析量化对模型精度的影响并提供优化方法
4. 学习如何在部署时应用量化模型
"""

class QuantizableLeNet(nn.Module):
    """LeNet的可量化实现版本"""
    def __init__(self, num_classes=10):
        super(QuantizableLeNet, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        
        self.fc3 = nn.Linear(84, num_classes)
        
        # 量化层
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        """前向传播"""
        # 量化输入
        x = self.quant(x)
        
        # 第一个卷积块
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu3(x)
        
        x = self.fc2(x)
        x = self.relu4(x)
        
        x = self.fc3(x)
        
        # 反量化输出
        x = self.dequant(x)
        
        return x
    
    def fuse_model(self):
        """融合操作，将卷积+BN+ReLU合并为一个运算，提高量化性能"""
        torch.quantization.fuse_modules(self, ['conv1', 'relu1'], inplace=True)
        torch.quantization.fuse_modules(self, ['conv2', 'relu2'], inplace=True)
        torch.quantization.fuse_modules(self, ['fc1', 'relu3'], inplace=True)
        torch.quantization.fuse_modules(self, ['fc2', 'relu4'], inplace=True)
        print("Model fused.")

# 普通不可量化的LeNet，用于对比
class StandardLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(StandardLeNet, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        
        # 池化层
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        """前向传播"""
        # 第一个卷积块
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # 第二个卷积块
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def prepare_for_quantization(model, dtype='static'):
    """准备模型进行量化"""
    # 使用默认的量化配置
    if dtype == 'static':
        # 静态量化
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    elif dtype == 'dynamic':
        # 动态量化
        model.qconfig = torch.quantization.default_dynamic_qconfig
    elif dtype == 'qat':
        # 量化感知训练
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # 准备量化
    torch.quantization.prepare(model, inplace=True)
    print(f"Model prepared for {dtype} quantization with qconfig: {model.qconfig}")

def calibrate_model(model, calib_data_loader):
    """使用校准数据集校准模型（用于静态量化）"""
    model.eval()
    with torch.no_grad():
        for inputs, _ in calib_data_loader:
            model(inputs)
    print("Model calibrated.")

def convert_to_quantized(model):
    """将模型转换为量化版本"""
    torch.quantization.convert(model, inplace=True)
    print("Model converted to quantized version.")

def quantize_weight_only(model):
    """仅量化模型权重，不量化激活（动态量化）"""
    model_quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    print("Model weights quantized.")
    return model_quantized

def compare_models(fp32_model, int8_model, int4_model=None, test_input=None):
    """比较不同精度模型的推理性能和准确性"""
    if test_input is None:
        test_input = torch.randn(1, 1, 28, 28)
    
    models = {
        'FP32': fp32_model,
        'INT8': int8_model
    }
    
    if int4_model is not None:
        models['INT4'] = int4_model
    
    # 比较模型大小
    model_sizes = {}
    for name, model in models.items():
        size_bytes = 0
        for param in model.parameters():
            size_bytes += param.nelement() * param.element_size()
        model_sizes[name] = size_bytes / 1024  # 转换为KB
    
    # 比较推理时间
    inference_times = {}
    results = {}
    
    for name, model in models.items():
        model.eval()
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = model(test_input)
        
        # 计时
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                output = model(test_input)
        end_time = time.time()
        
        inference_times[name] = (end_time - start_time) / 100 * 1000  # 转换为毫秒
        results[name] = output
    
    # 比较输出差异
    if 'FP32' in results and 'INT8' in results:
        output_diff_int8 = torch.abs(results['FP32'] - results['INT8']).mean().item()
    else:
        output_diff_int8 = None
    
    if 'FP32' in results and 'INT4' in results:
        output_diff_int4 = torch.abs(results['FP32'] - results['INT4']).mean().item()
    else:
        output_diff_int4 = None
    
    # 打印结果
    print("\n模型比较结果:")
    print("-" * 40)
    
    print("模型大小:")
    for name, size in model_sizes.items():
        print(f"  {name}: {size:.2f} KB")
    
    print("\n推理时间:")
    for name, time_ms in inference_times.items():
        print(f"  {name}: {time_ms:.3f} ms")
    
    print("\n输出差异 (平均绝对误差):")
    if output_diff_int8 is not None:
        print(f"  FP32 vs INT8: {output_diff_int8:.6f}")
    if output_diff_int4 is not None:
        print(f"  FP32 vs INT4: {output_diff_int4:.6f}")
    
    # 绘制对比图
    plt.figure(figsize=(15, 5))
    
    # 模型大小对比
    plt.subplot(1, 3, 1)
    plt.bar(model_sizes.keys(), model_sizes.values())
    plt.title('Model Size (KB)')
    plt.ylabel('Size (KB)')
    plt.grid(axis='y', alpha=0.3)
    
    # 推理时间对比
    plt.subplot(1, 3, 2)
    plt.bar(inference_times.keys(), inference_times.values())
    plt.title('Inference Time (ms)')
    plt.ylabel('Time (ms)')
    plt.grid(axis='y', alpha=0.3)
    
    # 速度和大小提升倍数
    plt.subplot(1, 3, 3)
    speedup = inference_times['FP32'] / inference_times['INT8']
    size_reduction = model_sizes['FP32'] / model_sizes['INT8']
    
    if 'INT4' in inference_times:
        speedup_int4 = inference_times['FP32'] / inference_times['INT4']
        size_reduction_int4 = model_sizes['FP32'] / model_sizes['INT4']
        
        metrics = ['INT8 Speedup', 'INT8 Size Reduction', 
                  'INT4 Speedup', 'INT4 Size Reduction']
        values = [speedup, size_reduction, speedup_int4, size_reduction_int4]
    else:
        metrics = ['INT8 Speedup', 'INT8 Size Reduction']
        values = [speedup, size_reduction]
    
    plt.bar(metrics, values)
    plt.title('Performance Improvements')
    plt.ylabel('Factor (x)')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return model_sizes, inference_times

def visualize_quantization_effects(fp32_model, int8_model):
    """可视化量化对权重和激活值分布的影响"""
    # 提取权重
    fp32_weights = []
    int8_weights = []
    
    # 获取FP32模型权重
    for name, param in fp32_model.named_parameters():
        if 'weight' in name:
            fp32_weights.append(param.detach().cpu().numpy().flatten())
    
    # 获取INT8模型权重（需要转换回浮点数）
    for name, param in int8_model.named_parameters():
        if 'weight' in name:
            # 这里简化处理，假设量化后的参数仍然可以通过.numpy()获取
            int8_weights.append(param.detach().cpu().numpy().flatten())
    
    # 绘制权重分布对比
    plt.figure(figsize=(15, 5))
    
    # FP32权重分布
    plt.subplot(1, 2, 1)
    for i, weights in enumerate(fp32_weights):
        plt.hist(weights, bins=50, alpha=0.5, label=f'Layer {i+1}')
    plt.title('FP32 Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # INT8权重分布
    plt.subplot(1, 2, 2)
    for i, weights in enumerate(int8_weights):
        plt.hist(weights, bins=50, alpha=0.5, label=f'Layer {i+1}')
    plt.title('INT8 Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def test():
    """测试量化LeNet的各种实现"""
    # 创建模型
    fp32_model = QuantizableLeNet()
    standard_model = StandardLeNet()
    
    # 创建随机输入
    x = torch.randn(1, 1, 28, 28)
    
    # 测试FP32模型
    y_fp32 = fp32_model(x)
    y_std = standard_model(x)
    print(f"FP32模型输出形状: {y_fp32.shape}")
    print(f"标准模型输出形状: {y_std.shape}")
    
    # 准备模型进行量化
    quantized_model = QuantizableLeNet()
    
    # 融合模型
    quantized_model.fuse_model()
    
    # 执行静态量化
    prepare_for_quantization(quantized_model, 'static')
    
    # 创建模拟校准数据集
    calib_data = [(torch.randn(1, 1, 28, 28), torch.tensor([0])) for _ in range(10)]
    calib_loader = torch.utils.data.DataLoader(calib_data, batch_size=1)
    
    # 校准模型
    calibrate_model(quantized_model, calib_loader)
    
    # 转换为量化模型
    convert_to_quantized(quantized_model)
    
    # 测试INT8模型
    y_int8 = quantized_model(x)
    print(f"INT8模型输出形状: {y_int8.shape}")
    
    # 执行动态量化（仅权重量化）
    dynamic_model = QuantizableLeNet()
    dynamic_quantized = quantize_weight_only(dynamic_model)
    
    # 测试动态量化模型
    y_dynamic = dynamic_quantized(x)
    print(f"动态量化模型输出形状: {y_dynamic.shape}")
    
    # 比较模型性能和准确性
    print("\n比较模型性能和准确性...")
    model_sizes, inference_times = compare_models(fp32_model, quantized_model, dynamic_quantized, x)
    
    # 可视化量化效果
    print("\n可视化量化对权重分布的影响...")
    visualize_quantization_effects(fp32_model, quantized_model)
    
    # 打印量化信息
    print("\n量化信息概要:")
    print(f"FP32模型大小: {model_sizes['FP32']:.2f} KB")
    print(f"INT8模型大小: {model_sizes['INT8']:.2f} KB")
    print(f"压缩比: {model_sizes['FP32'] / model_sizes['INT8']:.2f}x")
    print(f"FP32推理时间: {inference_times['FP32']:.3f} ms")
    print(f"INT8推理时间: {inference_times['INT8']:.3f} ms")
    print(f"加速比: {inference_times['FP32'] / inference_times['INT8']:.2f}x")

if __name__ == '__main__':
    test() 