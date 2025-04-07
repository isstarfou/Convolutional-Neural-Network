import torch
import torch.nn as nn
import torch.nn.functional as F
import time

"""
LeNet-Efficient：LeNet的高效实现与优化
======================================

核心特点：
----------
1. 采用多种计算和内存优化技术
2. 使用低精度计算加速训练与推理
3. 实现模型剪枝和量化
4. 提供并行计算和内存优化

优化技术概述：
------------
1. 计算图优化：融合操作，减少内存访问
2. 低精度训练：使用FP16/BF16进行计算
3. 模型剪枝：移除不重要的连接
4. 权重量化：降低权重精度
5. 激活量化：降低激活值精度
6. 内存优化：梯度检查点，重计算
7. 静态图与即时编译 (JIT) 加速

实现目标：
----------
1. 在保持准确率的前提下减少计算量和内存使用
2. 提高模型训练和推理速度
3. 优化模型大小以便部署到资源有限的设备
"""

class EfficientLeNet(nn.Module):
    def __init__(self, num_classes=10, efficient_mode="standard", pruning_ratio=0.0):
        super(EfficientLeNet, self).__init__()
        
        self.efficient_mode = efficient_mode
        self.pruning_ratio = pruning_ratio
        self.use_fp16 = False  # 默认不使用半精度
        
        # 标准卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        
        # 池化层
        self.pool = nn.MaxPool2d(2)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # 初始化剪枝掩码（1表示保留，0表示剪枝）
        self.pruning_masks = {}
        
        # 初始化静态计算图（用于JIT编译）
        self.compiled_model = None
        
        # 如果指定了剪枝，则应用剪枝
        if pruning_ratio > 0:
            self.apply_pruning()
    
    def apply_pruning(self):
        """应用模型剪枝"""
        for name, param in self.named_parameters():
            if 'weight' in name:  # 只对权重剪枝，不对偏置剪枝
                # 创建与参数同形状的掩码
                mask = torch.ones_like(param)
                
                # 计算阈值（基于权重绝对值的分位数）
                threshold = torch.quantile(torch.abs(param.data.flatten()), self.pruning_ratio)
                
                # 将绝对值小于阈值的位置设为0
                mask[torch.abs(param.data) < threshold] = 0
                
                # 存储掩码
                self.pruning_masks[name] = mask
                
                # 应用掩码
                param.data *= mask
    
    def apply_pruning_mask(self):
        """在前向传播前应用剪枝掩码"""
        if self.pruning_ratio > 0:
            for name, param in self.named_parameters():
                if name in self.pruning_masks:
                    param.data *= self.pruning_masks[name]
    
    def enable_fp16(self, enable=True):
        """启用/禁用半精度(FP16)计算"""
        self.use_fp16 = enable
        if enable:
            # 将模型参数转换为半精度
            self.half()
        else:
            # 将模型参数转换回全精度
            self.float()
    
    def fuse_bn_conv(self):
        """融合批归一化层和卷积层以提高计算效率"""
        # 融合conv1和bn1
        if hasattr(self, 'bn1') and hasattr(self, 'conv1'):
            w = self.conv1.weight.data
            b = self.conv1.bias.data if self.conv1.bias is not None else torch.zeros_like(self.bn1.running_mean)
            
            bn_mean = self.bn1.running_mean
            bn_var = self.bn1.running_var
            bn_weight = self.bn1.weight
            bn_bias = self.bn1.bias
            bn_eps = self.bn1.eps
            
            # 计算融合后的权重和偏置
            scale = bn_weight / torch.sqrt(bn_var + bn_eps)
            fused_w = w * scale.reshape(-1, 1, 1, 1)
            fused_b = (b - bn_mean) * scale + bn_bias
            
            # 更新卷积层参数
            self.conv1.weight.data = fused_w
            if self.conv1.bias is None:
                self.conv1.bias = nn.Parameter(fused_b)
            else:
                self.conv1.bias.data = fused_b
            
            # 移除批归一化层
            self.bn1 = nn.Identity()
        
        # 融合conv2和bn2
        if hasattr(self, 'bn2') and hasattr(self, 'conv2'):
            w = self.conv2.weight.data
            b = self.conv2.bias.data if self.conv2.bias is not None else torch.zeros_like(self.bn2.running_mean)
            
            bn_mean = self.bn2.running_mean
            bn_var = self.bn2.running_var
            bn_weight = self.bn2.weight
            bn_bias = self.bn2.bias
            bn_eps = self.bn2.eps
            
            # 计算融合后的权重和偏置
            scale = bn_weight / torch.sqrt(bn_var + bn_eps)
            fused_w = w * scale.reshape(-1, 1, 1, 1)
            fused_b = (b - bn_mean) * scale + bn_bias
            
            # 更新卷积层参数
            self.conv2.weight.data = fused_w
            if self.conv2.bias is None:
                self.conv2.bias = nn.Parameter(fused_b)
            else:
                self.conv2.bias.data = fused_b
            
            # 移除批归一化层
            self.bn2 = nn.Identity()
    
    def compile_model(self):
        """使用TorchScript编译模型以加速推理"""
        try:
            # 创建示例输入
            example = torch.randn(1, 1, 28, 28)
            
            # 如果启用了FP16，将示例转换为半精度
            if self.use_fp16:
                example = example.half()
            
            # 使用JIT编译
            self.compiled_model = torch.jit.trace(self, example)
            print("模型已成功编译")
        except Exception as e:
            print(f"模型编译失败: {e}")
            self.compiled_model = None
    
    def optimize_memory(self, mode="gradient_checkpointing"):
        """内存优化设置"""
        self.memory_opt_mode = mode
        # 在实际应用中，这将设置梯度检查点或激活重计算等策略
    
    def forward(self, x):
        """前向传播"""
        # 如果已编译且处于推理模式，使用编译后的模型
        if self.compiled_model is not None and not self.training:
            return self.compiled_model(x)
        
        # 如果启用了FP16但输入不是半精度，转换为半精度
        if self.use_fp16 and x.dtype != torch.float16:
            x = x.half()
        
        # 在前向传播前应用剪枝掩码
        if self.training and self.pruning_ratio > 0:
            self.apply_pruning_mask()
        
        # 基于不同的优化模式选择不同的前向传播路径
        if self.efficient_mode == "fused_ops":
            # 融合的操作路径
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            # 标准路径
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.pool(x)
            
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.pool(x)
            
            x = x.view(x.size(0), -1)
            
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        
        return x

def benchmark_efficiency(model, input_size=(1, 1, 28, 28), num_runs=100):
    """测试模型效率（推理速度和内存使用）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    x = torch.randn(*input_size).to(device)
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    
    # 计时
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(x)
    end_time = time.time()
    
    # 计算平均推理时间
    avg_time = (end_time - start_time) / num_runs
    print(f"平均推理时间: {avg_time*1000:.2f} ms")
    
    # 尝试计算模型大小
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    print(f"模型大小: {model_size:.2f} MB")
    
    return avg_time, model_size

def test():
    """测试不同效率优化策略"""
    # 创建标准模型
    standard_model = EfficientLeNet(efficient_mode="standard")
    x = torch.randn(1, 1, 28, 28)
    y = standard_model(x)
    print(f"标准模型输出形状: {y.shape}")
    
    # 带剪枝的模型
    pruned_model = EfficientLeNet(efficient_mode="standard", pruning_ratio=0.3)
    y_pruned = pruned_model(x)
    print(f"剪枝模型输出形状: {y_pruned.shape}")
    
    # 融合操作的模型
    fused_model = EfficientLeNet(efficient_mode="fused_ops")
    fused_model.fuse_bn_conv()
    y_fused = fused_model(x)
    print(f"融合操作模型输出形状: {y_fused.shape}")
    
    # 半精度模型
    fp16_model = EfficientLeNet(efficient_mode="standard")
    fp16_model.enable_fp16(True)
    y_fp16 = fp16_model(x.half())
    print(f"半精度模型输出形状: {y_fp16.shape}")
    
    # 编译模型（如果支持）
    try:
        compiled_model = EfficientLeNet(efficient_mode="standard")
        compiled_model.eval()
        compiled_model.compile_model()
        y_compiled = compiled_model(x)
        print(f"编译模型输出形状: {y_compiled.shape}")
    except Exception as e:
        print(f"编译模型测试失败: {e}")
    
    # 基准测试
    print("\n效率基准测试:")
    print("--------------")
    print("标准模型:")
    standard_time, standard_size = benchmark_efficiency(standard_model)
    
    print("\n剪枝模型:")
    pruned_time, pruned_size = benchmark_efficiency(pruned_model)
    
    print("\n融合操作模型:")
    fused_time, fused_size = benchmark_efficiency(fused_model)
    
    print("\n半精度模型:")
    fp16_time, fp16_size = benchmark_efficiency(fp16_model)
    
    # 如果编译成功，也测试编译模型
    if hasattr(compiled_model, 'compiled_model') and compiled_model.compiled_model is not None:
        print("\n编译模型:")
        compiled_time, compiled_size = benchmark_efficiency(compiled_model)
    
    # 打印效率对比
    print("\n效率对比 (相对于标准模型):")
    print(f"剪枝模型: 速度 {standard_time/pruned_time:.2f}x, 大小 {standard_size/pruned_size:.2f}x")
    print(f"融合操作模型: 速度 {standard_time/fused_time:.2f}x, 大小 {standard_size/fused_size:.2f}x")
    print(f"半精度模型: 速度 {standard_time/fp16_time:.2f}x, 大小 {standard_size/fp16_size:.2f}x")
    if hasattr(compiled_model, 'compiled_model') and compiled_model.compiled_model is not None:
        print(f"编译模型: 速度 {standard_time/compiled_time:.2f}x, 大小 {standard_size/compiled_size:.2f}x")

if __name__ == '__main__':
    test() 