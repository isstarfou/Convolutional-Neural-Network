import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
FP16LeNet：使用半精度浮点数(FP16)的LeNet实现
==========================================

核心特点：
----------
1. 使用FP16格式存储权重和计算梯度
2. 实现混合精度训练 (AMP - Automatic Mixed Precision)
3. 使用梯度缩放避免数值下溢
4. 保持主权重为FP32以保证精度
5. 在推理时可完全使用FP16加速计算

混合精度训练优势：
---------------
1. 计算速度提升：FP16可在支持的硬件上获得2-8倍的计算加速
2. 内存使用减半：FP16权重占用空间为FP32的一半
3. 带宽减半：网络通信量减少，利于分布式训练

潜在挑战：
---------
1. 数值精度：FP16的表示范围有限，需要特殊处理避免溢出或下溢
2. 舍入误差：某些操作在FP16下精度显著降低
3. 梯度更新：梯度太小时可能变为零，需要梯度缩放

实现目标：
---------
1. 理解混合精度训练的原理和实现方法
2. 在不损失精度的情况下加速模型训练和推理
3. 掌握针对小值处理的技术
"""

class FP16LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(FP16LeNet, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        
        # 池化层
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # 是否使用混合精度
        self.use_amp = False
        
        # 梯度缩放因子（用于混合精度训练）
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # 主权重副本（FP32精度）
        self.master_weights = {}
    
    def to_half(self):
        """将模型参数转换为FP16格式"""
        # 保存FP32主权重副本
        for name, param in self.named_parameters():
            self.master_weights[name] = param.data.clone()
        
        # 转换模型为半精度
        self.half()
        self.use_amp = True
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        return self
    
    def to_float(self):
        """将模型参数转换回FP32格式"""
        # 如果存在主权重，使用它们恢复
        if self.master_weights:
            for name, param in self.named_parameters():
                if name in self.master_weights:
                    param.data = self.master_weights[name].clone()
            self.master_weights = {}
        else:
            # 否则直接转换
            self.float()
        
        self.use_amp = False
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
        return self
    
    def update_master_weights(self):
        """使用当前FP16权重更新FP32主权重"""
        if not self.master_weights:
            return
        
        for name, param in self.named_parameters():
            if name in self.master_weights:
                self.master_weights[name] = param.data.float().clone()
    
    def sync_weights_from_master(self):
        """从FP32主权重同步到FP16权重"""
        if not self.master_weights:
            return
        
        for name, param in self.named_parameters():
            if name in self.master_weights:
                param.data = self.master_weights[name].half()
    
    def forward(self, x):
        """前向传播"""
        # 如果输入不是FP16但模型是，转换输入
        if self.use_amp and x.dtype != torch.float16:
            x = x.half()
        
        # 第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 第二层
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class MixedPrecisionTrainer:
    """混合精度训练器"""
    def __init__(self, model, optimizer, loss_fn=F.cross_entropy):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scaler = torch.cuda.amp.GradScaler(enabled=model.use_amp)
    
    def train_step(self, inputs, targets):
        """单步训练"""
        # 零梯度
        self.optimizer.zero_grad()
        
        # 使用autocast进行前向传播
        with torch.cuda.amp.autocast(enabled=self.model.use_amp):
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
        
        # 反向传播 - 使用缩放器缩放梯度
        self.scaler.scale(loss).backward()
        
        # 更新参数 - 使用缩放器卸载和更新
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # 如果使用混合精度，从FP16更新FP32主权重
        if self.model.use_amp:
            self.model.update_master_weights()
        
        return loss.item(), outputs
    
    def evaluate(self, dataloader):
        """模型评估"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                # 使用autocast进行前向传播
                with torch.cuda.amp.autocast(enabled=self.model.use_amp):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train_epoch(self, train_loader, device):
        """训练一个完整周期"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            loss, outputs = self.train_step(inputs, targets)
            
            total_loss += loss * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy

def compare_precision():
    """比较FP16和FP32的精度和性能差异"""
    # 创建FP32模型
    fp32_model = FP16LeNet()
    
    # 创建并转换为FP16模型
    fp16_model = FP16LeNet().to_half()
    
    # 创建随机输入
    x = torch.randn(16, 1, 28, 28)
    x_half = x.half()
    
    # 启用推理模式
    fp32_model.eval()
    fp16_model.eval()
    
    # 计时比较
    import time
    
    # FP32前向传播
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            y_fp32 = fp32_model(x)
    fp32_time = time.time() - start
    
    # FP16前向传播
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            y_fp16 = fp16_model(x_half)
    fp16_time = time.time() - start
    
    # 比较输出差异
    if y_fp16.dtype != y_fp32.dtype:
        y_fp16 = y_fp16.float()
    
    output_diff = torch.abs(y_fp32 - y_fp16).mean().item()
    
    # 输出结果
    print(f"FP32模型推理时间: {fp32_time:.4f}秒")
    print(f"FP16模型推理时间: {fp16_time:.4f}秒")
    print(f"加速比: {fp32_time / fp16_time:.2f}x")
    print(f"输出平均差异: {output_diff:.8f}")
    print(f"FP32模型内存: {sum(p.numel() * 4 for p in fp32_model.parameters()) / 1024 / 1024:.2f} MB")
    print(f"FP16模型内存: {sum(p.numel() * 2 for p in fp16_model.parameters()) / 1024 / 1024:.2f} MB")

def test():
    """测试FP16LeNet模型"""
    # 测试基本功能
    model = FP16LeNet()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(f"FP32模型输出形状: {y.shape}")
    
    # 测试FP16功能
    model = model.to_half()
    x_half = x.half()
    y_half = model(x_half)
    print(f"FP16模型输出形状: {y_half.shape}")
    print(f"FP16模型参数类型: {next(model.parameters()).dtype}")
    
    # 测试混合精度训练
    model = FP16LeNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = MixedPrecisionTrainer(model.to_half(), optimizer)
    
    # 创建一个小的模拟批次
    inputs = torch.randn(8, 1, 28, 28).half()
    targets = torch.randint(0, 10, (8,))
    
    # 进行一步训练
    loss, _ = trainer.train_step(inputs, targets)
    print(f"训练损失: {loss:.4f}")
    
    # 比较FP16和FP32的精度和性能
    print("\n精度和性能比较:")
    compare_precision()

if __name__ == '__main__':
    test() 