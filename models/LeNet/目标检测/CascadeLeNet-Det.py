import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import nms

"""
CascadeLeNet-Det：级联LeNet目标检测模型
======================================

核心特点：
----------
1. 多级联检测器架构
2. 渐进式特征提取
3. 级联分类器设计
4. 自适应特征融合
5. 多尺度目标检测

实现原理：
----------
1. 使用多个LeNet网络级联
2. 每个阶段专注于特定尺度的目标
3. 特征图逐步细化
4. 级联分类器逐步过滤
5. 自适应特征融合策略

评估指标：
----------
1. 级联检测准确率
2. 各阶段检测性能
3. 误检率分析
4. 检测速度评估
5. 级联效率分析
"""

class CascadeLeNet(nn.Module):
    def __init__(self, num_classes=20, num_stages=3):
        super(CascadeLeNet, self).__init__()
        
        self.num_stages = num_stages
        self.num_classes = num_classes
        
        # 基础特征提取网络
        self.base_net = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # 级联检测器
        self.cascade_detectors = nn.ModuleList([
            self._make_detector_stage(i) for i in range(num_stages)
        ])
        
        # 特征融合层
        self.fusion_layers = nn.ModuleList([
            nn.Conv2d(16 * (i + 1), 16, kernel_size=1)
            for i in range(num_stages - 1)
        ])
        
        # 评估指标
        self.metrics = {
            'stage_accuracy': np.zeros(num_stages),
            'overall_accuracy': 0,
            'false_positive_rate': np.zeros(num_stages),
            'detection_speed': 0
        }
    
    def _make_detector_stage(self, stage_idx):
        """创建单个检测器阶段"""
        return nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, self.num_classes + 4, kernel_size=1)
        )
    
    def forward(self, x):
        """前向传播"""
        # 提取基础特征
        base_features = self.base_net(x)
        
        # 存储各阶段特征
        stage_features = [base_features]
        detections = []
        
        # 逐阶段处理
        for i in range(self.num_stages):
            # 特征融合
            if i > 0:
                fused_features = torch.cat(stage_features, dim=1)
                current_features = self.fusion_layers[i-1](fused_features)
            else:
                current_features = base_features
            
            # 当前阶段检测
            stage_output = self.cascade_detectors[i](current_features)
            
            # 分离分类和回归结果
            cls_pred = stage_output[:, :self.num_classes]
            bbox_pred = stage_output[:, self.num_classes:]
            
            detections.append((cls_pred, bbox_pred))
            stage_features.append(current_features)
        
        return detections
    
    def cascade_nms(self, detections, iou_threshold=0.5, score_threshold=0.5):
        """级联非极大值抑制"""
        final_boxes = []
        final_scores = []
        final_labels = []
        
        for stage_det in detections:
            cls_pred, bbox_pred = stage_det
            
            # 对每个类别进行NMS
            for c in range(self.num_classes):
                scores = cls_pred[:, c]
                mask = scores > score_threshold
                
                if not mask.any():
                    continue
                
                boxes = bbox_pred[mask]
                scores = scores[mask]
                
                # 进行NMS
                keep = nms(boxes, scores, iou_threshold)
                
                if len(keep) > 0:
                    final_boxes.append(boxes[keep])
                    final_scores.append(scores[keep])
                    final_labels.append(torch.full_like(scores[keep], c))
        
        if final_boxes:
            return (torch.cat(final_boxes),
                    torch.cat(final_scores),
                    torch.cat(final_labels))
        else:
            return (torch.zeros((0, 4)),
                    torch.zeros(0),
                    torch.zeros(0))
    
    def compute_stage_metrics(self, detections, targets):
        """计算各阶段评估指标"""
        for i, (cls_pred, bbox_pred) in enumerate(detections):
            # 计算当前阶段的准确率
            accuracy = self._compute_accuracy(cls_pred, targets)
            self.metrics['stage_accuracy'][i] = accuracy
            
            # 计算误检率
            fpr = self._compute_false_positive_rate(cls_pred, targets)
            self.metrics['false_positive_rate'][i] = fpr
        
        # 计算整体准确率
        self.metrics['overall_accuracy'] = np.mean(self.metrics['stage_accuracy'])
    
    def _compute_accuracy(self, predictions, targets):
        """计算准确率"""
        pred_labels = predictions.argmax(dim=1)
        correct = (pred_labels == targets).float().mean()
        return correct.item()
    
    def _compute_false_positive_rate(self, predictions, targets):
        """计算误检率"""
        pred_labels = predictions.argmax(dim=1)
        false_positives = ((pred_labels != targets) & (pred_labels != 0)).float().sum()
        total_predictions = (pred_labels != 0).float().sum()
        return (false_positives / total_predictions).item() if total_predictions > 0 else 0
    
    def visualize_cascade(self, image, detections):
        """可视化级联检测结果"""
        plt.figure(figsize=(15, 5 * self.num_stages))
        
        for i, (cls_pred, bbox_pred) in enumerate(detections):
            plt.subplot(self.num_stages, 1, i + 1)
            plt.imshow(image.permute(1, 2, 0).cpu().numpy())
            
            # 绘制检测框
            scores = cls_pred.max(dim=1)[0]
            mask = scores > 0.5
            if mask.any():
                boxes = bbox_pred[mask]
                for box in boxes:
                    x1, y1, x2, y2 = box.cpu().numpy()
                    plt.gca().add_patch(plt.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        fill=False, edgecolor='red', linewidth=2
                    ))
            
            plt.title(f'Stage {i+1} Detection Results')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def test():
    """测试函数"""
    # 创建模型
    model = CascadeLeNet(num_classes=20, num_stages=3)
    
    # 创建随机输入
    x = torch.randn(1, 3, 224, 224)
    
    # 前向传播
    detections = model(x)
    
    # 打印输出形状
    for i, (cls_pred, bbox_pred) in enumerate(detections):
        print(f'Stage {i+1}:')
        print(f'  Classification shape: {cls_pred.shape}')
        print(f'  Bounding box shape: {bbox_pred.shape}')
    
    # 进行NMS
    boxes, scores, labels = model.cascade_nms(detections)
    print(f'\nFinal detections:')
    print(f'  Boxes: {boxes.shape}')
    print(f'  Scores: {scores.shape}')
    print(f'  Labels: {labels.shape}')

if __name__ == '__main__':
    test() 