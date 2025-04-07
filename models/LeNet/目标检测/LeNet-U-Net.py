import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

"""
LeNet-U-Net：基于U-Net的语义分割模型
================================

核心特点：
----------
1. 编码器-解码器结构
2. 跳跃连接
3. 多尺度特征融合
4. 像素级分类
5. 边界框预测

实现原理：
----------
1. 使用LeNet作为编码器
2. 添加对称的解码器路径
3. 使用跳跃连接融合不同尺度的特征
4. 输出像素级类别预测
5. 可选地输出边界框预测

评估指标：
----------
1. 像素准确率(PA)
2. 平均交并比(mIoU)
3. 类别平均准确率(CPA)
4. 边界框准确率
5. 分割速度(FPS)
"""

class UNetLeNet(nn.Module):
    def __init__(self, num_classes=20, use_bbox=False):
        super(UNetLeNet, self).__init__()
        
        self.num_classes = num_classes
        self.use_bbox = use_bbox
        
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        # 中间层
        self.mid = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(16, 6, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(12, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True)
        )
        
        # 分割头
        self.seg_head = nn.Conv2d(6, num_classes, kernel_size=1)
        
        # 边界框预测头
        if use_bbox:
            self.bbox_head = nn.Sequential(
                nn.Conv2d(6, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 4, kernel_size=1)
            )
        
        # 评估指标
        self.metrics = {
            'pa': 0,
            'miou': 0,
            'cpa': np.zeros(num_classes),
            'bbox_acc': 0 if use_bbox else None,
            'fps': 0
        }
    
    def forward(self, x):
        """前向传播"""
        # 编码器路径
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        # 中间层
        x = self.mid(x)
        
        # 解码器路径
        x = self.up2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # 分割预测
        seg_pred = self.seg_head(x)
        
        # 边界框预测
        bbox_pred = None
        if self.use_bbox:
            bbox_pred = self.bbox_head(x)
        
        return seg_pred, bbox_pred
    
    def compute_pixel_accuracy(self, pred, target):
        """计算像素准确率"""
        pred = torch.argmax(pred, dim=1)
        correct = (pred == target).float()
        return correct.mean().item()
    
    def compute_miou(self, pred, target):
        """计算平均交并比"""
        pred = torch.argmax(pred, dim=1)
        miou = 0
        for c in range(self.num_classes):
            pred_c = (pred == c)
            target_c = (target == c)
            
            intersection = (pred_c & target_c).float().sum()
            union = (pred_c | target_c).float().sum()
            
            if union > 0:
                miou += intersection / union
        
        return miou / self.num_classes
    
    def compute_class_pixel_accuracy(self, pred, target):
        """计算类别平均准确率"""
        pred = torch.argmax(pred, dim=1)
        cpa = np.zeros(self.num_classes)
        
        for c in range(self.num_classes):
            pred_c = (pred == c)
            target_c = (target == c)
            
            correct = (pred_c & target_c).float().sum()
            total = target_c.float().sum()
            
            if total > 0:
                cpa[c] = correct / total
        
        return cpa
    
    def compute_bbox_accuracy(self, pred_boxes, gt_boxes, iou_threshold=0.5):
        """计算边界框准确率"""
        if not self.use_bbox:
            return 0
        
        batch_size = pred_boxes.size(0)
        total_correct = 0
        total_boxes = 0
        
        for i in range(batch_size):
            pred = pred_boxes[i]
            gt = gt_boxes[i]
            
            if len(gt) == 0:
                continue
            
            # 计算IoU
            ious = self._compute_iou(pred, gt)
            
            # 计算准确率
            max_ious, _ = torch.max(ious, dim=1)
            correct = (max_ious >= iou_threshold).float().sum()
            
            total_correct += correct
            total_boxes += len(gt)
        
        return total_correct / total_boxes if total_boxes > 0 else 0
    
    def _compute_iou(self, box1, box2):
        """计算IoU"""
        # 计算交集
        x1 = torch.max(box1[:, 0], box2[:, 0])
        y1 = torch.max(box1[:, 1], box2[:, 1])
        x2 = torch.min(box1[:, 2], box2[:, 2])
        y2 = torch.min(box1[:, 3], box2[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # 计算并集
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union = area1 + area2 - intersection
        
        # 计算IoU
        iou = intersection / union
        
        return iou
    
    def update_metrics(self, pred, target, pred_boxes=None, gt_boxes=None):
        """更新评估指标"""
        self.metrics['pa'] = self.compute_pixel_accuracy(pred, target)
        self.metrics['miou'] = self.compute_miou(pred, target)
        self.metrics['cpa'] = self.compute_class_pixel_accuracy(pred, target)
        
        if self.use_bbox and pred_boxes is not None and gt_boxes is not None:
            self.metrics['bbox_acc'] = self.compute_bbox_accuracy(pred_boxes, gt_boxes)
    
    def visualize_results(self, image, pred, target, pred_boxes=None, gt_boxes=None):
        """可视化结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # 显示原始图像
        axes[0, 0].imshow(image.permute(1, 2, 0).cpu().numpy())
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 显示预测分割
        pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
        axes[0, 1].imshow(pred_mask)
        axes[0, 1].set_title('Predicted Segmentation')
        axes[0, 1].axis('off')
        
        # 显示真实分割
        target_mask = target.squeeze().cpu().numpy()
        axes[1, 0].imshow(target_mask)
        axes[1, 0].set_title('Ground Truth')
        axes[1, 0].axis('off')
        
        # 显示边界框（如果有）
        if self.use_bbox and pred_boxes is not None and gt_boxes is not None:
            axes[1, 1].imshow(image.permute(1, 2, 0).cpu().numpy())
            for box in pred_boxes[0]:
                x1, y1, x2, y2 = box
                axes[1, 1].add_patch(plt.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    fill=False, edgecolor='red', linewidth=2
                ))
            for box in gt_boxes[0]:
                x1, y1, x2, y2 = box
                axes[1, 1].add_patch(plt.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    fill=False, edgecolor='green', linewidth=2
                ))
            axes[1, 1].set_title('Bounding Boxes (Red: Pred, Green: GT)')
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()

def test():
    """测试UNetLeNet模型"""
    # 创建模型
    model = UNetLeNet(num_classes=20, use_bbox=True)
    
    # 创建随机输入
    x = torch.randn(1, 3, 224, 224)
    
    # 前向传播
    seg_pred, bbox_pred = model(x)
    print(f"分割预测形状: {seg_pred.shape}")
    if bbox_pred is not None:
        print(f"边界框预测形状: {bbox_pred.shape}")
    
    # 创建随机目标
    target = torch.randint(0, 20, (1, 224, 224))
    gt_boxes = torch.randn(1, 5, 4)  # 假设有5个真实框
    
    # 更新评估指标
    model.update_metrics(seg_pred, target, bbox_pred, gt_boxes)
    
    # 打印评估指标
    print(f"像素准确率: {model.metrics['pa']:.4f}")
    print(f"平均交并比: {model.metrics['miou']:.4f}")
    print(f"类别平均准确率: {model.metrics['cpa'].mean():.4f}")
    if model.use_bbox:
        print(f"边界框准确率: {model.metrics['bbox_acc']:.4f}")
    
    # 可视化结果
    model.visualize_results(x, seg_pred, target, bbox_pred, gt_boxes)

if __name__ == '__main__':
    test() 