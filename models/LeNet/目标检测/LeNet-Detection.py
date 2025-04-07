import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import nms

"""
LeNet-Detection：支持目标检测的LeNet变体
======================================

核心特点：
----------
1. 支持多目标检测
2. 实现边界框回归
3. 支持多类别分类
4. 实现非极大值抑制(NMS)
5. 提供目标检测评估指标

实现原理：
----------
1. 使用卷积层提取特征
2. 通过额外的卷积层预测边界框和类别
3. 使用锚框(Anchor Boxes)进行目标定位
4. 实现NMS去除冗余检测框
5. 支持多种评估指标计算

评估指标：
----------
1. 平均精度(AP)
2. 平均精度均值(mAP)
3. 交并比(IoU)
4. 精确率-召回率曲线
5. 检测速度(FPS)
"""

class DetectionLeNet(nn.Module):
    def __init__(self, num_classes=20, num_anchors=5):
        super(DetectionLeNet, self).__init__()
        
        # 基础特征提取网络
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2)
        
        # 检测头
        self.detection_conv = nn.Conv2d(16, 256, kernel_size=3, padding=1)
        self.bn_det = nn.BatchNorm2d(256)
        
        # 边界框预测
        self.bbox_conv = nn.Conv2d(256, num_anchors * 4, kernel_size=3, padding=1)
        
        # 类别预测
        self.cls_conv = nn.Conv2d(256, num_anchors * num_classes, kernel_size=3, padding=1)
        
        # 锚框配置
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        # 初始化锚框
        self.anchors = self._generate_anchors()
        
        # 评估指标
        self.metrics = {
            'ap': np.zeros(num_classes),
            'precision': [],
            'recall': [],
            'fps': 0
        }
    
    def _generate_anchors(self):
        """生成锚框"""
        # 这里使用简单的固定尺寸锚框
        anchor_sizes = [32, 64, 128, 256, 512]
        anchor_ratios = [0.5, 1.0, 2.0]
        
        anchors = []
        for size in anchor_sizes:
            for ratio in anchor_ratios:
                w = size * ratio
                h = size / ratio
                anchors.append([-w/2, -h/2, w/2, h/2])
        
        return torch.tensor(anchors)
    
    def forward(self, x):
        """前向传播"""
        # 特征提取
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # 检测特征
        x = F.relu(self.bn_det(self.detection_conv(x)))
        
        # 预测边界框
        bbox_pred = self.bbox_conv(x)
        bbox_pred = bbox_pred.view(-1, self.num_anchors, 4, bbox_pred.size(2), bbox_pred.size(3))
        
        # 预测类别
        cls_pred = self.cls_conv(x)
        cls_pred = cls_pred.view(-1, self.num_anchors, self.num_classes, cls_pred.size(2), cls_pred.size(3))
        
        return bbox_pred, cls_pred
    
    def decode_boxes(self, bbox_pred, cls_pred):
        """解码预测框"""
        # 将预测的偏移量转换为实际边界框
        batch_size = bbox_pred.size(0)
        height = bbox_pred.size(3)
        width = bbox_pred.size(4)
        
        # 生成网格点
        grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width))
        grid = torch.stack((grid_x, grid_y), dim=2).float()
        
        # 将预测的偏移量转换为实际坐标
        boxes = []
        scores = []
        for b in range(batch_size):
            batch_boxes = []
            batch_scores = []
            for a in range(self.num_anchors):
                # 获取锚框
                anchor = self.anchors[a]
                
                # 解码边界框
                dx = bbox_pred[b, a, 0]
                dy = bbox_pred[b, a, 1]
                dw = bbox_pred[b, a, 2]
                dh = bbox_pred[b, a, 3]
                
                # 计算实际坐标
                x = grid[:, :, 0] + dx
                y = grid[:, :, 1] + dy
                w = anchor[2] * torch.exp(dw)
                h = anchor[3] * torch.exp(dh)
                
                # 转换为(x1, y1, x2, y2)格式
                x1 = x - w/2
                y1 = y - h/2
                x2 = x + w/2
                y2 = y + h/2
                
                # 获取类别分数
                score = torch.sigmoid(cls_pred[b, a])
                
                batch_boxes.append(torch.stack((x1, y1, x2, y2), dim=2))
                batch_scores.append(score)
            
            boxes.append(torch.stack(batch_boxes))
            scores.append(torch.stack(batch_scores))
        
        return torch.stack(boxes), torch.stack(scores)
    
    def non_max_suppression(self, boxes, scores, iou_threshold=0.5, score_threshold=0.5):
        """非极大值抑制"""
        batch_size = boxes.size(0)
        num_anchors = boxes.size(1)
        height = boxes.size(2)
        width = boxes.size(3)
        
        # 展平预测结果
        boxes = boxes.view(batch_size, -1, 4)
        scores = scores.view(batch_size, -1, self.num_classes)
        
        # 存储最终结果
        final_boxes = []
        final_scores = []
        final_labels = []
        
        for b in range(batch_size):
            batch_boxes = []
            batch_scores = []
            batch_labels = []
            
            # 对每个类别单独进行NMS
            for c in range(self.num_classes):
                # 获取当前类别的分数
                class_scores = scores[b, :, c]
                
                # 过滤低分数预测
                mask = class_scores > score_threshold
                if not mask.any():
                    continue
                
                # 获取有效的框和分数
                valid_boxes = boxes[b, mask]
                valid_scores = class_scores[mask]
                
                # 按分数排序
                _, order = valid_scores.sort(descending=True)
                valid_boxes = valid_boxes[order]
                valid_scores = valid_scores[order]
                
                # 进行NMS
                keep = []
                while valid_boxes.size(0) > 0:
                    # 保留当前最高分数的框
                    keep.append(0)
                    
                    if valid_boxes.size(0) == 1:
                        break
                    
                    # 计算与最高分框的IoU
                    ious = self._compute_iou(valid_boxes[0:1], valid_boxes[1:])
                    
                    # 移除IoU大于阈值的框
                    mask = ious <= iou_threshold
                    valid_boxes = valid_boxes[1:][mask]
                    valid_scores = valid_scores[1:][mask]
                
                # 保存结果
                if keep:
                    batch_boxes.append(valid_boxes[keep])
                    batch_scores.append(valid_scores[keep])
                    batch_labels.append(torch.full_like(valid_scores[keep], c))
            
            if batch_boxes:
                final_boxes.append(torch.cat(batch_boxes))
                final_scores.append(torch.cat(batch_scores))
                final_labels.append(torch.cat(batch_labels))
            else:
                final_boxes.append(torch.zeros((0, 4)))
                final_scores.append(torch.zeros(0))
                final_labels.append(torch.zeros(0))
        
        return final_boxes, final_scores, final_labels
    
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
    
    def compute_ap(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
        """计算平均精度(AP)"""
        # 初始化结果
        ap = np.zeros(self.num_classes)
        
        # 对每个类别计算AP
        for c in range(self.num_classes):
            # 获取当前类别的预测
            class_mask = pred_labels == c
            if not class_mask.any():
                continue
            
            class_boxes = pred_boxes[class_mask]
            class_scores = pred_scores[class_mask]
            
            # 获取当前类别的真实框
            gt_class_mask = gt_labels == c
            gt_class_boxes = gt_boxes[gt_class_mask]
            
            # 如果没有真实框，跳过
            if len(gt_class_boxes) == 0:
                continue
            
            # 按分数排序
            sorted_indices = np.argsort(-class_scores)
            class_boxes = class_boxes[sorted_indices]
            class_scores = class_scores[sorted_indices]
            
            # 计算TP和FP
            tp = np.zeros(len(class_boxes))
            fp = np.zeros(len(class_boxes))
            
            for i, box in enumerate(class_boxes):
                # 计算与所有真实框的IoU
                ious = self._compute_iou(torch.tensor(box).unsqueeze(0), 
                                       torch.tensor(gt_class_boxes))
                max_iou = torch.max(ious).item()
                
                if max_iou >= iou_threshold:
                    tp[i] = 1
                else:
                    fp[i] = 1
            
            # 计算精确率和召回率
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recall = tp_cumsum / len(gt_class_boxes)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            # 计算AP
            ap[c] = self._compute_ap_from_pr(precision, recall)
        
        return ap
    
    def _compute_ap_from_pr(self, precision, recall):
        """从精确率-召回率曲线计算AP"""
        # 在召回率轴上插值
        recall_interp = np.linspace(0, 1, 101)
        precision_interp = np.zeros_like(recall_interp)
        
        for i, r in enumerate(recall_interp):
            precision_interp[i] = np.max(precision[recall >= r])
        
        # 计算AP
        ap = np.mean(precision_interp)
        return ap
    
    def visualize_detections(self, image, boxes, scores, labels, class_names):
        """可视化检测结果"""
        plt.figure(figsize=(10, 10))
        plt.imshow(image.permute(1, 2, 0).cpu().numpy())
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            plt.gca().add_patch(plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                fill=False, edgecolor='red', linewidth=2
            ))
            plt.text(
                x1, y1,
                f'{class_names[label]}: {score:.2f}',
                bbox=dict(facecolor='red', alpha=0.5)
            )
        
        plt.axis('off')
        plt.show()

def test():
    """测试DetectionLeNet模型"""
    # 创建模型
    model = DetectionLeNet(num_classes=20)
    
    # 创建随机输入
    x = torch.randn(1, 3, 224, 224)
    
    # 前向传播
    bbox_pred, cls_pred = model(x)
    print(f"边界框预测形状: {bbox_pred.shape}")
    print(f"类别预测形状: {cls_pred.shape}")
    
    # 解码预测框
    boxes, scores = model.decode_boxes(bbox_pred, cls_pred)
    print(f"解码后的边界框形状: {boxes.shape}")
    print(f"解码后的分数形状: {scores.shape}")
    
    # 应用NMS
    final_boxes, final_scores, final_labels = model.non_max_suppression(boxes, scores)
    print(f"NMS后的边界框数量: {len(final_boxes[0])}")
    print(f"NMS后的分数数量: {len(final_scores[0])}")
    print(f"NMS后的标签数量: {len(final_labels[0])}")
    
    # 测试AP计算
    pred_boxes = torch.randn(10, 4)
    pred_scores = torch.rand(10)
    pred_labels = torch.randint(0, 20, (10,))
    gt_boxes = torch.randn(5, 4)
    gt_labels = torch.randint(0, 20, (5,))
    
    ap = model.compute_ap(
        pred_boxes.numpy(), pred_scores.numpy(), pred_labels.numpy(),
        gt_boxes.numpy(), gt_labels.numpy()
    )
    print(f"AP: {ap.mean():.4f}")

if __name__ == '__main__':
    test() 