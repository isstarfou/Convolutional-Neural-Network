import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import nms

"""
LeNet-YOLO：基于YOLO的目标检测模型
================================

核心特点：
----------
1. 单阶段目标检测
2. 网格化检测
3. 边界框预测
4. 置信度预测
5. 类别预测

实现原理：
----------
1. 使用LeNet作为基础特征提取网络
2. 将输入图像划分为SxS网格
3. 每个网格预测B个边界框
4. 每个边界框预测位置、置信度和类别
5. 使用非极大值抑制(NMS)去除冗余检测框

评估指标：
----------
1. 平均精度(AP)
2. 平均精度均值(mAP)
3. 交并比(IoU)
4. 精确率-召回率曲线
5. 检测速度(FPS)
"""

class YOLOLeNet(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOLeNet, self).__init__()
        
        self.S = S  # 网格大小
        self.B = B  # 每个网格预测的边界框数量
        self.C = C  # 类别数量
        
        # 基础特征提取网络
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2)
        
        # YOLO检测头
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # 输出层
        self.conv5 = nn.Conv2d(64, S * S * (B * 5 + C), kernel_size=1)
        
        # 评估指标
        self.metrics = {
            'ap': np.zeros(C),
            'precision': [],
            'recall': [],
            'fps': 0
        }
    
    def forward(self, x):
        """前向传播"""
        # 基础特征提取
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # YOLO检测头
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # 输出层
        x = self.conv5(x)
        
        # 重塑输出
        batch_size = x.size(0)
        x = x.view(batch_size, self.S, self.S, self.B * 5 + self.C)
        
        return x
    
    def decode_predictions(self, predictions):
        """解码预测结果"""
        batch_size = predictions.size(0)
        device = predictions.device
        
        # 获取预测的边界框
        boxes = []
        scores = []
        labels = []
        
        for b in range(batch_size):
            batch_boxes = []
            batch_scores = []
            batch_labels = []
            
            for i in range(self.S):
                for j in range(self.S):
                    # 获取当前网格的预测
                    cell_pred = predictions[b, i, j]
                    
                    # 对每个边界框
                    for box_idx in range(self.B):
                        # 获取边界框参数
                        box_start = box_idx * 5
                        x = (j + torch.sigmoid(cell_pred[box_start])) / self.S
                        y = (i + torch.sigmoid(cell_pred[box_start + 1])) / self.S
                        w = torch.exp(cell_pred[box_start + 2]) / self.S
                        h = torch.exp(cell_pred[box_start + 3]) / self.S
                        conf = torch.sigmoid(cell_pred[box_start + 4])
                        
                        # 获取类别预测
                        class_pred = cell_pred[self.B * 5:]
                        class_conf = torch.softmax(class_pred, dim=0)
                        class_score, class_label = torch.max(class_conf, dim=0)
                        
                        # 计算最终置信度
                        final_conf = conf * class_score
                        
                        # 转换为(x1, y1, x2, y2)格式
                        x1 = x - w/2
                        y1 = y - h/2
                        x2 = x + w/2
                        y2 = y + h/2
                        
                        batch_boxes.append([x1, y1, x2, y2])
                        batch_scores.append(final_conf)
                        batch_labels.append(class_label)
            
            boxes.append(torch.tensor(batch_boxes, device=device))
            scores.append(torch.tensor(batch_scores, device=device))
            labels.append(torch.tensor(batch_labels, device=device))
        
        return torch.stack(boxes), torch.stack(scores), torch.stack(labels)
    
    def non_max_suppression(self, boxes, scores, labels, iou_threshold=0.5, score_threshold=0.5):
        """非极大值抑制"""
        batch_size = boxes.size(0)
        final_boxes = []
        final_scores = []
        final_labels = []
        
        for i in range(batch_size):
            batch_boxes = []
            batch_scores = []
            batch_labels = []
            
            # 对每个类别单独进行NMS
            for c in range(self.C):
                # 获取当前类别的预测
                class_mask = labels[i] == c
                if not class_mask.any():
                    continue
                
                class_boxes = boxes[i, class_mask]
                class_scores = scores[i, class_mask]
                
                # 过滤低分数预测
                score_mask = class_scores > score_threshold
                if not score_mask.any():
                    continue
                
                class_boxes = class_boxes[score_mask]
                class_scores = class_scores[score_mask]
                
                # 按分数排序
                _, order = class_scores.sort(descending=True)
                class_boxes = class_boxes[order]
                class_scores = class_scores[order]
                
                # 进行NMS
                keep = []
                while class_boxes.size(0) > 0:
                    # 保留当前最高分数的框
                    keep.append(0)
                    
                    if class_boxes.size(0) == 1:
                        break
                    
                    # 计算与最高分框的IoU
                    ious = self._compute_iou(class_boxes[0:1], class_boxes[1:])
                    
                    # 移除IoU大于阈值的框
                    mask = ious <= iou_threshold
                    class_boxes = class_boxes[1:][mask]
                    class_scores = class_scores[1:][mask]
                
                # 保存结果
                if keep:
                    batch_boxes.append(class_boxes[keep])
                    batch_scores.append(class_scores[keep])
                    batch_labels.append(torch.full_like(class_scores[keep], c))
            
            if batch_boxes:
                final_boxes.append(torch.cat(batch_boxes))
                final_scores.append(torch.cat(batch_scores))
                final_labels.append(torch.cat(batch_labels))
            else:
                final_boxes.append(torch.zeros((0, 4), device=boxes.device))
                final_scores.append(torch.zeros(0, device=scores.device))
                final_labels.append(torch.zeros(0, device=labels.device))
        
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
        ap = np.zeros(self.C)
        
        # 对每个类别计算AP
        for c in range(self.C):
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
    """测试YOLOLeNet模型"""
    # 创建模型
    model = YOLOLeNet(S=7, B=2, C=20)
    
    # 创建随机输入
    x = torch.randn(1, 3, 224, 224)
    
    # 前向传播
    predictions = model(x)
    print(f"预测输出形状: {predictions.shape}")
    
    # 解码预测结果
    boxes, scores, labels = model.decode_predictions(predictions)
    print(f"解码后的边界框形状: {boxes.shape}")
    print(f"解码后的分数形状: {scores.shape}")
    print(f"解码后的标签形状: {labels.shape}")
    
    # 应用NMS
    final_boxes, final_scores, final_labels = model.non_max_suppression(boxes, scores, labels)
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