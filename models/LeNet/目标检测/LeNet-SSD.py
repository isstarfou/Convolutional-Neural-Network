import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import nms

"""
LeNet-SSD：基于SSD的目标检测模型
================================

核心特点：
----------
1. 单阶段目标检测
2. 多尺度特征图检测
3. 默认框(Default Boxes)机制
4. 多类别分类
5. 边界框回归

实现原理：
----------
1. 使用LeNet作为基础特征提取网络
2. 添加额外的卷积层生成多尺度特征图
3. 在每个特征图上预测类别和边界框
4. 使用默认框机制进行目标定位
5. 实现非极大值抑制(NMS)去除冗余检测框

评估指标：
----------
1. 平均精度(AP)
2. 平均精度均值(mAP)
3. 交并比(IoU)
4. 精确率-召回率曲线
5. 检测速度(FPS)
"""

class SSDLeNet(nn.Module):
    def __init__(self, num_classes=20, num_anchors=6):
        super(SSDLeNet, self).__init__()
        
        # 基础特征提取网络
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2)
        
        # 额外的卷积层用于多尺度特征图
        self.extra_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        ])
        
        # 检测头
        self.loc_layers = nn.ModuleList([
            nn.Conv2d(16, num_anchors * 4, kernel_size=3, padding=1),
            nn.Conv2d(32, num_anchors * 4, kernel_size=3, padding=1),
            nn.Conv2d(64, num_anchors * 4, kernel_size=3, padding=1)
        ])
        
        self.conf_layers = nn.ModuleList([
            nn.Conv2d(16, num_anchors * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(32, num_anchors * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(64, num_anchors * num_classes, kernel_size=3, padding=1)
        ])
        
        # 默认框配置
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.default_boxes = self._generate_default_boxes()
        
        # 评估指标
        self.metrics = {
            'ap': np.zeros(num_classes),
            'precision': [],
            'recall': [],
            'fps': 0
        }
    
    def _generate_default_boxes(self):
        """生成默认框"""
        feature_maps = [(28, 28), (14, 14), (7, 7)]
        scales = [0.2, 0.4, 0.6]
        aspect_ratios = [1.0, 2.0, 0.5]
        
        default_boxes = []
        for k, (h, w) in enumerate(feature_maps):
            scale = scales[k]
            for i in range(h):
                for j in range(w):
                    cx = (j + 0.5) / w
                    cy = (i + 0.5) / h
                    
                    for ar in aspect_ratios:
                        default_boxes.append([
                            cx,
                            cy,
                            scale * np.sqrt(ar),
                            scale / np.sqrt(ar)
                        ])
        
        return torch.tensor(default_boxes)
    
    def forward(self, x):
        """前向传播"""
        sources = []
        loc = []
        conf = []
        
        # 基础特征提取
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        sources.append(x)
        
        # 额外的特征图
        for layer in self.extra_layers:
            x = layer(x)
            sources.append(x)
        
        # 预测
        for (x, l, c) in zip(sources, self.loc_layers, self.conf_layers):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        return loc, conf
    
    def decode_boxes(self, loc_pred, conf_pred):
        """解码预测框"""
        batch_size = loc_pred.size(0)
        num_boxes = loc_pred.size(1) // 4
        
        # 解码边界框
        boxes = []
        scores = []
        for i in range(batch_size):
            # 解码位置预测
            loc = loc_pred[i].view(-1, 4)
            conf = conf_pred[i].view(-1, self.num_classes)
            
            # 将预测的偏移量转换为实际坐标
            boxes_i = []
            scores_i = []
            for j in range(num_boxes):
                default_box = self.default_boxes[j]
                loc_j = loc[j]
                
                # 计算实际坐标
                cx = default_box[0] + loc_j[0] * 0.1 * default_box[2]
                cy = default_box[1] + loc_j[1] * 0.1 * default_box[3]
                w = default_box[2] * torch.exp(loc_j[2] * 0.2)
                h = default_box[3] * torch.exp(loc_j[3] * 0.2)
                
                # 转换为(x1, y1, x2, y2)格式
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                
                boxes_i.append([x1, y1, x2, y2])
                scores_i.append(torch.softmax(conf[j], dim=0))
            
            boxes.append(torch.tensor(boxes_i))
            scores.append(torch.stack(scores_i))
        
        return torch.stack(boxes), torch.stack(scores)
    
    def non_max_suppression(self, boxes, scores, iou_threshold=0.5, score_threshold=0.5):
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
            for c in range(1, self.num_classes):  # 跳过背景类
                # 获取当前类别的分数
                class_scores = scores[i, :, c]
                
                # 过滤低分数预测
                mask = class_scores > score_threshold
                if not mask.any():
                    continue
                
                # 获取有效的框和分数
                valid_boxes = boxes[i, mask]
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
        for c in range(1, self.num_classes):  # 跳过背景类
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
    """测试SSDLeNet模型"""
    # 创建模型
    model = SSDLeNet(num_classes=20)
    
    # 创建随机输入
    x = torch.randn(1, 3, 224, 224)
    
    # 前向传播
    loc_pred, conf_pred = model(x)
    print(f"位置预测形状: {loc_pred.shape}")
    print(f"类别预测形状: {conf_pred.shape}")
    
    # 解码预测框
    boxes, scores = model.decode_boxes(loc_pred, conf_pred)
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
    pred_labels = torch.randint(1, 20, (10,))
    gt_boxes = torch.randn(5, 4)
    gt_labels = torch.randint(1, 20, (5,))
    
    ap = model.compute_ap(
        pred_boxes.numpy(), pred_scores.numpy(), pred_labels.numpy(),
        gt_boxes.numpy(), gt_labels.numpy()
    )
    print(f"AP: {ap.mean():.4f}")

if __name__ == '__main__':
    test()
