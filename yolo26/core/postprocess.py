import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import math


def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)


class PostProcess:
    """
    Post-processing for YOLOv26 dual-head inference.
    O2M branch: standard NMS
    O2O branch: Top-K with Hungarian matching (NMS-free)
    """

    def __init__(self, reg_max=4, num_classes=80, conf_thresh=0.25, iou_thresh=0.45, topk=300):
        self.reg_max = reg_max
        self.nc = num_classes
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.topk = topk

    def o2o_decode(self, o2o_preds, anchors, strides, image_size=640):
        """Decode O2O predictions: 4 reg_max channels -> direct bbox."""
        decoded = []
        for level_idx, (pred, anchors_lvl, stride) in enumerate(
                zip(o2o_preds, anchors, strides)):
            B, _, H, W = pred["reg"].shape
            cls_pred = pred["cls"]
            reg_pred = pred["reg"]
            device = reg_pred.device

            reg = reg_pred.view(B, 4, self.reg_max, H, W)
            reg = F.softmax(reg, dim=2)
            reg_range = torch.arange(self.reg_max, device=device, dtype=reg.dtype).view(1, self.reg_max, 1, 1, 1)
            reg = (reg * reg_range).sum(dim=1)

            cls_flat = cls_pred.permute(0, 2, 3, 1).reshape(B, -1, self.nc).sigmoid()
            reg_flat = reg.permute(0, 2, 3, 1).reshape(B, -1, 4)

            # anchors_lvl: (n, 4) = [cx_norm, cy_norm, w_norm, h_norm] normalized to [0,1]
            anchor_flat = anchors_lvl.to(device).reshape(1, -1, 4)
            stride_for_level = strides[level_idx]
            stride_tensor = torch.tensor(stride_for_level, device=device, dtype=torch.float32).view(1, 1, 1)

            # Anchor centers in absolute pixels
            anchor_cx_px = anchor_flat[..., 0:1] * image_size
            anchor_cy_px = anchor_flat[..., 1:2] * image_size
            anchor_w_px = anchor_flat[..., 2:3] * image_size
            anchor_h_px = anchor_flat[..., 3:4] * image_size

            # Decode: absolute = anchor + reg * stride
            dx_px = reg_flat[..., 0:1] * stride_tensor
            dy_px = reg_flat[..., 1:2] * stride_tensor
            dw_px = reg_flat[..., 2:3] * stride_tensor
            dh_px = reg_flat[..., 3:4] * stride_tensor

            cx_px = anchor_cx_px + dx_px
            cy_px = anchor_cy_px + dy_px
            w_px = anchor_w_px * torch.exp(dw_px.clamp(max=math.log(image_size * 2)))
            h_px = anchor_h_px * torch.exp(dh_px.clamp(max=math.log(image_size * 2)))

            # Normalize to [0, 1]
            bboxes = torch.cat([
                cx_px / image_size,
                cy_px / image_size,
                w_px / image_size,
                h_px / image_size,
            ], dim=-1).clamp(0, 1)
            decoded.append({"boxes": bboxes, "scores": cls_flat})

        boxes_cat = torch.cat([d["boxes"] for d in decoded], dim=1)
        scores_cat = torch.cat([d["scores"] for d in decoded], dim=1)
        return boxes_cat, scores_cat

    def __call__(self, o2o_preds, anchors, strides, image_size=640, mode="o2o"):
        """
        Post-process predictions.

        Args:
            o2o_preds: list of dicts with 'cls' and 'reg' tensors
            anchors: list of anchor tensors per level
            strides: list of stride values per level
            mode: 'o2o' for NMS-free, 'o2m' for NMS-based
        """
        if mode == "o2o":
            return self._o2o_postprocess(o2o_preds, anchors, strides, image_size)
        else:
            return self._o2m_postprocess(o2o_preds, anchors, strides, image_size)

    def _o2o_postprocess(self, o2o_preds, anchors, strides, image_size):
        """NMS-free O2O post-processing using Top-K + direct decode."""
        boxes, scores = self.o2o_decode(o2o_preds, anchors, strides, image_size)
        B = boxes.shape[0]
        results = []

        for b in range(B):
            box_b = boxes[b]
            score_b = scores[b]

            max_scores, _ = score_b.max(dim=1)
            keep = max_scores > self.conf_thresh

            if keep.sum() == 0:
                results.append({"boxes": torch.zeros(0, 4), "scores": torch.zeros(0),
                                 "labels": torch.zeros(0, dtype=torch.long)})
                continue

            box_k = box_b[keep]
            score_k = score_b[keep]
            conf_k = max_scores[keep]

            topk_idx = torch.topk(conf_k, min(self.topk, conf_k.numel())).indices
            box_topk = box_k[topk_idx]
            score_topk = score_k[topk_idx]
            conf_topk = conf_k[topk_idx]

            cls_idx = score_topk.argmax(dim=1)

            # boxes already normalized to [0,1] by decode
            final_scores = conf_topk * score_topk.gather(1, cls_idx.unsqueeze(1)).squeeze()

            results.append({
                "boxes": box_topk,
                "scores": final_scores,
                "labels": cls_idx,
            })

        return results

    def _o2m_postprocess(self, o2m_preds, anchors, strides, image_size):
        """O2M post-processing with standard NMS."""
        boxes, scores = self.o2o_decode(o2m_preds, anchors, strides, image_size)
        B = boxes.shape[0]
        results = []

        for b in range(B):
            box_b = boxes[b]
            score_b = scores[b]
            max_scores, labels = score_b.max(dim=1)

            keep_mask = max_scores > self.conf_thresh
            if keep_mask.sum() == 0:
                results.append({"boxes": torch.zeros(0, 4), "scores": torch.zeros(0),
                                 "labels": torch.zeros(0, dtype=torch.long)})
                continue

            box_k = box_b[keep_mask]
            conf_k = max_scores[keep_mask]
            label_k = labels[keep_mask]

            results_b = {"boxes": [], "scores": [], "labels": []}

            for cls in label_k.unique():
                cls_mask = label_k == cls
                box_cls = box_k[cls_mask]
                conf_cls = conf_k[cls_mask]

                keep_nms = self._nms(box_cls, conf_cls)
                results_b["boxes"].append(box_cls[keep_nms])
                results_b["scores"].append(conf_cls[keep_nms])
                results_b["labels"].append(cls.expand(keep_nms.sum()))

            if results_b["boxes"]:
                results.append({
                    "boxes": torch.cat(results_b["boxes"]),
                    "scores": torch.cat(results_b["scores"]),
                    "labels": torch.cat(results_b["labels"]),
                })
            else:
                results.append({"boxes": torch.zeros(0, 4), "scores": torch.zeros(0),
                                 "labels": torch.zeros(0, dtype=torch.long)})

        return results

    def _nms(self, boxes, scores, eps=1e-7):
        """Standard NMS."""
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        areas = (x2 - x1) * (y2 - y1)

        order = scores.sort(descending=True)[1]
        keep = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break

            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])

            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + eps)

            mask = iou <= self.iou_thresh
            order = order[1:][mask]

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
