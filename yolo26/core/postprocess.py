import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


class PostProcess:
    """
    Post-processing for YOLOv26 dual-head inference.
    O2M branch: standard NMS
    O2O branch: NMS-free via per-anchor TopK
    """

    def __init__(self, reg_max=4, num_classes=80, conf_thresh=0.25, iou_thresh=0.45, topk=300):
        self.reg_max = reg_max
        self.nc = num_classes
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.topk = topk

    def _decode_o2o(self, preds, anchors, strides, image_size):
        """Decode O2O predictions: per-level decode, flatten all levels.

        preds: list of dicts with 'cls' and 'reg' tensors
        Returns: (B, N_all, 4) boxes and (B, N_all, nc) scores, normalized
        """
        decoded = []
        for level_idx, (pred, anchors_lvl, stride) in enumerate(zip(preds, anchors, strides)):
            reg_pred = pred["reg"]
            cls_pred = pred["cls"]
            B, _, H, W = reg_pred.shape
            device = reg_pred.device

            # Decode reg: (B, 4, H, W)
            reg = reg_pred.view(B, 4, self.reg_max, H, W)
            reg = F.softmax(reg, dim=2)
            reg_range = torch.arange(self.reg_max, device=device, dtype=reg.dtype)
            reg_range = reg_range.view(1, self.reg_max, 1, 1, 1)
            reg = (reg * reg_range).sum(dim=1)

            # Flatten spatial
            reg_flat = reg.permute(0, 2, 3, 1).reshape(B, -1, 4)  # (B, N_lvl, 4)

            # Anchor centers and sizes in pixels
            anc = anchors_lvl.to(device)  # (N_lvl, 4)
            stride_t = torch.tensor(stride, device=device, dtype=reg.dtype)

            cx_px = anc[:, 0:1] * image_size + reg_flat[..., 0:1] * stride_t
            cy_px = anc[:, 1:2] * image_size + reg_flat[..., 1:2] * stride_t
            w_px = anc[:, 2:3] * image_size * torch.exp(reg_flat[..., 2:3].clamp(max=math.log(image_size * 2)))
            h_px = anc[:, 3:4] * image_size * torch.exp(reg_flat[..., 3:4].clamp(max=math.log(image_size * 2)))

            boxes = torch.cat([cx_px / image_size, cy_px / image_size,
                               w_px / image_size, h_px / image_size], dim=-1).clamp(0, 1)

            # Class scores
            cls_flat = cls_pred.permute(0, 2, 3, 1).reshape(B, -1, self.nc).sigmoid()

            decoded.append({"boxes": boxes, "scores": cls_flat})

        boxes_cat = torch.cat([d["boxes"] for d in decoded], dim=1)
        scores_cat = torch.cat([d["scores"] for d in decoded], dim=1)
        return boxes_cat, scores_cat

    def __call__(self, preds, anchors, strides, image_size=640, mode="o2o"):
        if mode == "o2o":
            return self._o2o_postprocess(preds, anchors, strides, image_size)
        else:
            return self._o2m_postprocess(preds, anchors, strides, image_size)

    def _o2o_postprocess(self, preds, anchors, strides, image_size):
        """NMS-free O2O: Top-K over all levels."""
        boxes, scores = self._decode_o2o(preds, anchors, strides, image_size)
        B = boxes.shape[0]
        results = []

        for b in range(B):
            box_b = boxes[b]
            score_b = scores[b]

            max_scores, cls_idx = score_b.max(dim=1)
            keep = max_scores > self.conf_thresh

            if keep.sum() == 0:
                results.append({
                    "boxes": torch.zeros(0, 4, device=boxes.device),
                    "scores": torch.zeros(0, device=boxes.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=boxes.device),
                })
                continue

            box_k = box_b[keep]
            cls_k = cls_idx[keep]
            conf_k = max_scores[keep]

            topk_idx = torch.topk(conf_k, min(self.topk, conf_k.numel())).indices
            results.append({
                "boxes": box_k[topk_idx],
                "scores": conf_k[topk_idx],
                "labels": cls_k[topk_idx],
            })

        return results

    def _o2m_postprocess(self, preds, anchors, strides, image_size):
        """O2M: decode + class-aware NMS."""
        boxes, scores = self._decode_o2o(preds, anchors, strides, image_size)
        B = boxes.shape[0]
        results = []

        for b in range(B):
            box_b = boxes[b]
            score_b = scores[b]

            max_scores, labels = score_b.max(dim=1)
            keep = max_scores > self.conf_thresh

            if keep.sum() == 0:
                results.append({
                    "boxes": torch.zeros(0, 4, device=boxes.device),
                    "scores": torch.zeros(0, device=boxes.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=boxes.device),
                })
                continue

            box_k = box_b[keep]
            score_k = max_scores[keep]
            label_k = labels[keep]

            fb, fs, fl = [], [], []
            for cls in label_k.unique():
                mask = label_k == cls
                keep_nms = self._nms(box_k[mask], score_k[mask])
                fb.append(box_k[mask][keep_nms])
                fs.append(score_k[mask][keep_nms])
                fl.append(cls.expand(keep_nms.sum()))

            if fb:
                results.append({
                    "boxes": torch.cat(fb),
                    "scores": torch.cat(fs),
                    "labels": torch.cat(fl),
                })
            else:
                results.append({
                    "boxes": torch.zeros(0, 4, device=boxes.device),
                    "scores": torch.zeros(0, device=boxes.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=boxes.device),
                })

        return results

    def _nms(self, boxes, scores, eps=1e-7):
        """Standard NMS on cxcywh boxes."""
        if boxes.shape[0] == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)

        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = w * h

        order = scores.sort(descending=True)[1]
        keep = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break

            xx1 = torch.maximum(cx[i] - w[i] / 2, cx[order[1:]] - w[order[1:]] / 2)
            yy1 = torch.maximum(cy[i] - h[i] / 2, cy[order[1:]] - h[order[1:]] / 2)
            xx2 = torch.minimum(cx[i] + w[i] / 2, cx[order[1:]] + w[order[1:]] / 2)
            yy2 = torch.minimum(cy[i] + h[i] / 2, cy[order[1:]] + h[order[1:]] / 2)

            inter = ((xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0))
            iou = inter / (areas[i] + areas[order[1:]] - inter + eps)

            mask = iou <= self.iou_thresh
            order = order[1:][mask]

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
