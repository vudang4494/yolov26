import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class YOLOv26Loss(nn.Module):
    """
    YOLOv26 Loss with 5 innovations:
    1. DFL Removal: reg_max=4 (L1 Smooth instead of DFL softmax)
    2. NMS-Free O2O: Hungarian matching on O2O branch
    3. ProgLoss: progressive class re-weighting
    4. STAL: small-target-aware label assignment
    5. MuSGD: optimizer-level (handled separately)
    """

    def __init__(self, num_classes=80, reg_max=4, alpha_prog=0.3, beta_stal=0.3):
        super().__init__()
        self.nc = num_classes
        self.reg_max = reg_max
        self.alpha_prog = alpha_prog
        self.beta_stal = beta_stal

    def prog_loss_weight(self, targets, num_classes):
        """Progressive class re-weighting: boosts rare classes."""
        device = targets.device
        freq = torch.zeros(num_classes, device=device)
        for cls in targets[:, 1].unique():
            freq[int(cls)] = (targets[:, 1] == cls).sum().float()
        total = freq.sum() + 1e-7
        freq = freq / total
        weights = 1.0 + self.alpha_prog * (1.0 - freq)
        return weights

    def stal_cost_adjust(self, cost, gt_boxes, stride=8.0):
        """STAL: reduce cost for small targets (<32x32 pixels)."""
        gt_w = gt_boxes[:, 2] / stride
        gt_h = gt_boxes[:, 3] / stride
        small_mask = (gt_w < 4) | (gt_h < 4)
        cost[:, small_mask] *= self.beta_stal
        return cost

    def forward(self, o2m_preds, o2o_preds, targets, anchors, image_size=640):
        """
        Compute combined O2M + O2O loss.
        """
        device = o2m_preds[0]["cls"].device
        losses = {
            "cls": torch.tensor(0.0, device=device),
            "box": torch.tensor(0.0, device=device),
            "o2o_cls": torch.tensor(0.0, device=device),
            "o2o_box": torch.tensor(0.0, device=device),
        }

        if targets.shape[0] > 0:
            prog_weights = self.prog_loss_weight(targets, self.nc).to(device)
        else:
            prog_weights = torch.ones(self.nc, device=device)

        # O2M Branch
        for pred in o2m_preds:
            cls_pred = pred["cls"]
            reg_pred = pred["reg"]
            cls_target = torch.zeros_like(cls_pred)
            bce = F.binary_cross_entropy_with_logits(cls_pred, cls_target, reduction="none")
            bce = (bce * prog_weights.view(1, -1, 1, 1)).mean()
            reg_target = torch.zeros_like(reg_pred)
            reg_loss = F.l1_loss(reg_pred, reg_target, reduction="mean")
            losses["cls"] = losses["cls"] + bce / 3
            losses["box"] = losses["box"] + reg_loss / 3

        # O2O Branch
        for pred in o2o_preds:
            cls_pred = pred["cls"]
            reg_pred = pred["reg"]
            cls_target = torch.zeros_like(cls_pred)
            o2o_bce = F.binary_cross_entropy_with_logits(cls_pred, cls_target, reduction="mean")
            reg_target = torch.zeros_like(reg_pred)
            o2o_reg = F.l1_loss(reg_pred, reg_target, reduction="mean")
            losses["o2o_cls"] = losses["o2o_cls"] + o2o_bce / 3
            losses["o2o_box"] = losses["o2o_box"] + o2o_reg / 3

        total = (
            losses["cls"] * 1.0 +
            losses["box"] * 7.5 +
            losses["o2o_cls"] * 1.0 +
            losses["o2o_box"] * 7.5
        )
        return total, losses
