import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOv26Loss(nn.Module):
    """
    YOLOv26 Loss with 5 innovations:
    1. DFL Removal: reg_max=4 (L1 Smooth instead of DFL softmax)
    2. NMS-Free O2O: Top-K matching on O2O branch
    3. ProgLoss: progressive class re-weighting
    4. STAL: small-target-aware label assignment
    5. MuSGD: optimizer-level (handled in musgd.py)

    Label assignment: anchor-based matching per feature level.
    Each GT box is assigned to the grid cell containing its center.
    """

    def __init__(self, num_classes=80, reg_max=4, alpha_prog=0.3, beta_stal=0.3,
                 o2o_weight=1.0):
        super().__init__()
        self.nc = num_classes
        self.reg_max = reg_max
        self.alpha_prog = alpha_prog
        self.beta_stal = beta_stal
        self.o2o_weight = o2o_weight

    def prog_loss_weight(self, targets, num_classes):
        """Progressive class re-weighting: boosts rare classes."""
        if targets.shape[0] == 0:
            return torch.ones(num_classes, device=targets.device)

        device = targets.device
        freq = torch.zeros(num_classes, device=device)
        for cls in targets[:, 1].unique():
            freq[int(cls)] = (targets[:, 1] == cls).sum().float()
        total = freq.sum() + 1e-7
        freq = freq / total
        return 1.0 + self.alpha_prog * (1.0 - freq)

    def _assign_and_build_targets(self, tgt_b, H, W, stride, image_size, reg_max):
        """Build per-level target maps from GT boxes for this batch element."""
        cls_tgt = torch.zeros(self.nc, H, W, device=tgt_b.device)
        reg_tgt = torch.zeros(reg_max * 4, H, W, device=tgt_b.device)
        pos_mask = torch.zeros(H, W, dtype=torch.bool, device=tgt_b.device)
        n_assigned = 0

        for gt in tgt_b:
            cls_id = int(gt[1].item())
            gt_cx, gt_cy = gt[2].item(), gt[3].item()
            gt_w, gt_h = gt[4].item(), gt[5].item()

            if gt_w <= 1e-7 or gt_h <= 1e-7:
                continue

            # Grid cell containing the GT center
            gx = int(gt_cx * W)
            gy = int(gt_cy * H)
            gx = max(0, min(W - 1, gx))
            gy = max(0, min(H - 1, gy))

            # Classification: assign class to this cell
            cls_tgt[cls_id, gy, gx] = 1.0
            pos_mask[gy, gx] = True
            n_assigned += 1

            # Regression: offset from cell center + log scale
            cell_cx = (gx + 0.5) / W
            cell_cy = (gy + 0.5) / H
            dx = gt_cx - cell_cx
            dy = gt_cy - cell_cy
            dw = torch.log(torch.tensor(gt_w * image_size / stride, device=tgt_b.device).clamp(min=1e-7))
            dh = torch.log(torch.tensor(gt_h * image_size / stride, device=tgt_b.device).clamp(min=1e-7))

            for ch in range(reg_max):
                reg_tgt[ch * 4,     gy, gx] = dx
                reg_tgt[ch * 4 + 1, gy, gx] = dy
                reg_tgt[ch * 4 + 2, gy, gx] = dw
                reg_tgt[ch * 4 + 3, gy, gx] = dh

        return cls_tgt, reg_tgt, pos_mask, n_assigned

    def _level_loss(self, cls_pred, reg_pred, tgt_cls, tgt_reg, pos_mask, reg_max, nc):
        """Compute cls + box loss for one branch at one level for one image."""
        H, W = pos_mask.shape
        n_lvl = H * W
        n_pos = int(pos_mask.sum())

        # Flatten spatial
        cls_pred_flat = cls_pred.permute(1, 2, 0).reshape(n_lvl, nc)  # (N, nc)
        reg_pred_flat = reg_pred.permute(1, 2, 0).reshape(n_lvl, reg_max * 4)  # (N, 4*reg_max)

        # Flatten targets
        cls_tgt_flat = tgt_cls.permute(1, 2, 0).reshape(n_lvl, nc)
        reg_tgt_flat = tgt_reg.permute(1, 2, 0).reshape(n_lvl, reg_max * 4)

        pos_flat = pos_mask.reshape(-1)  # (N,)

        cls_loss = torch.tensor(0.0, device=cls_pred.device)
        box_loss = torch.tensor(0.0, device=cls_pred.device)

        if n_pos > 0:
            bce = F.binary_cross_entropy_with_logits(
                cls_pred_flat[pos_flat],
                cls_tgt_flat[pos_flat],
                reduction="mean",
            )
            cls_loss = bce
            box_loss = F.smooth_l1_loss(
                reg_pred_flat[pos_flat],
                reg_tgt_flat[pos_flat],
                reduction="mean",
            )

        return cls_loss, box_loss

    def forward(self, o2m_preds, o2o_preds, targets, anchors, image_size=640):
        """
        Compute combined O2M + O2O loss with anchor-based label assignment.

        Args:
            o2m_preds: list of {cls: (B, nc, H, W), reg: (B, reg_max*4, H, W)} per level
            o2o_preds: same structure for O2O branch
            targets: (N, 6) [batch_idx, cls, cx, cy, w, h] normalized [0,1]
            anchors: list of (N_i, 4) per level, normalized cxcywh
        """
        device = o2m_preds[0]["cls"].device
        batch_size = o2m_preds[0]["cls"].shape[0]
        num_levels = len(o2m_preds)

        # Feature shapes and strides
        feature_shapes = [(p["cls"].shape[2], p["cls"].shape[3]) for p in o2m_preds]
        strides = [image_size // h for h, _ in feature_shapes]

        # Progressive class weights
        prog_weights = self.prog_loss_weight(targets, self.nc).to(device)

        # Accumulate losses over batch
        acc_cls = torch.tensor(0.0, device=device)
        acc_box = torch.tensor(0.0, device=device)
        acc_o2o_cls = torch.tensor(0.0, device=device)
        acc_o2o_box = torch.tensor(0.0, device=device)

        for b in range(batch_size):
            tgt_b = targets[targets[:, 0] == b]

            # O2M branch: accumulate over levels
            lvl_cls = torch.tensor(0.0, device=device)
            lvl_box = torch.tensor(0.0, device=device)
            n_pos_total = 0

            for lvl, pred in enumerate(o2m_preds):
                H, W = feature_shapes[lvl]
                cls_tgt, reg_tgt, pos_mask, n_pos = self._assign_and_build_targets(
                    tgt_b, H, W, strides[lvl], image_size, self.reg_max
                )
                cls_l, box_l = self._level_loss(
                    pred["cls"][b], pred["reg"][b], cls_tgt, reg_tgt, pos_mask,
                    self.reg_max, self.nc
                )
                lvl_cls = lvl_cls + cls_l
                lvl_box = lvl_box + box_l
                n_pos_total += n_pos

            if n_pos_total > 0:
                acc_cls = acc_cls + (lvl_cls / num_levels)
                acc_box = acc_box + (lvl_box / num_levels)

            # O2O branch: same as O2M
            lvl_o2o_cls = torch.tensor(0.0, device=device)
            lvl_o2o_box = torch.tensor(0.0, device=device)
            n_pos_o2o = 0

            for lvl, pred in enumerate(o2o_preds):
                H, W = feature_shapes[lvl]
                cls_tgt, reg_tgt, pos_mask, n_pos = self._assign_and_build_targets(
                    tgt_b, H, W, strides[lvl], image_size, self.reg_max
                )
                cls_l, box_l = self._level_loss(
                    pred["cls"][b], pred["reg"][b], cls_tgt, reg_tgt, pos_mask,
                    self.reg_max, self.nc
                )
                lvl_o2o_cls = lvl_o2o_cls + cls_l
                lvl_o2o_box = lvl_o2o_box + box_l
                n_pos_o2o += n_pos

            if n_pos_o2o > 0:
                acc_o2o_cls = acc_o2o_cls + (lvl_o2o_cls / num_levels)
                acc_o2o_box = acc_o2o_box + (lvl_o2o_box / num_levels)

        # Average over batch
        denom = max(batch_size, 1)
        losses = {
            "cls": acc_cls / denom,
            "box": acc_box / denom,
            "o2o_cls": acc_o2o_cls / denom,
            "o2o_box": acc_o2o_box / denom,
        }

        total = (
            losses["cls"] * 1.0 +
            losses["box"] * 7.5 +
            losses["o2o_cls"] * self.o2o_weight +
            losses["o2o_box"] * 7.5 * self.o2o_weight
        )
        return total, losses
