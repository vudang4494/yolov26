import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class YOLOv26Loss(nn.Module):
    """
    YOLOv26 Loss with 5 innovations:
    1. DFL Removal: reg_max=4 (L1 Smooth instead of DFL softmax)
    2. NMS-Free O2O: Top-K matching on O2O branch (Hungarian matching)
    3. ProgLoss: progressive class re-weighting — boosts rare classes
    4. STAL: small-target-aware label assignment
    5. MuSGD: optimizer-level (handled in musgd.py)

    Label assignment: anchor-based matching per feature level.
    Each GT box is assigned to the grid cell containing its center.
    """

    def __init__(self, num_classes=80, reg_max=4, alpha_prog=0.3, beta_stal=0.3,
                 o2o_weight=1.0, box_weight=7.5, cls_weight=1.0):
        super().__init__()
        self.nc = num_classes
        self.reg_max = reg_max
        self.alpha_prog = alpha_prog
        self.beta_stal = beta_stal
        self.o2o_weight = o2o_weight
        self.box_weight = box_weight
        self.cls_weight = cls_weight

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

    def _stal_weight(self, gt_w, gt_h, image_size, stride):
        """
        STAL: small-target-aware label assignment.
        Smaller objects get higher weight — computed from box area ratio.
        """
        box_area_px = gt_w * image_size * gt_h * image_size
        max_area = image_size * image_size
        area_ratio = box_area_px / max_area
        stal = 1.0 + self.beta_stal * (1.0 - area_ratio)
        return max(1.0, min(2.0, stal))

    def _assign_and_build_targets(self, tgt_b, H, W, stride, image_size, reg_max):
        """Build per-level target maps from GT boxes for this batch element."""
        cls_tgt = torch.zeros(self.nc, H, W, device=tgt_b.device)
        reg_tgt = torch.zeros(reg_max * 4, H, W, device=tgt_b.device)
        stal_tgt = torch.zeros(H, W, device=tgt_b.device)
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

            # STAL weight for this GT box
            stal_w = self._stal_weight(gt_w, gt_h, image_size, stride)
            stal_tgt[gy, gx] = stal_w

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

        return cls_tgt, reg_tgt, stal_tgt, pos_mask, n_assigned

    def _level_loss(self, cls_pred, reg_pred, tgt_cls, tgt_reg, tgt_stal, pos_mask, reg_max, nc, prog_weights):
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
        stal_flat = tgt_stal.reshape(-1)
        pos_flat = pos_mask.reshape(-1)

        cls_loss = torch.tensor(0.0, device=cls_pred.device)
        box_loss = torch.tensor(0.0, device=cls_pred.device)

        if n_pos > 0:
            pos_indices = pos_flat.nonzero(as_tuple=True)[0]

            # Classification: BCE weighted by ProgLoss class weights
            pos_cls_pred = cls_pred_flat[pos_indices]
            pos_cls_tgt = cls_tgt_flat[pos_indices]
            bce_per_sample = F.binary_cross_entropy_with_logits(
                pos_cls_pred, pos_cls_tgt, reduction="none"
            )
            cls_ids = pos_cls_tgt.argmax(dim=1)
            prog_w = prog_weights[cls_ids]
            bce_weighted = (bce_per_sample.mean(dim=1) * prog_w).mean()
            cls_loss = bce_weighted

            # Regression: SmoothL1 weighted by STAL
            pos_reg_pred = reg_pred_flat[pos_indices]
            pos_reg_tgt = reg_tgt_flat[pos_indices]
            pos_stal = stal_flat[pos_indices]
            box_diff = F.smooth_l1_loss(pos_reg_pred, pos_reg_tgt, reduction="none")
            box_per_sample = box_diff.mean(dim=1)
            box_loss = (box_per_sample * pos_stal).mean()

        return cls_loss, box_loss

    def _hungarian_matching(self, cls_pred, reg_pred, tgt_b, feature_shape, stride, image_size):
        """
        O2O branch: Hungarian matching between predictions and GT boxes.
        Produces one prediction per GT using optimal bipartite matching.
        """
        H, W = feature_shape
        n_lvl = H * W
        nc = self.nc

        # Flatten predictions
        cls_flat = cls_pred.permute(1, 2, 0).reshape(n_lvl, nc).sigmoid()
        reg_flat = reg_pred.permute(1, 2, 0).reshape(n_lvl, self.reg_max * 4)

        # Decode regression: softmax over reg_max, weighted sum
        reg_soft = reg_flat.view(n_lvl, 4, self.reg_max)
        reg_soft = F.softmax(reg_soft, dim=2)
        reg_range = torch.arange(self.reg_max, device=reg_pred.device, dtype=reg_soft.dtype)
        reg_range = reg_range.view(1, self.reg_max, 1)
        reg_decoded = (reg_soft * reg_range).sum(dim=2)  # (N, 4)

        n_gt = tgt_b.shape[0]
        if n_gt == 0:
            return torch.zeros(0, device=reg_pred.device), torch.zeros(0, device=reg_pred.device), \
                   torch.zeros(0, dtype=torch.long, device=reg_pred.device)

        # Build cost matrix: cost[i, j] = cost of matching pred_i to gt_j
        # Cost = classification cost + box regression cost
        try:
            from scipy.optimize import linear_sum_assignment
            has_scipy = True
        except ImportError:
            has_scipy = False

        if not has_scipy or n_gt == 0:
            # Fallback: simple top-1 per GT
            matched_cls = []
            matched_box = []
            matched_idx = []
            for i, gt in enumerate(tgt_b):
                _, best_j = cls_flat[:, int(gt[1].item())].max(dim=0)
                matched_cls.append(cls_flat[best_j])
                matched_box.append(reg_decoded[best_j])
                matched_idx.append(best_j.item())
            return torch.stack(matched_cls) if matched_cls else torch.zeros(0, nc, device=reg_pred.device), \
                   torch.stack(matched_box) if matched_box else torch.zeros(0, 4, device=reg_pred.device), \
                   torch.tensor(matched_idx, dtype=torch.long, device=reg_pred.device)

        # Hungarian matching using scipy
        cost_matrix = torch.zeros(n_lvl, n_gt, device=reg_pred.device)

        for j, gt in enumerate(tgt_b):
            cls_id = int(gt[1].item())
            gt_cx, gt_cy = gt[2].item(), gt[3].item()
            gt_w, gt_h = gt[4].item(), gt[5].item()

            # Classification cost: negative log-likelihood
            cls_cost = -cls_flat[:, cls_id]

            # Box regression cost: L1 distance
            cell_cx = (torch.arange(W, device=reg_pred.device) + 0.5) / W
            cell_cy = (torch.arange(H, device=reg_pred.device) + 0.5) / H
            cy_grid, cx_grid = torch.meshgrid(cell_cy, cell_cx, indexing="ij")
            cx_flat = cx_grid.reshape(-1)
            cy_flat = cy_grid.reshape(-1)

            dx = reg_decoded[:, 0]
            dy = reg_decoded[:, 1]
            dw_pred = reg_decoded[:, 2]
            dh_pred = reg_decoded[:, 3]

            pred_cx = cx_flat + dx * (stride / image_size)
            pred_cy = cy_flat + dy * (stride / image_size)
            pred_w = torch.exp(dw_pred.clamp(max=math.log(image_size * 2))) * (stride / image_size)
            pred_h = torch.exp(dh_pred.clamp(max=math.log(image_size * 2))) * (stride / image_size)

            box_cost = (pred_cx - gt_cx).abs() + (pred_cy - gt_cy).abs() + \
                       (pred_w - gt_w).abs() + (pred_h - gt_h).abs()

            cost_matrix[:, j] = cls_cost + 0.1 * box_cost

        cost_np = cost_matrix.detach().cpu().numpy()
        row_indices, col_indices = linear_sum_assignment(cost_np)
        # row_indices[j] = GT matched to prediction row j
        # col_indices[j] = prediction matched to GT column j (j=0..n_gt-1)
        matched_cls = cls_flat[col_indices]     # (n_gt, nc)
        matched_reg = reg_decoded[col_indices]  # (n_gt, 4)
        return matched_cls, matched_reg, torch.tensor(col_indices, dtype=torch.long, device=reg_pred.device)

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

        # Progressive class weights (ProgLoss Innovation #3)
        prog_weights = self.prog_loss_weight(targets, self.nc).to(device)

        # Accumulate losses over batch
        acc_cls = torch.tensor(0.0, device=device)
        acc_box = torch.tensor(0.0, device=device)
        acc_o2o_cls = torch.tensor(0.0, device=device)
        acc_o2o_box = torch.tensor(0.0, device=device)

        for b in range(batch_size):
            tgt_b = targets[targets[:, 0] == b]

            # O2M branch: per-level anchor-based matching (Innovation #4 STAL)
            lvl_cls = torch.tensor(0.0, device=device)
            lvl_box = torch.tensor(0.0, device=device)
            n_pos_total = 0

            for lvl, pred in enumerate(o2m_preds):
                H, W = feature_shapes[lvl]
                cls_tgt, reg_tgt, stal_tgt, pos_mask, n_pos = self._assign_and_build_targets(
                    tgt_b, H, W, strides[lvl], image_size, self.reg_max
                )
                cls_l, box_l = self._level_loss(
                    pred["cls"][b], pred["reg"][b], cls_tgt, reg_tgt, stal_tgt, pos_mask,
                    self.reg_max, self.nc, prog_weights
                )
                lvl_cls = lvl_cls + cls_l
                lvl_box = lvl_box + box_l
                n_pos_total += n_pos

            if n_pos_total > 0:
                acc_cls = acc_cls + (lvl_cls / num_levels)
                acc_box = acc_box + (lvl_box / num_levels)

            # O2O branch: Hungarian matching (Innovation #2)
            o2o_cls_sum = torch.tensor(0.0, device=device)
            o2o_box_sum = torch.tensor(0.0, device=device)
            n_levels_with_matches = 0

            for lvl, pred in enumerate(o2o_preds):
                H, W = feature_shapes[lvl]
                lvl_tgt = tgt_b
                if lvl_tgt.shape[0] == 0:
                    continue

                matched_cls, matched_reg, matched_idx = self._hungarian_matching(
                    pred["cls"][b], pred["reg"][b], lvl_tgt,
                    (H, W), strides[lvl], image_size
                )

                if matched_cls.shape[0] > 0:
                    # Safe BCE: use clamped sigmoid probabilities
                    matched_prob = matched_cls.clamp(min=1e-7, max=1 - 1e-7)
                    gt_cls_tgt = F.one_hot(
                        lvl_tgt[:, 1].long(), num_classes=self.nc
                    ).float().to(device)
                    bce = -(gt_cls_tgt * matched_prob.log() +
                             (1 - gt_cls_tgt) * (1 - matched_prob).log()).mean()
                    o2o_cls_sum = o2o_cls_sum + bce

                    # Box loss: Hungarian aligns matched_reg[i] with lvl_tgt[i].
                    # Use the SAME decode as the Hungarian cost function:
                    #   decoded_cx = cell_cx + dx * (stride/image_size)  (normalized)
                    #   decoded_w  = exp(dw) * (stride/image_size)       (normalized)
                    # GT is already in normalized [0,1] space.
                    stride = strides[lvl]
                    s_over_I = stride / image_size

                    dx = matched_reg[:, 0]
                    dy = matched_reg[:, 1]
                    dw = matched_reg[:, 2]
                    dh = matched_reg[:, 3]

                    # Decode: same formula as Hungarian cost function
                    pred_cx_grid = (torch.arange(W, device=device, dtype=matched_reg.dtype) + 0.5) / W
                    pred_cy_grid = (torch.arange(H, device=device, dtype=matched_reg.dtype) + 0.5) / H
                    cy_g, cx_g = torch.meshgrid(pred_cy_grid, pred_cx_grid, indexing="ij")
                    cx_flat = cx_g.reshape(-1)
                    cy_flat = cy_g.reshape(-1)

                    pred_cx = cx_flat[matched_idx] + dx * s_over_I
                    pred_cy = cy_flat[matched_idx] + dy * s_over_I
                    pred_w = torch.exp(dw.clamp(max=math.log(image_size * 2))) * s_over_I
                    pred_h = torch.exp(dh.clamp(max=math.log(image_size * 2))) * s_over_I

                    # GT is already in normalized [0,1] space
                    gt_cx = lvl_tgt[:, 2]
                    gt_cy = lvl_tgt[:, 3]
                    gt_w = lvl_tgt[:, 4]
                    gt_h = lvl_tgt[:, 5]

                    box_l = (
                        F.smooth_l1_loss(pred_cx, gt_cx, reduction="mean") +
                        F.smooth_l1_loss(pred_cy, gt_cy, reduction="mean") +
                        F.smooth_l1_loss(pred_w, gt_w, reduction="mean") +
                        F.smooth_l1_loss(pred_h, gt_h, reduction="mean")
                    ) / 4.0
                    o2o_box_sum = o2o_box_sum + box_l
                    n_levels_with_matches += 1

            if n_levels_with_matches > 0:
                # Average over levels with matches, then over batch
                acc_o2o_cls = acc_o2o_cls + (o2o_cls_sum / n_levels_with_matches)
                acc_o2o_box = acc_o2o_box + (o2o_box_sum / n_levels_with_matches)

        # Average over batch
        denom = max(batch_size, 1)
        losses = {
            "cls": acc_cls / denom,
            "box": acc_box / denom,
            "o2o_cls": acc_o2o_cls / denom,
            "o2o_box": acc_o2o_box / denom,
        }

        total = (
            losses["cls"] * self.cls_weight +
            losses["box"] * self.box_weight +
            losses["o2o_cls"] * self.o2o_weight +
            losses["o2o_box"] * self.box_weight * self.o2o_weight
        )
        return total, losses
