import torch
import torch.nn as nn


class MuSGD(torch.optim.SGD):
    """
    MuSGD: SGD with optional Newton-Schulz orthogonalization for matrix params.

    5th Innovation: For Conv2d weight matrices, apply a lightweight
    Newton-Schulz orthogonalization step to improve gradient flow and
    reduce co-adaptation of filters.

    Correct application: runs BEFORE super().step() so it modifies gradients
    before the weight update, compatible with AMP GradScaler.

    Algorithm:
        grad <- grad + weight_decay * W
        if eligible: W <- W - lr * (grad + NS correction)
        NS correction: penalize ||W.T @ W - I|| and ||W @ W.T - I||
    """

    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=5e-4,
                 nesterov=True, ns_iterations=3, ns_lr=0.1, ns_warmup=999999):
        super().__init__(params, lr=lr, momentum=momentum,
                         weight_decay=weight_decay, nesterov=nesterov)
        self.ns_iterations = ns_iterations
        self.ns_lr = ns_lr
        self.ns_warmup = ns_warmup
        self._step_count = 0

    def step(self, closure=None):
        self._step_count += 1

        # Apply Newton-Schulz gradient correction before the SGD step
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self._is_eligible(p):
                    continue
                # Only apply after warmup to let model stabilize first
                if self._step_count <= self.ns_warmup:
                    continue
                self._apply_ns_correction(p, group["lr"])

        return super().step(closure)

    def _is_eligible(self, p):
        """Only apply NS to 4D Conv2d weights that are large enough."""
        if p.dim() != 4:
            return False
        out_ch, in_ch, kH, kW = p.shape
        mat = out_ch * in_ch * kH * kW
        return mat >= 512 and out_ch >= 16

    def _apply_ns_correction(self, W, lr=0.1):
        """
        Newton-Schulz gradient correction for (out_ch, in_ch * kH * kW) matrices.
        Adds gradient penalty for orthogonality to reduce filter co-adaptation.

        Correct analytical gradients (from ||W.T @ W - I||^2):
            d/dW ||W.T @ W - I||^2 = 4 * W @ (W.T @ W - I)  [shape: m x n]
            d/dW ||W @ W.T - I||^2 = 4 * (W @ W.T - I) @ W  [shape: m x n]
        Combined: grad = 4 * (W @ (W.T @ W - I) + (W @ W.T - I) @ W)

        This modifies p.grad in-place before super().step() applies the update.
        Compatible with AMP GradScaler since it runs before the scaler step.
        """
        if W.grad is None:
            return

        shape = W.shape
        out_ch, in_ch, kH, kW = shape
        m, n = out_ch, in_ch * kH * kW

        W_mat = W.reshape(out_ch, -1)

        WTW = W_mat.T @ W_mat
        I_n = torch.eye(n, device=W.device, dtype=W_mat.dtype)
        WWT = W_mat @ W_mat.T
        I_m = torch.eye(m, device=W.device, dtype=W_mat.dtype)

        # Correct analytical gradient with factor 4
        grad_orth = 4.0 * (W_mat @ (WTW - I_n) + (WWT - I_m) @ W_mat)

        # Apply correction: very small scale to avoid destabilizing training
        alpha = self.ns_lr * 0.01
        W.grad.add_(grad_orth.reshape(shape), alpha=alpha)
