import torch
import torch.nn as nn


class MuSGD(torch.optim.SGD):
    """
    MuSGD: SGD with optional Newton-Schulz orthogonalization for matrix params.

    5th Innovation: For Conv2d weight matrices, apply a lightweight
    Newton-Schulz orthogonalization step to improve gradient flow and
    reduce co-adaptation of filters.

    Algorithm (Goldfarb et al. 2020, adapted):
        W <- W - lr * grad
        if eligible: W <- W / ||W||_F * sqrt(m * n)
        if eligible: repeat 3x: W <- (3/4) * W + (3/4) * W @ W.T @ W + (1/4) * W @ (W.T @ W)
    """

    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=5e-4,
                 nesterov=True, ns_iterations=3, ns_lr=0.1):
        super().__init__(params, lr=lr, momentum=momentum,
                         weight_decay=weight_decay, nesterov=nesterov)
        self.ns_iterations = ns_iterations
        self.ns_lr = ns_lr

    def step(self, closure=None):
        loss = super().step(closure)

        # Apply Newton-Schulz orthogonalization to eligible Conv2d weights
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self._is_eligible(p):
                    continue
                self._apply_newton_schulz(p, group["lr"])

        return loss

    def _is_eligible(self, p):
        """Only apply NS to 4D Conv2d weights that are large enough."""
        if p.dim() != 4:
            return False
        out_ch, in_ch, kH, kW = p.shape
        mat = out_ch * in_ch * kH * kW
        # Apply to expanding or neutral layers with enough elements
        return mat >= 512 and out_ch >= 16

    def _apply_newton_schulz(self, W, lr=0.1, iterations=3):
        """
        Newton-Schulz iteration for (out_ch, in_ch * kH * kW) matrices.
        Orthogonalizes columns so W.T @ W ≈ I.
        """
        shape = W.shape
        out_ch, in_ch, kH, kW = shape
        m, n = out_ch, in_ch * kH * kW

        W_mat = W.reshape(out_ch, -1)

        # Normalize: scale to identity covariance
        target_norm = (m * n) ** 0.5
        current_norm = W_mat.norm()
        if current_norm < 1e-7:
            return
        W_mat = W_mat * (target_norm / current_norm)

        # Newton-Schulz iterations
        for _ in range(iterations):
            WWT = W_mat @ W_mat.T
            WTW = W_mat.T @ W_mat
            I_m = torch.eye(m, device=W.device, dtype=W_mat.dtype)
            I_n = torch.eye(n, device=W.device, dtype=W_mat.dtype)

            # W <- (3/4)W + (3/4)W(W.T W) + (1/4)W(W W.T)W
            # Simplified: W_new = W + alpha * (I_n - WTW) @ W + beta * W @ (I_m - WWT)
            alpha = 0.5 * lr
            beta = 0.5 * lr

            # Gradient step toward orthogonality: penalize ||W.T @ W - I|| and ||W @ W.T - I||
            grad_orth = (W_mat @ (I_n - WTW) * alpha) + ((I_m - WWT) @ W_mat * beta)
            W_mat = W_mat + grad_orth

            # Re-normalize
            current_norm = W_mat.norm()
            if current_norm < 1e-7:
                break
            W_mat = W_mat * (target_norm / current_norm)

        W.data = W_mat.reshape(shape)
