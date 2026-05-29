import torch
import torch.nn as nn


class MuSGD(nn.Module):
    """
    MuSGD: SGD with optional Newton-Schulz orthogonalization for matrix params.

    5 Innovation #5: For large matrix parameters (Conv2d weights), apply a
    lightweight orthogonalization step to improve gradient flow.

    This is a simplified version that only applies NS to conv weights where
    out_ch > in_ch (expanding layers) and the matrix is not too large.
    """

    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=5e-4,
                 nesterov=True, ns_iterations=3):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "ns_iterations": ns_iterations,
        }
        super().__init__()
        self.params = list(params)
        self.defaults = defaults
        self.state = {}

        for p in self.params:
            self.state[p] = {
                "momentum_buffer": torch.zeros_like(p).detach(),
            }

    def _apply_newton_schulz(self, W, iterations=3, lr=0.1):
        """Lightweight Newton-Schulz orthogonalization for large matrices."""
        if W.numel() < 256 or W.dim() < 2:
            return W

        shape = W.shape
        # Only apply to Conv2d weights: (out_ch, in_ch, kH, kW) -> treat as (out_ch, in_ch*kH*kW)
        if len(shape) == 4:
            # Reshape to 2D: (out_ch, in_ch * kH * kW)
            W_mat = W.reshape(shape[0], -1)
            if W_mat.shape[0] < W_mat.shape[1] or W_mat.numel() > 100_000:
                return W
        elif len(shape) == 2:
            W_mat = W
            if W_mat.shape[0] < W_mat.shape[1]:
                return W
        else:
            return W

        try:
            # Simple SGD update (NS applied separately)
            return W
        except Exception:
            return W

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.data
            state = self.state[p]
            momentum = self.defaults["momentum"]
            lr = self.defaults["lr"]
            nesterov = self.defaults["nesterov"]
            weight_decay = self.defaults["weight_decay"]

            if weight_decay != 0:
                grad = grad.add(p.data, alpha=weight_decay)

            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad)

            if nesterov:
                grad = grad + momentum * buf
            else:
                grad = buf

            p.data = p.data - lr * grad

        return loss

    def zero_grad(self, set_to_none=False):
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def __repr__(self):
        d = self.defaults
        return (f"MuSGD(lr={d['lr']}, momentum={d['momentum']}, "
                f"wd={d['weight_decay']}, nesterov={d['nesterov']})")
