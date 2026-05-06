import torch


class Muon(torch.optim.Optimizer):
    """Small Muon implementation for 2D matrix parameters.

    Non-matrix parameters should be handled by a separate AdamW group. This
    optimizer follows the common momentum + Newton-Schulz orthogonalized update
    recipe used in recent open-source Muon implementations.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, ns_steps=5, eps=1e-7):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps, eps=eps)
        super().__init__(params, defaults)

    @staticmethod
    def _zeropower_via_newtonschulz5(g, steps: int, eps: float):
        if g.ndim != 2:
            return g
        transposed = False
        if g.size(0) > g.size(1):
            g = g.T
            transposed = True
        g = g / (g.norm() + eps)
        a, b, c = 3.4445, -4.7750, 2.0315
        x = g
        for _ in range(steps):
            xx_t = x @ x.T
            x = a * x + (b * xx_t + c * xx_t @ xx_t) @ x
        if transposed:
            x = x.T
        return x

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if weight_decay:
                    p.mul_(1 - lr * weight_decay)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = self._zeropower_via_newtonschulz5(buf.float(), ns_steps, eps).to(dtype=p.dtype)
                if p.ndim == 2:
                    update = update * max(1.0, p.size(0) / max(p.size(1), 1)) ** 0.5
                p.add_(update, alpha=-lr)
        return loss


class OptimBundle:
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        for opt in self.optimizers:
            opt.step()

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]


def build_optimizer(model, name, lr, weight_decay, sgd_momentum=0.0, muon_momentum=0.95, muon_ns_steps=5):
    name = name.lower()
    groups = model.param_groups(lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=sgd_momentum)
    if name == "muon":
        matrix_params, other_groups = [], []
        for group in groups:
            matrix = [p for p in group["params"] if p.ndim == 2]
            other = [p for p in group["params"] if p.ndim != 2]
            if matrix:
                matrix_params.append(
                    {
                        "params": matrix,
                        "lr": group["lr"],
                        "weight_decay": group["weight_decay"],
                        "momentum": muon_momentum,
                        "ns_steps": muon_ns_steps,
                    }
                )
            if other:
                other_groups.append({**group, "params": other})
        muon = Muon(matrix_params, lr=lr, momentum=muon_momentum, weight_decay=weight_decay, ns_steps=muon_ns_steps)
        adam = torch.optim.AdamW(other_groups, betas=(0.9, 0.95), eps=1e-8) if other_groups else None
        return OptimBundle([opt for opt in [muon, adam] if opt is not None])
    raise ValueError(f"Unknown optimizer: {name}")

