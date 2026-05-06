import json
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_beta(beta):
    if isinstance(beta, str) and beta.lower() in {"inf", "infty", "infinity"}:
        return math.inf
    return float(beta)


def centered_logits(logits):
    return logits - logits.mean(dim=-1, keepdim=True)


def hard_teacher_margin_loss(student_logits, teacher_logits):
    teacher_argmax = teacher_logits.argmax(dim=-1)
    student_top = student_logits.max(dim=-1).values
    teacher_energy = student_logits.gather(dim=-1, index=teacher_argmax.unsqueeze(-1)).squeeze(-1)
    return (student_top - teacher_energy).mean()


def distill_loss(student_logits, teacher_logits, targets, loss_type, beta):
    vocab = student_logits.size(-1)
    s = student_logits.reshape(-1, vocab).float()
    t = teacher_logits.reshape(-1, vocab).float()
    if loss_type == "mse":
        s_centered = centered_logits(s)
        t_centered = centered_logits(t)
        loss = F.mse_loss(s_centered, t_centered)
        with torch.no_grad():
            raw_mse = F.mse_loss(s, t)
        return loss, {"train_loss_centered_logit_mse": loss.detach(), "train_loss_raw_logit_mse": raw_mse.detach()}
    if math.isinf(beta):
        loss = hard_teacher_margin_loss(s, t)
        return loss, {"train_loss_hard_teacher_margin": loss.detach()}
    p = F.softmax(beta * t, dim=-1)
    log_q = F.log_softmax(beta * s, dim=-1)
    unscaled_loss = -(p * log_q).sum(dim=-1).mean()
    loss = unscaled_loss / beta
    with torch.no_grad():
        kl = (p * (torch.log(p.clamp_min(1e-30)) - log_q)).sum(dim=-1).mean()
    return loss, {
        "train_loss_temp_xent": loss.detach(),
        "train_loss_temp_xent_unscaled": unscaled_loss.detach(),
        "train_kl_beta": kl.detach(),
    }


@torch.no_grad()
def categorical_observables(logits, beta, prefix):
    if math.isinf(beta):
        return {
            f"{prefix}_entropy_beta": 0.0,
            f"{prefix}_top1_conf_beta": 1.0,
            f"{prefix}_effective_support_beta": 1.0,
            f"{prefix}_fisher_trace_beta": 0.0,
        }
    probs = F.softmax(beta * logits, dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-30))
    entropy = -(probs * log_probs).sum(dim=-1)
    return {
        f"{prefix}_entropy_beta": entropy.mean().item(),
        f"{prefix}_top1_conf_beta": probs.max(dim=-1).values.mean().item(),
        f"{prefix}_effective_support_beta": torch.exp(entropy).mean().item(),
        f"{prefix}_fisher_trace_beta": (beta**2 * (1.0 - probs.square().sum(dim=-1))).mean().item(),
    }


@torch.no_grad()
def fixed_beta_prime_metrics(s, t, beta_prime=1.0):
    p = F.softmax(beta_prime * t, dim=-1)
    q = F.softmax(beta_prime * s, dim=-1)
    log_p = torch.log(p.clamp_min(1e-30))
    log_q = torch.log(q.clamp_min(1e-30))
    student_entropy = -(q * log_q).sum(dim=-1)
    return {
        "kl_beta_prime_1": (p * (log_p - log_q)).sum(dim=-1).mean().item(),
        "xent_beta_prime_1": -(p * log_q).sum(dim=-1).mean().item(),
        "teacher_entropy_beta_prime_1": -(p * log_p).sum(dim=-1).mean().item(),
        "student_entropy_beta_prime_1": student_entropy.mean().item(),
        "student_fisher_trace_beta_prime_1": (
            beta_prime**2 * (1.0 - q.square().sum(dim=-1))
        ).mean().item(),
    }


@torch.no_grad()
def scalar_metrics(student_logits, teacher_logits, targets, beta):
    vocab = student_logits.size(-1)
    s = student_logits.reshape(-1, vocab).float()
    t = teacher_logits.reshape(-1, vocab).float()
    y = targets.reshape(-1)
    out = {}
    out.update(fixed_beta_prime_metrics(s, t, beta_prime=1.0))
    out["hard_xent"] = F.cross_entropy(s, y).item()
    out["logit_mse"] = F.mse_loss(s, t).item()
    out["centered_logit_mse"] = F.mse_loss(centered_logits(s), centered_logits(t)).item()
    out["log_softmax_mse"] = F.mse_loss(F.log_softmax(s, dim=-1), F.log_softmax(t, dim=-1)).item()
    out["student_true_acc"] = (s.argmax(dim=-1) == y).float().mean().item()
    teacher_argmax = t.argmax(dim=-1)
    out["teacher_true_acc"] = (teacher_argmax == y).float().mean().item()
    out["student_teacher_argmax_acc"] = (s.argmax(dim=-1) == teacher_argmax).float().mean().item()
    out["hard_teacher_xent"] = F.cross_entropy(s, teacher_argmax).item()
    out["hard_teacher_margin"] = hard_teacher_margin_loss(s, t).item()
    out["teacher_margin"] = (t.topk(2, dim=-1).values[:, 0] - t.topk(2, dim=-1).values[:, 1]).mean().item()
    out.update(categorical_observables(s, beta, "student"))
    out.update(categorical_observables(t, beta, "teacher"))
    if math.isinf(beta):
        p = F.one_hot(teacher_argmax, num_classes=vocab).float()
        q = F.one_hot(s.argmax(dim=-1), num_classes=vocab).float()
        out["distill_eval_loss"] = out["hard_teacher_margin"]
        out["ce_grad_signal_norm_beta"] = (q - p).norm(dim=-1).mean().item()
        return out
    p = F.softmax(beta * t, dim=-1)
    q = F.softmax(beta * s, dim=-1)
    log_q = torch.log(q.clamp_min(1e-30))
    unscaled_distill_eval_loss = -(p * log_q).sum(dim=-1).mean()
    out["distill_eval_loss"] = (unscaled_distill_eval_loss / beta).item()
    out["distill_eval_loss_unscaled"] = unscaled_distill_eval_loss.item()
    out["kl_beta"] = (p * (torch.log(p.clamp_min(1e-30)) - log_q)).sum(dim=-1).mean().item()
    out["ce_grad_signal_norm_beta"] = (beta * (q - p).norm(dim=-1)).mean().item()
    return out


def grad_norm(parameters):
    sq = 0.0
    for p in parameters:
        if p.grad is not None:
            sq += p.grad.detach().float().pow(2).sum().item()
    return math.sqrt(sq)


@torch.no_grad()
def param_update_norm(before, parameters):
    sq = 0.0
    for old, p in zip(before, parameters):
        sq += (p.detach().float() - old).pow(2).sum().item()
    return math.sqrt(sq)


def selected_named_parameters(model, regex):
    pat = re.compile(regex)
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad and pat.search(n)]


def empirical_kernel(model, x, teacher_logits, params, max_tokens=16):
    model.eval()
    bsz = min(x.size(0), max_tokens)
    x = x[:bsz]
    target_ids = teacher_logits[:bsz, -1].argmax(dim=-1)
    rows = []
    for i in range(bsz):
        model.zero_grad(set_to_none=True)
        logits = model(x[i : i + 1])
        scalar = logits[0, -1, target_ids[i]]
        grads = torch.autograd.grad(scalar, [p for _, p in params], retain_graph=False, allow_unused=True)
        flat = [g.detach().flatten().float().cpu() for g in grads if g is not None]
        rows.append(torch.cat(flat) if flat else torch.zeros(1))
    jac = torch.stack(rows)
    gram = jac @ jac.T
    denom = torch.linalg.matrix_norm(gram).clamp_min(1e-12)
    return gram / denom


@torch.no_grad()
def target_kernel_from_teacher(teacher_logits, beta, max_tokens=16):
    t = teacher_logits[:max_tokens, -1].float()
    if math.isinf(beta):
        y = F.one_hot(t.argmax(dim=-1), num_classes=t.size(-1)).float()
    else:
        y = F.softmax(beta * t, dim=-1)
    gram = y @ y.T
    return gram / torch.linalg.matrix_norm(gram).clamp_min(1e-12)


def kernel_overlaps(k, k0, kt):
    def align(a, b):
        b = b.to(device=a.device, dtype=a.dtype)
        return (a * b).sum().item() / (torch.linalg.matrix_norm(a).item() * torch.linalg.matrix_norm(b).item() + 1e-12)

    eig = torch.linalg.eigvalsh(k.float()).clamp_min(0)
    participation = eig.sum().square() / eig.square().sum().clamp_min(1e-12)
    return {
        "kernel_overlap_initial": align(k, k0),
        "kernel_overlap_teacher_target": align(k, kt),
        "kernel_effective_rank": participation.item(),
    }


class JsonlLogger:
    def __init__(self, path, mode="w"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open(mode, buffering=1)

    def write(self, row):
        clean = {}
        for k, v in row.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().float().item()
            clean[k] = v
        self.f.write(json.dumps(clean, sort_keys=True) + "\n")

    def close(self):
        self.f.close()
