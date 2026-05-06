import argparse
from contextlib import nullcontext
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from tempdistill.data import OnlineTokenBatcher, load_text_stream, make_probe_batch
from tempdistill.metrics import (
    JsonlLogger,
    distill_loss,
    empirical_kernel,
    grad_norm,
    kernel_overlaps,
    param_update_norm,
    parse_beta,
    scalar_metrics,
    selected_named_parameters,
    target_kernel_from_teacher,
)
from tempdistill.model import GPTConfig, MuPGPT
from tempdistill.optim import build_optimizer
from tempdistill.teacher_cache import (
    AsyncTeacherLogitBatcher,
    load_or_compute_probe_batch,
    teacher_cache_subdir,
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_model", default="EleutherAI/pythia-70m-deduped")
    p.add_argument("--dataset_name", default="allenai/c4")
    p.add_argument("--dataset_config", default="en")
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out_dir", default="runs_5000_centered_mse_student_obs")
    p.add_argument("--run_name", default=None)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=0)
    p.add_argument("--weights_out_dir", default="/n/netscratch/pehlevan_lab/Lab/sab")
    p.add_argument("--teacher_cache_dir", default="/n/netscratch/pehlevan_lab/Lab/sab/teacher_logit_cache")
    p.add_argument("--teacher_cache_mode", choices=["readwrite", "readonly", "off"], default="readwrite")
    p.add_argument("--teacher_cache_prefetch", type=int, default=2)
    p.add_argument("--teacher_cache_load_workers", type=int, default=4)
    p.add_argument("--teacher_cache_dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--student_n_layer", type=int, default=8)
    p.add_argument("--student_n_embd", type=int, default=512)
    p.add_argument("--student_n_head", type=int, default=8)
    p.add_argument("--student_base_width", type=int, default=256)
    p.add_argument("--student_dropout", type=float, default=0.0)
    p.add_argument("--loss_type", choices=["xent", "mse"], default="xent")
    p.add_argument("--beta", default="1")
    p.add_argument("--optimizer", choices=["adamw", "sgd", "muon"], default="adamw")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--sgd_momentum", type=float, default=0.0)
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--muon_ns_steps", type=int, default=5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--min_lr_frac", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--probe_batch_size", type=int, default=8)
    p.add_argument("--kernel_every", type=int, default=200)
    p.add_argument("--kernel_param_regex", default=r"blocks\.(6|7)|ln_f|lm_head")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_scale(step, warmup_steps, max_steps, min_lr_frac):
    if step < warmup_steps:
        return max(1e-8, (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_frac + (1.0 - min_lr_frac) * cosine


def set_optimizer_lr(optimizer, base_lrs, scale):
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * scale


def autocast_context(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def teacher_forward(teacher, x, precision):
    with autocast_context(x.device, precision):
        return teacher(x).logits


def main():
    args = get_args()
    beta = parse_beta(args.beta)
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    beta_name = "inf" if math.isinf(beta) else str(beta).replace(".", "p")
    run_name = args.run_name or f"{args.loss_type}_beta-{beta_name}_{args.optimizer}_seed{args.seed}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher_cache_dir = None
    if args.teacher_cache_mode != "off":
        teacher_cache_dir = teacher_cache_subdir(args.teacher_cache_dir, args, tokenizer)
    if args.precision == "bf16" and device.type == "cuda":
        teacher_dtype = torch.bfloat16
    elif args.precision == "fp16" and device.type == "cuda":
        teacher_dtype = torch.float16
    else:
        teacher_dtype = torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=teacher_dtype if device.type == "cuda" else torch.float32,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    cfg = GPTConfig(
        vocab_size=teacher.config.vocab_size,
        block_size=args.seq_len,
        n_layer=args.student_n_layer,
        n_head=args.student_n_head,
        n_embd=args.student_n_embd,
        base_width=args.student_base_width,
        dropout=args.student_dropout,
    )
    student = MuPGPT(cfg).to(device)
    if args.compile and hasattr(torch, "compile"):
        student = torch.compile(student)

    optimizer = build_optimizer(
        student,
        args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        sgd_momentum=args.sgd_momentum,
        muon_momentum=args.muon_momentum,
        muon_ns_steps=args.muon_ns_steps,
    )
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    train_ds = load_text_stream(args.dataset_name, args.dataset_config, args.dataset_split, args.streaming, args.seed)
    probe_ds = load_text_stream(args.dataset_name, args.dataset_config, args.dataset_split, args.streaming, args.seed + 123)
    batcher = OnlineTokenBatcher(train_ds, tokenizer, args.batch_size, args.seq_len, device, args.seed)
    if teacher_cache_dir is None:
        teacher_batcher = None
        probe_x, probe_y = make_probe_batch(probe_ds, tokenizer, args.probe_batch_size, args.seq_len, device)
        probe_teacher_logits = teacher_forward(teacher, probe_x, args.precision)
    else:
        teacher_batcher = None
        probe_x, probe_y, probe_teacher_logits = load_or_compute_probe_batch(
            cache_dir=teacher_cache_dir,
            mode=args.teacher_cache_mode,
            batch_builder=lambda: make_probe_batch(
                probe_ds,
                tokenizer,
                args.probe_batch_size,
                args.seq_len,
                device,
            ),
            teacher=teacher,
            forward_fn=teacher_forward,
            precision=args.precision,
            device=device,
            cache_dtype=args.teacher_cache_dtype,
        )
    kernel_params = selected_named_parameters(student, args.kernel_param_regex)
    k0 = empirical_kernel(student, probe_x, probe_teacher_logits, kernel_params) if args.kernel_every else None
    kt = target_kernel_from_teacher(probe_teacher_logits, beta) if args.kernel_every else None
    if teacher_cache_dir is not None:
        teacher_batcher = AsyncTeacherLogitBatcher(
            batcher=batcher,
            teacher=teacher,
            forward_fn=teacher_forward,
            precision=args.precision,
            cache_dir=teacher_cache_dir,
            mode=args.teacher_cache_mode,
            prefetch=args.teacher_cache_prefetch,
            device=device,
            cache_dtype=args.teacher_cache_dtype,
            load_workers=args.teacher_cache_load_workers,
        )

    logger = JsonlLogger(run_dir / "metrics.jsonl")
    n_params = sum(p.numel() for p in student.parameters())
    logger.write({
        "event": "start",
        "n_student_params": n_params,
        "device": str(device),
        "run_name": run_name,
        "teacher_cache_dir": str(teacher_cache_dir) if teacher_cache_dir is not None else None,
        "teacher_cache_mode": args.teacher_cache_mode,
        "teacher_cache_prefetch": args.teacher_cache_prefetch if teacher_cache_dir is not None else 0,
        "teacher_cache_load_workers": args.teacher_cache_load_workers if teacher_cache_dir is not None else 0,
    })
    start = time.time()

    pbar = tqdm(range(args.max_steps), dynamic_ncols=True)
    for step in pbar:
        student.train()
        set_optimizer_lr(optimizer, base_lrs, lr_scale(step, args.warmup_steps, args.max_steps, args.min_lr_frac))
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        before = None
        if step % args.log_every == 0:
            before = [p.detach().float().clone() for p in student.parameters() if p.requires_grad]
        for _ in range(args.grad_accum_steps):
            if teacher_batcher is None:
                x, y = batcher.next()
                t_logits = teacher_forward(teacher, x, args.precision)
            else:
                x, y, t_logits = teacher_batcher.next()
            with autocast_context(device, args.precision):
                s_logits = student(x)
                loss, train_parts = distill_loss(s_logits, t_logits, y, args.loss_type, beta)
                loss = loss / args.grad_accum_steps
            loss.backward()
            loss_accum += loss.detach().float().item()
        gnorm = grad_norm(student.parameters())
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0:
            params_now = [p for p in student.parameters() if p.requires_grad]
            row = {
                "event": "train",
                "step": step,
                "loss": loss_accum,
                "grad_norm": gnorm,
                "update_norm": param_update_norm(before, params_now) if before is not None else None,
                "lr": optimizer.param_groups[0]["lr"],
                "tokens_seen": (step + 1) * args.batch_size * args.seq_len * args.grad_accum_steps,
                "seconds": time.time() - start,
                "loss_type": args.loss_type,
                "beta": "inf" if math.isinf(beta) else beta,
                "optimizer": args.optimizer,
            }
            row.update({k: float(v) for k, v in train_parts.items()})
            logger.write(row)
            pbar.set_postfix(loss=f"{loss_accum:.3f}", g=f"{gnorm:.2f}")

        if step % args.eval_every == 0 or step == args.max_steps - 1:
            student.eval()
            with torch.no_grad():
                t_eval = probe_teacher_logits
                with autocast_context(device, args.precision):
                    s_eval = student(probe_x)
                row = {
                    "event": "eval",
                    "step": step,
                    "loss_type": args.loss_type,
                    "beta": "inf" if math.isinf(beta) else beta,
                    "optimizer": args.optimizer,
                    "seconds": time.time() - start,
                }
                row.update(scalar_metrics(s_eval, t_eval, probe_y, beta))
            if args.kernel_every and (step % args.kernel_every == 0 or step == args.max_steps - 1):
                k = empirical_kernel(student, probe_x, probe_teacher_logits, kernel_params)
                row.update(kernel_overlaps(k, k0, kt))
            logger.write(row)

        if args.save_every and step > 0 and step % args.save_every == 0:
            raw_student = student._orig_mod if hasattr(student, "_orig_mod") else student
            torch.save({"model": raw_student.state_dict(), "args": vars(args), "step": step}, run_dir / f"ckpt_{step}.pt")

    raw_student = student._orig_mod if hasattr(student, "_orig_mod") else student
    final_payload = {"model": raw_student.state_dict(), "args": vars(args), "step": args.max_steps}
    final_path = run_dir / "final.pt"
    torch.save(final_payload, final_path)
    if args.weights_out_dir:
        weights_dir = Path(args.weights_out_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)
        scratch_path = weights_dir / f"{run_name}_student_final.pt"
        torch.save(final_payload, scratch_path)
        logger.write({"event": "weights_saved", "step": args.max_steps, "path": str(scratch_path)})
    logger.write({"event": "done", "step": args.max_steps, "seconds": time.time() - start})
    logger.close()
    if teacher_batcher is not None:
        teacher_batcher.close()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
