import hashlib
import json
import os
import queue
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


CACHE_VERSION = 1


def _slug(value, max_len=48):
    text = str(value or "none").lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text).strip("-")
    return text[:max_len] or "none"


def _atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _torch_load_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def teacher_cache_config(args, tokenizer):
    return {
        "cache_version": CACHE_VERSION,
        "teacher_model": args.teacher_model,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "streaming": args.streaming,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "probe_batch_size": args.probe_batch_size,
        "seq_len": args.seq_len,
        "precision": args.precision,
        "teacher_cache_dtype": args.teacher_cache_dtype,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "tokenizer_vocab_size": len(tokenizer),
    }


def teacher_cache_subdir(root, args, tokenizer):
    if not root:
        return None
    config = teacher_cache_config(args, tokenizer)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    name = (
        f"{_slug(args.teacher_model)}_{_slug(args.dataset_name)}_"
        f"seed{args.seed}_bs{args.batch_size}_seq{args.seq_len}_{args.precision}_{digest}"
    )
    path = Path(root) / name
    path.mkdir(parents=True, exist_ok=True)
    manifest = path / "manifest.json"
    if not manifest.exists():
        _atomic_write_text(manifest, json.dumps(config, indent=2, sort_keys=True) + "\n")
    return path


def _cache_dtype(dtype_name, tensor):
    if dtype_name == "auto":
        return tensor.detach().cpu()
    dtypes = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    return tensor.detach().to(device="cpu", dtype=dtypes[dtype_name])


def _to_cpu_payload(index, x, y, teacher_logits, cache_dtype):
    return {
        "cache_version": CACHE_VERSION,
        "index": index,
        "x": x.detach().cpu(),
        "y": y.detach().cpu(),
        "teacher_logits": _cache_dtype(cache_dtype, teacher_logits),
    }


def _payload_to_device(payload, device):
    return (
        payload["x"].to(device, non_blocking=True),
        payload["y"].to(device, non_blocking=True),
        payload["teacher_logits"].to(device, non_blocking=True),
    )


class AsyncTeacherLogitBatcher:
    def __init__(
        self,
        batcher,
        teacher,
        forward_fn,
        precision,
        cache_dir,
        mode,
        prefetch,
        device,
        cache_dtype="auto",
        load_workers=0,
    ):
        if mode not in {"readwrite", "readonly"}:
            raise ValueError(f"Unsupported teacher cache mode for cached batcher: {mode}")
        self.batcher = batcher
        self.teacher = teacher
        self.forward_fn = forward_fn
        self.precision = precision
        self.cache_dir = Path(cache_dir) / "train"
        self.mode = mode
        self.prefetch = max(0, int(prefetch))
        self.device = device
        self.cache_dtype = cache_dtype
        self.load_workers = max(0, int(load_workers))
        self.next_index = 0
        self.stop_event = threading.Event()
        self.worker = None
        self.queue = None
        if self.prefetch > 0:
            self.queue = queue.Queue(maxsize=self.prefetch)
            target = self._parallel_cache_worker_loop if self.load_workers > 1 else self._worker_loop
            self.worker = threading.Thread(target=target, name="teacher-logit-prefetch", daemon=True)
            self.worker.start()

    def next(self):
        if self.worker is None:
            payload = self._load_or_compute(self.next_index)
            self.next_index += 1
            return _payload_to_device(payload, self.device)
        payload = self.queue.get()
        if isinstance(payload, BaseException):
            self.close()
            raise payload
        return _payload_to_device(payload, self.device)

    def close(self):
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=2.0)

    def _worker_loop(self):
        index = 0
        try:
            while not self.stop_event.is_set():
                payload = self._load_or_compute(index)
                index += 1
                self._put(payload)
        except BaseException as exc:
            self._put(exc)

    def _put(self, payload):
        while not self.stop_event.is_set():
            try:
                self.queue.put(payload, timeout=0.1)
                return
            except queue.Full:
                continue

    def _parallel_cache_worker_loop(self):
        index = 0
        submit_index = 0
        pending = {}
        executor = ThreadPoolExecutor(max_workers=self.load_workers, thread_name_prefix="teacher-cache-load")
        try:
            while not self.stop_event.is_set():
                while len(pending) < self.prefetch and self._batch_path(submit_index).exists():
                    pending[submit_index] = executor.submit(self._load_cached, submit_index)
                    submit_index += 1

                if index in pending:
                    payload = pending.pop(index).result()
                else:
                    payload = self._load_or_compute(index)

                index += 1
                submit_index = max(submit_index, index)
                self._put(payload)
        except BaseException as exc:
            self._put(exc)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _batch_path(self, index):
        return self.cache_dir / f"batch_{index:08d}.pt"

    def _load_cached(self, index):
        return _torch_load_payload(self._batch_path(index))

    def _load_or_compute(self, index):
        path = self._batch_path(index)
        if path.exists():
            return _torch_load_payload(path)
        if self.mode == "readonly":
            raise FileNotFoundError(f"Missing cached teacher-logit batch: {path}")
        x, y = self.batcher.next()
        teacher_logits = self.forward_fn(self.teacher, x, self.precision)
        payload = _to_cpu_payload(index, x, y, teacher_logits, self.cache_dtype)
        if not path.exists():
            _atomic_torch_save(payload, path)
        return payload


def load_or_compute_probe_batch(cache_dir, mode, batch_builder, teacher, forward_fn, precision, device, cache_dtype="auto"):
    if mode not in {"readwrite", "readonly"}:
        raise ValueError(f"Unsupported teacher cache mode for cached probe batch: {mode}")
    path = Path(cache_dir) / "probe.pt"
    if path.exists():
        return _payload_to_device(_torch_load_payload(path), device)
    if mode == "readonly":
        raise FileNotFoundError(f"Missing cached probe teacher logits: {path}")
    x, y = batch_builder()
    teacher_logits = forward_fn(teacher, x, precision)
    payload = _to_cpu_payload("probe", x, y, teacher_logits, cache_dtype)
    if not path.exists():
        _atomic_torch_save(payload, path)
    return _payload_to_device(payload, device)
