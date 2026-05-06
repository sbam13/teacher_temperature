from itertools import islice

import torch
from datasets import load_dataset


def load_text_stream(dataset_name, dataset_config, split, streaming=True, seed=0):
    kwargs = dict(split=split, streaming=streaming)
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, **kwargs)
    else:
        ds = load_dataset(dataset_name, **kwargs)
    if streaming:
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
    return ds


def _get_text(example):
    for key in ("text", "content", "article", "document"):
        if key in example and example[key]:
            return example[key]
    values = [v for v in example.values() if isinstance(v, str)]
    return values[0] if values else ""


class OnlineTokenBatcher:
    def __init__(self, dataset, tokenizer, batch_size, seq_len, device, seed=0):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.iter = iter(dataset)
        self.buffer = []
        self.seed = seed

    def _refill(self, min_tokens):
        while len(self.buffer) < min_tokens:
            try:
                ex = next(self.iter)
            except StopIteration:
                self.iter = iter(self.dataset)
                ex = next(self.iter)
            text = _get_text(ex)
            if not text:
                continue
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if self.tokenizer.eos_token_id is not None:
                ids.append(self.tokenizer.eos_token_id)
            self.buffer.extend(ids)

    def next(self):
        need = self.batch_size * (self.seq_len + 1)
        self._refill(need)
        chunk = self.buffer[:need]
        self.buffer = self.buffer[need:]
        x = torch.tensor(chunk, dtype=torch.long).view(self.batch_size, self.seq_len + 1)
        return x[:, :-1].to(self.device, non_blocking=True), x[:, 1:].to(self.device, non_blocking=True)


def make_probe_batch(dataset, tokenizer, batch_size, seq_len, device):
    rows = list(islice(iter(dataset), max(128, batch_size * 4)))
    ids = []
    for ex in rows:
        text = _get_text(ex)
        if not text:
            continue
        toks = tokenizer(text, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            toks.append(tokenizer.eos_token_id)
        ids.extend(toks)
        if len(ids) >= batch_size * (seq_len + 1):
            break
    if len(ids) < batch_size * (seq_len + 1):
        raise RuntimeError("Could not build a probe batch from the dataset.")
    x = torch.tensor(ids[: batch_size * (seq_len + 1)], dtype=torch.long).view(batch_size, seq_len + 1)
    return x[:, :-1].to(device), x[:, 1:].to(device)

