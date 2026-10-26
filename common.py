import math
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F


class Batches:
    # Случайные окна из плоского uint16 файла токенов (как в nanoGPT)
    def __init__(self, path: Path | str, seq_len: int, batch_size: int, device: str):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device

    def random(self) -> torch.Tensor:
        ix = np.random.randint(0, len(self.data) - self.seq_len - 1, self.batch_size)
        x = np.stack([self.data[i:i + self.seq_len + 1] for i in ix]).astype(np.int64)

        return torch.from_numpy(x).to(self.device)

    def sequential(self) -> Iterator[torch.Tensor]:
        # Каждый токен файла ровно один раз, чтобы лосс на валидации был детерминированным
        n = (len(self.data) - 1) // self.seq_len

        for start in range(0, n, self.batch_size):
            rows = [
                self.data[i * self.seq_len:(i + 1) * self.seq_len + 1]
                for i in range(start, min(start + self.batch_size, n))
            ]

            yield torch.from_numpy(np.stack(rows).astype(np.int64)).to(self.device)


def lm_loss(model, batch: torch.Tensor, reduction: str = "mean") -> tuple[torch.Tensor, torch.Tensor | None]:
    out = model(input_ids=batch[:, :-1])
    ce = F.cross_entropy(out.logits.float().flatten(0, 1), batch[:, 1:].flatten(), reduction=reduction)

    return ce, out.get("aux_loss")


@torch.no_grad()
def evaluate(model, val: Batches, val_chars: int, autocast) -> tuple[float, float]:
    model.eval()

    nll, n = 0.0, 0

    for batch in val.sequential():
        with autocast:
            nll += lm_loss(model, batch, reduction="sum")[0].item()

        n += batch[:, 1:].numel()

    model.train()

    # Bits per character не зависит от токенизатора, поэтому запуски с разными словарями сравнимы
    chars = val_chars * n / (len(val.data) - 1)

    return nll / n, nll / math.log(2) / chars
