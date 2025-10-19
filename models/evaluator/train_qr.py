"""Train a Quantile Regression evaluator for T_clear.

Expected input dataset (Parquet or NPZ):
  - state: (C,H,W) float32 tensor or raw 16×8 state planes to be upsampled
  - samples: (S,) float32 Monte-Carlo samples of T_clear (frames)
  - censored: bool (optional)

This script is a skeleton; wire your actual dataset loader.
"""

import argparse, os, math, json
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import envs.specs.ram_to_state as ram_specs
from models.evaluator.qr_modules import QREvaluator, quantile_huber_loss


class DummyDataset(Dataset):
    def __init__(self, n=1024, C: int | None = None, H=128, W=128, S=32):
        if C is None:
            C = ram_specs.STATE_CHANNELS
        self.X = np.random.randn(n, C, H, W).astype(np.float32)
        self.Y = np.abs(np.random.randn(n, S).astype(np.float32) * 1000.0)  # fake T_clear samples

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--k', type=int, default=101)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--out', type=str, default='evaluator.ts')
    args = ap.parse_args()

    taus = torch.linspace(1 / (args.k + 1), args.k / (args.k + 1), args.k)  # fixed quantiles
    model = QREvaluator(in_channels=ram_specs.STATE_CHANNELS, k_quantiles=args.k)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    ds = DummyDataset()
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    model.train()
    for ep in range(args.epochs):
        total = 0.0
        for x, y in dl:
            pred = model(x)
            loss = quantile_huber_loss(pred, y, taus)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch {ep + 1}: loss={total / len(dl):.4f}")
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(args.out)
    print(f"saved TorchScript evaluator to {args.out}")


if __name__ == '__main__':
    main()
