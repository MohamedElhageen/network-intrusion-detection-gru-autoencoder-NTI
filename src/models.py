"""GRU autoencoder for windows of flows. Anomaly score = reconstruction error."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import BATCH, EPOCHS, HIDDEN, LATENT, LR, PATIENCE, SEED


class GRUAutoencoder(nn.Module):
    """Encoder GRU -> small latent vector -> decoder GRU. Trained on benign windows only."""

    def __init__(self, n_features: int, window: int, hidden: int = HIDDEN, latent: int = LATENT):
        super().__init__()
        self.window = window
        self.encoder = nn.GRU(n_features, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):                                   # x: (B, window, F)
        _, h = self.encoder(x)                              # h: (1, B, hidden)
        z = self.to_latent(h[-1])                           # (B, latent)
        d = self.from_latent(z).unsqueeze(1).repeat(1, self.window, 1)
        o, _ = self.decoder(d)                              # (B, window, hidden)
        return self.out(o)                                  # (B, window, F)


@torch.no_grad()
def reconstruction_errors(model: nn.Module, X: np.ndarray, batch: int = 4096, device: str = "cpu") -> np.ndarray:
    """Mean squared reconstruction error per window (higher = more anomalous)."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        out.append(((model(xb) - xb) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def train_autoencoder(X_train, X_val, hidden=HIDDEN, latent=LATENT, epochs=EPOCHS, batch=BATCH,
                      lr=LR, patience=PATIENCE, seed=SEED, device="cpu", verbose=True):
    """Train with early stopping on benign validation loss. Returns (best_model, history_df)."""
    torch.manual_seed(seed)
    n_features, window = X_train.shape[2], X_train.shape[1]
    model = GRUAutoencoder(n_features, window, hidden, latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=batch, shuffle=True)

    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(xb)
        tr_loss = total / len(X_train)
        va_loss = float(reconstruction_errors(model, X_val, device=device).mean())
        hist.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})
        if verbose:
            print(f"epoch {ep:02d}  train {tr_loss:.4f}  val {va_loss:.4f}")
        if va_loss < best - 1e-5:
            best, bad = va_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)


def save_model(model: GRUAutoencoder, path, hidden=HIDDEN, latent=LATENT) -> None:
    torch.save({"state_dict": model.state_dict(), "n_features": model.encoder.input_size,
                "window": model.window, "hidden": hidden, "latent": latent}, path)


def load_model(path, device: str = "cpu") -> GRUAutoencoder:
    ckpt = torch.load(path, map_location=device)
    model = GRUAutoencoder(ckpt["n_features"], ckpt["window"], ckpt["hidden"], ckpt["latent"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
