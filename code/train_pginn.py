#!/usr/bin/env python3
"""
Train an MLP to estimate (theta, s) for the single-population
coalescent-with-selection model, with an optional PGINN (theory-informed)
loss term.

Targets are [log(theta), log(1+s)]: prepare.py stores log(theta), and s
(the population-scaled selection coefficient, s = c*s_g) gets the log(1+s)
link here, which is 0 at the neutral point s=0. predict() inverts both
links. The PGINN term is the KL divergence between the
sweep-rescaled coalescent process implied by the predicted
(theta_hat, s_hat) and by the true (theta, s), evaluated at the sample's
observed (Monte-Carlo-averaged) internal-node ages -- see sweep.py for
the theta_eff(t; theta, s) rescaling and the paper (formulae.pdf) for the
full derivation.

--pginnratio 0 reduces this to a plain MSE-only baseline (still ages-aware
for a fair, identical-data-loading comparison against --pginnratio > 0).

Genealogy ages are in simtree.py's time unit (expected mutations per
site); coalescent time t (units of c generations) is age/theta, so the
sweep trajectory x(t; s) is evaluated with rate s/theta per age unit
(sweep.rate_per_age).
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random
import os

MIN_THETA = 1e-5
THETA_SUPPORT = (0.001, 0.05)  # prior support used for training (reproduce.sh);
S_SUPPORT = (0.0, 1000.0)      # predict() bounds predictions to it
MAX_LAMBDA = 1e12  # k(k-1)/theta_eff reaches ~3e7 at n=176, theta=0.001; KL uses only the ratio
EPS = 1e-8
X0 = 0.999          # must match sweep.X0
MAX_EXPONENT = 50.0  # clamp s*t before exp() to avoid overflow


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLPRegressor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def param_labels(out_dim):
    if out_dim == 2:
        return [r'$\Theta$', r'$s$']
    return [f'param {i}' for i in range(out_dim)]


def x_trajectory_torch(t, s, x0=X0):
    """Backward-time logistic allele-frequency trajectory, torch version of sweep.x_trajectory."""
    exponent = torch.clamp(s * t, max=MAX_EXPONENT)
    return x0 / (x0 + (1.0 - x0) * torch.exp(exponent))


def coalescent_kl_selection(pred_std, true_std, y_mean, y_std, ages, eps=EPS):
    """
    KL divergence between the sweep-rescaled coalescent process implied by
    the predicted vs. true (theta, s), evaluated at the observed internal-
    node ages (columns sorted descending: column 0 = deepest/root event,
    k=2 lineages; column K-1 = most recent event, k=K+1 lineages).

    pred_std, true_std: (B, 2) standardized [log_theta, log(1+s)]
    ages:               (B, K) raw (real-time-scale) mean internal-node ages
    Returns: (B,) per-example KL, averaged over the K intervals.
    """
    log_theta_hat = pred_std[:, 0] * y_std[0] + y_mean[0]
    log_theta_true = true_std[:, 0] * y_std[0] + y_mean[0]
    s_hat = torch.clamp(torch.expm1(pred_std[:, 1] * y_std[1] + y_mean[1]), min=0.0)
    s_true = torch.clamp(torch.expm1(true_std[:, 1] * y_std[1] + y_mean[1]), min=0.0)

    theta_hat = torch.exp(log_theta_hat) + MIN_THETA
    theta_true = torch.exp(log_theta_true) + MIN_THETA

    K = ages.shape[1]
    k = torch.arange(2, K + 2, device=ages.device, dtype=ages.dtype)  # [2,3,...,K+1]
    k_km1 = k * (k - 1)  # (K,)

    x_hat = x_trajectory_torch(ages, (s_hat / theta_hat).unsqueeze(1))     # (B,K)
    x_true = x_trajectory_torch(ages, (s_true / theta_true).unsqueeze(1))  # (B,K)

    theta_eff_hat = theta_hat.unsqueeze(1) * x_hat + eps
    theta_eff_true = theta_true.unsqueeze(1) * x_true + eps

    lam_hat = (k_km1.unsqueeze(0) / theta_eff_hat).clamp(max=MAX_LAMBDA)
    lam = (k_km1.unsqueeze(0) / theta_eff_true).clamp(max=MAX_LAMBDA)

    kl_per_interval = torch.log(lam_hat / lam + eps) + lam / lam_hat - 1.0
    return kl_per_interval.mean(dim=1)


def train_epoch(model, loader, optimizer, device, pginnratio, y_mean, y_std):
    model.train()
    total_loss = mtotal = ktotal = 0.0
    mse_criterion = nn.MSELoss()
    for xb, yb, ages in loader:
        xb, yb, ages = xb.to(device), yb.to(device), ages.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        mse = mse_criterion(pred, yb)
        kl_loss = coalescent_kl_selection(pred, yb, y_mean, y_std, ages).mean()
        loss = (1.0 - pginnratio) * mse + pginnratio * kl_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        mtotal += (1.0 - pginnratio) * mse.item() * xb.size(0)
        ktotal += pginnratio * kl_loss.item() * xb.size(0)
    n = len(loader.dataset)
    return total_loss / n, mtotal / n, ktotal / n


@torch.no_grad()
def eval_epoch(model, loader, device, pginnratio, y_mean, y_std):
    model.eval()
    total_loss = mtotal = ktotal = 0.0
    mse_criterion = nn.MSELoss()
    for xb, yb, ages in loader:
        xb, yb, ages = xb.to(device), yb.to(device), ages.to(device)
        pred = model(xb)
        mse = mse_criterion(pred, yb)
        kl_loss = coalescent_kl_selection(pred, yb, y_mean, y_std, ages).mean()
        loss = (1.0 - pginnratio) * mse + pginnratio * kl_loss
        total_loss += loss.item() * xb.size(0)
        mtotal += (1.0 - pginnratio) * mse.item() * xb.size(0)
        ktotal += pginnratio * kl_loss.item() * xb.size(0)
    n = len(loader.dataset)
    return total_loss / n, mtotal / n, ktotal / n


def fit_model(train_loader, val_loader, in_dim, out_dim, y_mean, y_std,
              hidden_dims=(256, 256), lr=1e-3, weight_decay=1e-4,
              epochs=100, pginnratio=0.2, device="cpu"):
    model = MLPRegressor(in_dim, out_dim, hidden_dims).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    y_mean_t = torch.from_numpy(y_mean).float().to(device)
    y_std_t = torch.from_numpy(y_std).float().to(device)

    best_val, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_mse, tr_kl = train_epoch(model, train_loader, optimizer, device,
                                              pginnratio, y_mean_t, y_std_t)
        val_loss, val_mse, val_kl = eval_epoch(model, val_loader, device,
                                                pginnratio, y_mean_t, y_std_t)
        print(f"epoch {epoch:03d}  train {tr_loss:.4f} (mse {tr_mse:.4f} kl {tr_kl:.4f})  "
              f"val {val_loss:.4f} (mse {val_mse:.4f} kl {val_kl:.4f})")
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, best_epoch


def main():
    parser = argparse.ArgumentParser(description="Train the theta/s MLP with an optional PGINN loss term.")
    parser.add_argument("--x-train", required=True)
    parser.add_argument("--y-train", required=True)
    parser.add_argument("--ages-train", required=True)
    parser.add_argument("--x-val", required=True)
    parser.add_argument("--y-val", required=True)
    parser.add_argument("--ages-val", required=True)
    parser.add_argument("--out-model", default="model.pt")
    parser.add_argument("--out-scaler", default="scaler.npz")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--pginnratio", type=float, default=0.2,
                         help="0 = MSE-only baseline; >0 mixes in the coalescent KL term")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2,
                        help="torch CPU threads; results are bit-reproducible only for the same seed AND thread count")
    parser.add_argument("--seed", type=int, default=67, help="random seed for init/shuffling (default 67, matches the original run)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print("Using device:", device)

    X_train, y_train, ages_train = np.load(args.x_train), np.load(args.y_train), np.load(args.ages_train)
    X_val, y_val, ages_val = np.load(args.x_val), np.load(args.y_val), np.load(args.ages_val)

    y_train, y_val = y_train.copy(), y_val.copy()
    y_train[:, 1] = np.log1p(y_train[:, 1])
    y_val[:, 1] = np.log1p(y_val[:, 1])

    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    X_train_std = (X_train - x_mean) / x_std
    X_val_std = (X_val - x_mean) / x_std

    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0)
    y_std[y_std == 0.0] = 1.0
    y_train_std = (y_train - y_mean) / y_std
    y_val_std = (y_val - y_mean) / y_std

    train_ds = TensorDataset(torch.from_numpy(X_train_std).float(),
                              torch.from_numpy(y_train_std).float(),
                              torch.from_numpy(ages_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val_std).float(),
                            torch.from_numpy(y_val_std).float(),
                            torch.from_numpy(ages_val).float())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    in_dim, out_dim = X_train.shape[1], y_train.shape[1]
    print("Dimensions:", in_dim, out_dim, "pginnratio:", args.pginnratio)

    model, best_val, best_epoch = fit_model(
        train_loader, val_loader, in_dim, out_dim, y_mean, y_std,
        hidden_dims=tuple(args.hidden), lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, pginnratio=args.pginnratio, device=device)
    print("Best validation loss:", best_val, "at epoch", best_epoch)

    torch.save(model.state_dict(), args.out_model)
    np.savez(args.out_scaler, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
              s_transform="log1p")
    print("Saved model to", args.out_model, "and scaler to", args.out_scaler)


def predict(model_path, scaler_path, X):
    """Natural-scale [theta_hat, s_hat] for feature rows X (array or .npy path),
    bounded to the prior support: the log links would otherwise turn a network
    output slightly outside the training range into an arbitrarily large value."""
    X = np.load(X) if isinstance(X, (str, os.PathLike)) else np.asarray(X)
    scaler = np.load(scaler_path, allow_pickle=True)
    assert str(scaler["s_transform"]) == "log1p", f"{scaler_path}: unexpected s link {scaler['s_transform']}"
    x_std = (X - scaler["x_mean"]) / scaler["x_std"]
    model = MLPRegressor(X.shape[1], scaler["y_mean"].shape[0])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        z = model(torch.from_numpy(x_std).float()).numpy() * scaler["y_std"] + scaler["y_mean"]
    theta = np.clip(np.exp(z[:, 0]) + MIN_THETA, *THETA_SUPPORT)
    s = np.clip(np.expm1(z[:, 1]), *S_SUPPORT)
    return np.column_stack([theta, s])


if __name__ == "__main__":
    main()
