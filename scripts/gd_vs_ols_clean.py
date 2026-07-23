"""Clean comparison: normalized GD-linear vs OLS vs GRU, all on same data."""
import sys, numpy as np, pandas as pd, torch
import torch.nn as nn
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from model.config.run_config import PHASE_A_FEATURES
from model.config.splits import get_split_mask
from model.inference.engine import _ffill_np
from model.body.gru_encoder import GRUEncoder


class LinearWindow(nn.Module):
    def __init__(self, n_features, window_length, horizon):
        super().__init__()
        self.n_features = n_features
        self.linear = nn.Linear(n_features * window_length, horizon)

    def forward(self, x):
        B = x.shape[0]
        return self.linear(x.reshape(B, -1))


def get_windows(df, stride):
    feat = _ffill_np(df[PHASE_A_FEATURES].values.astype(np.float32))
    feat = np.nan_to_num(feat, nan=0.0)
    nr = df["norm_return"].values.astype(np.float64)
    H, W = 12, 60
    wv = np.lib.stride_tricks.sliding_window_view(feat, window_shape=W, axis=0)
    bi = np.arange(W - 1, len(feat))[::stride]
    X, Y = [], []
    for idx in bi:
        if idx + 1 + H > len(nr):
            continue
        X.append(wv[idx - (W - 1)].T.copy())
        Y.append(nr[idx + 1 : idx + 1 + H])
    return np.array(X), np.array(Y)


def eval_mse(model, X, Y, is_gru=False):
    model.eval()
    mses, corrs = [], []
    for i in range(len(X)):
        t = torch.from_numpy(X[i]).unsqueeze(0)
        with torch.no_grad():
            out = model(t)
        m = out[0][0].numpy() if is_gru else out[0].numpy()
        tgt = Y[i]
        valid = ~np.isnan(tgt)
        if not valid.any():
            continue
        mses.append(((tgt[valid] - m[valid]) ** 2).mean())
        if m[valid].std() > 1e-8 and tgt[valid].std() > 1e-8:
            corrs.append(np.corrcoef(m[valid], tgt[valid])[0, 1])
    return np.array(mses).mean(), np.mean(corrs) if corrs else 0.0


def main():
    df = pd.read_parquet(_root / "data/processed/v1/SOLUSDT/1m").sort_values("timestamp")
    train_df = df[get_split_mask(df, "train")].reset_index(drop=True)
    val_df = df[get_split_mask(df, "val")].reset_index(drop=True)

    train_X, train_Y = get_windows(train_df, stride=60)
    val_X, val_Y = get_windows(val_df, stride=60)
    val_var = val_Y.var()
    print(f"Train: {len(train_X)}, Val: {len(val_X)}, Val var: {val_var:.6f}")
    print()

    # 1. OLS (closed-form)
    print("=== OLS (closed-form, stride=60 training data) ===")
    X_flat = train_X.reshape(len(train_X), -1)
    B, _, _, _ = np.linalg.lstsq(X_flat, train_Y, rcond=None)
    val_pred_ols = val_X.reshape(len(val_X), -1) @ B
    ols_mse = ((val_Y - val_pred_ols) ** 2).mean()
    ols_corr = np.mean([np.corrcoef(val_Y[i][~np.isnan(val_Y[i])], val_pred_ols[i][~np.isnan(val_Y[i])])[0,1] for i in range(len(val_Y)) if np.any(~np.isnan(val_Y[i]))])
    print(f"  val_mse={ols_mse:.6f}  val_corr={ols_corr:.4f}  vs baseline {(ols_mse-val_var)/val_var*100:+.1f}%")
    print()

    # 2. Normalized GD-linear (lr=1e-3, 30 epochs)
    print("=== Linear GD (normalized, lr=1e-3, 30 epochs) ===")
    # Compute normalization from training data
    X_flat_train = train_X.reshape(len(train_X), -1)
    mean = X_flat_train.mean(axis=0)
    std = X_flat_train.std(axis=0).clip(min=1e-6)

    lin = LinearWindow(10, 60, 12)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
    print(f"{'ep':>3} {'train_mse':>10} {'val_mse':>10} {'val_corr':>10}")
    for ep in range(30):
        lin.train()
        perm = np.random.permutation(len(train_X))
        for i in range(0, len(train_X), 256):
            idx = perm[i:i+256]
            xb = torch.from_numpy((train_X[idx].reshape(len(idx), -1) - mean) / std).float()
            yb = torch.from_numpy(train_Y[idx]).float()
            opt.zero_grad()
            loss = ((lin(xb) - yb)**2).mean()
            loss.backward()
            opt.step()
        if (ep+1) % 5 == 0 or ep == 0:
            lin.eval()
            xv = torch.from_numpy((val_X.reshape(len(val_X), -1) - mean) / std).float()
            with torch.no_grad():
                pred = lin(xv).numpy()
            tmse = ((train_Y[:500] - lin(torch.from_numpy((train_X[:500].reshape(500, -1) - mean) / std).float()).detach().numpy())**2).mean()
            vmse = ((val_Y - pred)**2).mean()
            vcorr = np.mean([np.corrcoef(val_Y[i][~np.isnan(val_Y[i])], pred[i][~np.isnan(val_Y[i])])[0,1] for i in range(len(val_Y)) if np.any(~np.isnan(val_Y[i]))])
            print(f"{ep+1:3d} {tmse:10.6f} {vmse:10.6f} {vcorr:10.6f}")
    print()

    # 3. Normalized GD-linear (lr=1e-2)
    print("=== Linear GD (normalized, lr=1e-2, 30 epochs) ===")
    lin2 = LinearWindow(10, 60, 12)
    opt2 = torch.optim.Adam(lin2.parameters(), lr=1e-2)
    print(f"{'ep':>3} {'train_mse':>10} {'val_mse':>10} {'val_corr':>10}")
    for ep in range(30):
        lin2.train()
        perm = np.random.permutation(len(train_X))
        for i in range(0, len(train_X), 256):
            idx = perm[i:i+256]
            xb = torch.from_numpy((train_X[idx].reshape(len(idx), -1) - mean) / std).float()
            yb = torch.from_numpy(train_Y[idx]).float()
            opt2.zero_grad()
            loss = ((lin2(xb) - yb)**2).mean()
            loss.backward()
            opt2.step()
        if (ep+1) % 5 == 0 or ep == 0:
            lin2.eval()
            xv = torch.from_numpy((val_X.reshape(len(val_X), -1) - mean) / std).float()
            with torch.no_grad():
                pred = lin2(xv).numpy()
            tmse = ((train_Y[:500] - lin2(torch.from_numpy((train_X[:500].reshape(500, -1) - mean) / std).float()).detach().numpy())**2).mean()
            vmse = ((val_Y - pred)**2).mean()
            vcorr = np.mean([np.corrcoef(val_Y[i][~np.isnan(val_Y[i])], pred[i][~np.isnan(val_Y[i])])[0,1] for i in range(len(val_Y)) if np.any(~np.isnan(val_Y[i]))])
            print(f"{ep+1:3d} {tmse:10.6f} {vmse:10.6f} {vcorr:10.6f}")
    print()

    # 4. GRU (normalized via its own feature_center/scale)
    print("=== GRU (lr=1e-3, 30 epochs) ===")
    gru = GRUEncoder(n_features=10, hidden_size=32, num_layers=1, dropout=0.0, horizon=12)
    opt_g = torch.optim.Adam(gru.parameters(), lr=1e-3)
    print(f"{'ep':>3} {'train_mse':>10} {'val_mse':>10} {'val_corr':>10}")
    for ep in range(30):
        gru.train()
        perm = np.random.permutation(len(train_X))
        for i in range(0, len(train_X), 256):
            idx = perm[i:i+256]
            xb = torch.from_numpy(train_X[idx]).float()
            yb = torch.from_numpy(train_Y[idx]).float()
            opt_g.zero_grad()
            mean_out, _ = gru(xb)
            loss = ((mean_out - yb)**2).mean()
            loss.backward()
            opt_g.step()
        if (ep+1) % 5 == 0 or ep == 0:
            vmse, vcorr = eval_mse(gru, val_X, val_Y, is_gru=True)
            tmse, _ = eval_mse(gru, train_X[:500], train_Y[:500], is_gru=True)
            print(f"{ep+1:3d} {tmse:10.6f} {vmse:10.6f} {vcorr:10.6f}")
    print()

    # Summary
    print("=" * 65)
    print(f"{'Model':<35} {'val_mse':>10} {'val_corr':>10}")
    print("-" * 65)
    print(f"{'Baseline (val var)':<35} {val_var:10.6f} {'---':>10}")
    print(f"{'OLS (closed-form)':<35} {ols_mse:10.6f} {ols_corr:10.4f}")
    print(f"{'Linear GD lr=1e-3 (ep30)':<35} {vmse:10.6f} {vcorr:10.4f}")
    print(f"{'GRU lr=1e-3 (ep30)':<35} {vmse:10.6f} {vcorr:10.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
