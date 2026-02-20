#!/usr/bin/env python
"""
Compare raw vs smoothed fluorescence on a shared UMAP embedding.
"""

import ast, pathlib, argparse
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import umap
from scipy.stats import pearsonr

# ─────────────────────────────────────────────────────────────
def load(csv_path):
    df = pd.read_csv(csv_path)
    X = np.vstack(df['mutation_vector'].apply(ast.literal_eval))
    raw    = df['log_fluorescence'].values
    smooth = df['smooth_fluorescence'].values
    return X, raw, smooth

# ─────────────────────────────────────────────────────────────
def metrics(raw, smooth):
    r, _ = pearsonr(raw, smooth)
    abs_change = np.mean(np.abs(raw - smooth))
    rank_raw    = pd.Series(raw).rank() / len(raw)
    rank_smooth = pd.Series(smooth).rank() / len(smooth)
    moved_decile = (np.abs(rank_raw - rank_smooth) > 0.1).mean()
    return r, abs_change, moved_decile

# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../output/processed_gfp_1000_smoothed.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, raw, smooth = load(args.csv)

    print("[INFO] fitting UMAP …")
    reducer = umap.UMAP(metric="hamming", random_state=args.seed)
    emb = reducer.fit_transform(X)
    

    # ── plots ────────────────────────────────────────────────
    outdir = pathlib.Path("../output/plots"); outdir.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)

    sc0 = ax[0].scatter(emb[:,0], emb[:,1], c=raw, s=10, cmap="viridis")
    ax[0].set(title="Raw log-fluorescence", xlabel="UMAP-1", ylabel="UMAP-2")

    sc1 = ax[1].scatter(emb[:,0], emb[:,1], c=smooth, s=10, cmap="viridis")
    ax[1].set(title="Smoothed fluorescence", xlabel="UMAP-1", ylabel="UMAP-2")

    fig.colorbar(sc1, ax=ax.ravel().tolist(), pad=0.02, label="fluorescence")
    fig.savefig(outdir / "umap_raw_vs_smooth.png", dpi=300)
    print("[INFO] wrote", outdir / "umap_raw_vs_smooth.png")

    # ── numbers ──────────────────────────────────────────────
    r, abs_change, moved = metrics(raw, smooth)
    print(f"Pearson r(raw, smooth)        : {r:6.3f}")
    print(f"Mean |Δf| per variant         : {abs_change:6.3f}")
    print(f"Variants shifting >1 decile   : {moved*100:5.1f}%")

if __name__ == "__main__":
    main()
