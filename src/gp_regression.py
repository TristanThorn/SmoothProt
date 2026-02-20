#!/usr/bin/env python
"""Gaussian-Process Regression on smoothed GFP fitness data with plotting utilities.

This script trains a Gaussian-Process Regressor (GPR) on the smoothed
fluorescence values produced by the graph-smoothing step.  It also offers
optional diagnostic plots—including an **UMAP** projection—for a richer view
of the mutation-fitness landscape.
"""

from __future__ import annotations

import argparse
import ast
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV

# ─────────────────────────────────────────────────────────────
# 1. Utilities
# ─────────────────────────────────────────────────────────────

def parse_mut_vec(v: str | list[int]) -> np.ndarray:
    """Convert the `mutation_vector` column into a NumPy array."""
    if isinstance(v, str):
        return np.array(ast.literal_eval(v), dtype=np.float32)
    return np.asarray(v, dtype=np.float32)


def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    X = np.vstack(df["mutation_vector"].apply(parse_mut_vec).values)
    y = df["smooth_fluorescence"].values.astype(np.float32)
    return X, y, df


# ─────────────────────────────────────────────────────────────
# 2. Model fitting
# ─────────────────────────────────────────────────────────────

def fit_gp(X: np.ndarray, y: np.ndarray, cv_splits: int = 5) -> GaussianProcessRegressor:
    """Fit a GPR with a small CV grid-search on length-scale and noise."""
    base_kernel = ConstantKernel(1.0, (1e-2, 10.0)) * RBF(
        length_scale=1.0, length_scale_bounds=(1e-2, 10.0)
    )
    kernel = base_kernel + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-4, 1.0))

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
    )

    param_grid = {
        "alpha": [0.0],  # white‑kernel handles noise
    }

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    search = GridSearchCV(
        gpr,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        verbose=0,
    )
    search.fit(X, y)
    return search.best_estimator_


# ─────────────────────────────────────────────────────────────
# 3. Plotting helpers
# ─────────────────────────────────────────────────────────────

def plot_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    X: np.ndarray,
    out_dir: Path,
    do_umap: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Scatter with error bars ---
    plt.figure(figsize=(5, 5))
    plt.errorbar(
        y_true,
        y_pred,
        yerr=y_std,
        fmt="o",
        ms=3,
        alpha=0.5,
        ecolor="lightgray",
        linewidth=0,
    )
    min_v, max_v = y_true.min(), y_true.max()
    plt.plot([min_v, max_v], [min_v, max_v], "k--", lw=1)
    plt.xlabel("True smooth fluorescence")
    plt.ylabel("GP prediction")
    plt.title("True vs Predicted")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_true_vs_pred.png", dpi=300)
    plt.close()

    # --- Residuals histogram ---
    residuals = y_true - y_pred
    plt.figure(figsize=(5, 4))
    plt.hist(residuals, bins=30, edgecolor="k")
    plt.xlabel("Residual (true - pred)")
    plt.ylabel("Count")
    plt.title("Residuals histogram")
    plt.tight_layout()
    plt.savefig(out_dir / "residuals_hist.png", dpi=300)
    plt.close()

    # --- PCA coloured by fitness ---
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=y_true,
        cmap="viridis",
        s=10,
        alpha=0.8,
    )
    plt.colorbar(sc, label="Smooth fluorescence")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Mutation space PCA coloured by fitness")
    plt.tight_layout()
    plt.savefig(out_dir / "pca_fitness.png", dpi=300)
    plt.close()

    # --- Optional UMAP ---
    if do_umap:
        reducer = umap.UMAP(random_state=42, metric="hamming")
        X_umap = reducer.fit_transform(X)
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(
            X_umap[:, 0],
            X_umap[:, 1],
            c=y_true,
            cmap="viridis",
            s=10,
            alpha=0.8,
        )
        plt.colorbar(sc, label="Smooth fluorescence")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.title("Mutation space UMAP coloured by fitness")
        plt.tight_layout()
        plt.savefig(out_dir / "umap_fitness.png", dpi=300)
        plt.close()


# ─────────────────────────────────────────────────────────────
# 4. CLI entry‑point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gaussian-Process regression on smoothed GFP data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default="../output/processed_gfp_1000_smoothed.csv",
        help="Input CSV with smoothed fluorescence",
    )
    parser.add_argument(
        "--out_csv",
        default="../output/processed_gfp_1000_smoothed_gp.csv",
        help="Output CSV with GP predictions (default <csv>_gp.csv)",
    )
    parser.add_argument(
        "--model",
        default="../output/gp_model.pkl",
        help="Path to save fitted GP model (.pkl)",
    )
    parser.add_argument(
        "--plots",
        default=True,
        action="store_true",
        help="Generate diagnostic plots",
    )
    parser.add_argument(
        "--plot_dir",
        default="../output/plots",
        help="Directory to write plots if --plots is given",
    )
    parser.add_argument(
        "--umap",
        default=True,
        action="store_true",
        help="Add UMAP visualisation (requires umap-learn)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_csv = Path(args.out_csv or csv_path.with_name(csv_path.stem + "_gp.csv"))
    model_path = Path(args.model) if args.model else None
    plot_dir = Path(args.plot_dir)

    print("[INFO] Loading data", csv_path)
    X, y, df = load_data(csv_path)

    print("[INFO] Fitting Gaussian Process …")
    t0 = time.perf_counter()
    gpr = fit_gp(X, y)
    print(f"       Done in {time.perf_counter() - t0:.2f} s")

    print("[INFO] Predicting on training data …")
    y_pred, y_std = gpr.predict(X, return_std=True)
    print("       RMSE = %.4f | R² = %.4f" % (
        mean_squared_error(y, y_pred, squared=False), r2_score(y, y_pred)
    ))

    df["gp_mean"] = y_pred
    df["gp_std"] = y_std
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("[INFO] Wrote predictions to", out_csv)

    if model_path:
        joblib.dump(gpr, model_path)
        print("[INFO] Saved model to", model_path)

    if args.plots:
        print("[INFO] Generating plots to", plot_dir)
        plot_diagnostics(y, y_pred, y_std, X, plot_dir, do_umap=args.umap)
        print("[INFO] Plots saved in", plot_dir)


if __name__ == "__main__":
    main()
