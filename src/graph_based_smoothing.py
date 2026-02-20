"""Graph-based smoothing for GFP fitness landscape

This script takes the pre-processed GFP dataset and applies Tikhonov (Laplacian) regularisation to
smooth the fluorescence measurements across the mutation graph.

The graph is constructed as follows:
* Nodes are sequences (rows);
* Edges connect pairs of sequences whose **Hamming distance = 1**.
* The smoothed signal is   f̂ = (I + λ·L)^-1 = y
  where L is the un-normalised graph Laplacian.
* With λ → 0 this converges to y; with λ → ∞ it converges to the graph
  harmonic (average of neighbours).
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

################################################################################
# Utility functions
################################################################################

def load_data(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load the processed CSV and return (X, y, dataframe).

    * X  : binary mutation matrix (n * L)
    * y  : fluorescence vector (n,)
    """
    df = pd.read_csv(csv_path)
    # mutation_vector column is a string like "[0, 1, ...]"
    df["mutation_vector"] = df["mutation_vector"].apply(ast.literal_eval)

    X = np.asarray(df["mutation_vector"].tolist(), dtype=np.uint8)
    y = df["log_fluorescence"].to_numpy(dtype=float)
    return X, y, df


def build_adj_matrix(X: np.ndarray) -> csr_matrix:
    """Return the unweighted adjacency matrix (CSR) where A_ij = 1 if
    HammingDistance(x_i, x_j) == 1.
    
    For n=1000 and L=237 this naive O(n²·L) scan is fast enough.
    """
    n, L = X.shape
    rows, cols = [], []

    for i in range(n):
        # XOR gives 1 where bits differ, then sum -> Hamming dist
        diff = np.abs(X[i] - X).sum(axis=1)
        # indices of neighbours at distance 1 (exclude i itself)
        neighbours = np.where(diff == 1)[0]
        rows.extend([i] * len(neighbours))
        cols.extend(neighbours)

    data = np.ones(len(rows), dtype=np.uint8)
    A = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)
    return A


def laplacian_smoothing(y: np.ndarray, A: csr_matrix, alpha: float) -> np.ndarray:
    """Solve (I + alpha·L) f̂ = y for f̂.

    L = D - A, where D is the degree matrix.
    """
    n = A.shape[0]
    degree = np.asarray(A.sum(axis=1)).flatten()
    D = diags(degree)
    L = D - A

    I = diags(np.ones(n))
    rhs = y.copy()

    f_hat = spsolve(I + alpha * L, rhs)
    return f_hat

################################################################################
# Main routine
################################################################################

def main():
    parser = argparse.ArgumentParser(description="Graph smoothing for GFP data")
    parser.add_argument("--csv", default="../output/processed_gfp_1000.csv", help="processed GFP csv file")
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="regularisation strength (lambda)")
    parser.add_argument("--out", type=str, default=None,
                        help="output CSV path (defaults to <csv>_smoothed.csv)")

    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out or csv_path.with_name(csv_path.stem + "_smoothed.csv"))

    print("[INFO] Loading data …", flush=True)
    X, y, df = load_data(csv_path)

    print("[INFO] Building mutation graph (|V|=%d) …" % X.shape[0], flush=True)
    t0 = time.perf_counter()
    A = build_adj_matrix(X)
    print(f"       Done in {time.perf_counter()-t0:.2f} s  (|E|={A.nnz//2} undirected)")

    print(f"[INFO] Solving (I + {args.alpha}·L) f̂ = y …", flush=True)
    t0 = time.perf_counter()
    f_hat = laplacian_smoothing(y, A, args.alpha)
    print(f"       Done in {time.perf_counter()-t0:.2f} s")

    df["smooth_fluorescence"] = f_hat
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("[INFO] Wrote", out_path)


if __name__ == "__main__":
    main()
