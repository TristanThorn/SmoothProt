#!/usr/bin/env python
"""
Gibbs-with-Gradients (GWG) sampler for GFP fitness optimisation.

Assumptions / design choices
----------------------------
1.  State-space = the 1000 sequences in `processed_gfp_1000_smoothed_gp.csv`.
    This keeps GP evaluations fast and avoids unseen neighbours.
2.  Energy  E(x) = GP_mean(x)            # we want to *maximise* fluorescence
3.  At each step we
        • enumerate Hamming-1 neighbours  N(x)
        • compute ΔE = E(nbr) - E(x)
        • set proposal weights  w = exp(-ΔE / T)
        • draw neighbour ∝ w   (Gibbs)
        • accept the move unconditionally           (detailed balance holds)
4.  The chain therefore targets  π(x) ∝ exp(-E(x)/T)  =  exp(GP_mean/T).

CLI
---
$ python gwg_sampler.py \
        --gp_csv      ../output/processed_gfp_1000_smoothed_gp.csv \
        --gp_model    ../output/gp_model.pkl          # optional
        --steps       500 \
        --T           0.2 \
        --out_dir     ../output
"""
from __future__ import annotations
import argparse, ast, json, time
from pathlib import Path
import numpy as np, pandas as pd, joblib
from collections import defaultdict
# ─────────────────────────────────────────────────────────────
def parse_mutvec(v):
    return np.array(ast.literal_eval(v), dtype=np.uint8) if isinstance(v, str) else np.asarray(v, dtype=np.uint8)

def load_dataset(csv_path: Path):
    df  = pd.read_csv(csv_path)
    X   = np.vstack(df["mutation_vector"].apply(parse_mutvec).values)       # (n, L)  uint8
    mu  = df["gp_mean"].values.astype(np.float32)                           # GP posterior mean
    return X, mu, df

# ─────────────────────────────────────────────────────────────
def build_adj_list(X: np.ndarray) -> list[list[int]]:
    """Adjacency list of Hamming-1 neighbours (undirected)."""
    n, L = X.shape
    adj  = [[] for _ in range(n)]
    for i in range(n):
        diff = np.abs(X[i] - X).sum(axis=1)
        nbrs = np.where(diff == 1)[0]
        adj[i].extend(nbrs.tolist())
    return adj

# ─────────────────────────────────────────────────────────────
def gwg_chain(X, mu, adj, steps=1000, T=0.3, start_idx=None, rng=None):
    """Run GWG; return list of visited indices."""
    rng = np.random.default_rng(rng)
    n   = len(mu)
    cur = start_idx if start_idx is not None else rng.integers(n)
    trace = [cur]
    for _ in range(steps):
        nbrs = adj[cur]
        if not nbrs:                       # isolated node (shouldn't happen)
            trace.append(cur); continue
        deltaE = -(mu[nbrs] - mu[cur])     # because E = -mu
        weights = np.exp(-deltaE / T)
        nxt = rng.choice(nbrs, p=weights/weights.sum())
        cur = nxt
        trace.append(cur)
    return trace

# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--gp_csv",  default="../output/processed_gfp_1000_smoothed_gp.csv",
                    help="CSV containing mutation_vector + gp_mean")
    ap.add_argument("--steps",   type=int, default=1000, help="# MCMC steps")
    ap.add_argument("--T",       type=float, default=0.3, help="temperature")
    ap.add_argument("--start",   type=int, default=None, help="index of start sequence (default random)")
    ap.add_argument("--out_dir", default="../output", help="directory to save trace")
    args = ap.parse_args()

    csv_path = Path(args.gp_csv)
    out_dir  = Path(args.out_dir); out_dir.mkdir(exist_ok=True, parents=True)

    print("[GWG] loading dataset …")
    X, mu, df = load_dataset(csv_path)

    print("[GWG] building adjacency …")
    t0 = time.perf_counter()
    adj = build_adj_list(X)
    print(f"      done in {time.perf_counter()-t0:.1f}s")

    print(f"[GWG] running chain: steps={args.steps}, T={args.T}")
    trace = gwg_chain(X, mu, adj, args.steps, args.T, args.start)

    # save trajectory as csv
    df_trace = pd.DataFrame({
        "step":        np.arange(len(trace)),
        "index":       trace,
        "gp_mean":     mu[trace],
        "mutation_cnt": np.sum(X[trace], axis=1),
        "sequence":    df.loc[trace, "amino_acid_sequence"].values,
    })
    out_csv = out_dir / f"gwg_trace_T{args.T}.csv"
    df_trace.to_csv(out_csv, index=False)
    print("[GWG] wrote", out_csv)

    # quick summary
    best_idx = int(df_trace["gp_mean"].idxmax())
    print("Best step  : %d" % best_idx)
    print("Best mean  : %.4f" % df_trace.loc[best_idx, "gp_mean"])
    print("Mutations  : %d"   % df_trace.loc[best_idx, "mutation_cnt"])
    print("Sequence   :", df_trace.loc[best_idx, "sequence"][:80] + "…")

if __name__ == "__main__":
    main()
