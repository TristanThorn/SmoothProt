"""
Landscape Visualization and Optimization Demonstration (CSV-based GP surrogate)

This script performs:
1. t-SNE embedding of mutation vectors (Hamming distance)
2. 3D PCA and UMAP projections
3. Heatmap of pairwise Hamming distance vs. fluorescence difference
4. Hill-climbing trajectories on raw vs. smoothed landscapes
5. Bayesian optimization simulation using GP predictions from CSV
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_vectors(df):
    # mutation_vector stored as string list
    return np.vstack(df['mutation_vector'].apply(lambda s: np.array(json.loads(s)) if isinstance(s, str) else np.array(s)).values)


def plot_tsne(X, y, out_path):
    tsne = TSNE(n_components=2, metric='hamming', random_state=42)
    X2 = tsne.fit_transform(X)
    plt.figure()
    plt.scatter(X2[:,0], X2[:,1], c=y, cmap='viridis', s=5)
    plt.colorbar(label='fluorescence')
    plt.title('t-SNE (Hamming)')
    plt.savefig(out_path)


def plot_3d_embed(X, y, method, out_path):
    if method == 'pca':
        emb = PCA(n_components=3).fit_transform(X)
    else:
        emb = umap.UMAP(n_components=3, metric='hamming').fit_transform(X)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(emb[:,0], emb[:,1], emb[:,2], c=y, cmap='viridis', s=5)
    fig.colorbar(sc, label='fluorescence')
    title = '3D ' + method.upper()
    ax.set_title(title)
    plt.savefig(out_path)


def plot_hamming_heatmap(X, y_raw, y_smooth, out_path):
    n = len(y_raw)
    # sample subset for performance
    idx = np.random.choice(n, size=min(n,500), replace=False)
    X_sub = X[idx]
    raw_sub = y_raw[idx]
    sm_sub = y_smooth[idx]
    # pairwise Hamming
    dists = np.array([[np.sum(xi != xj) for xj in X_sub] for xi in X_sub])
    # pairwise delta fluorescence
    diffs = np.abs(raw_sub[:,None] - raw_sub[None,:])
    plt.figure(figsize=(6,5))
    plt.hist2d(dists.ravel(), diffs.ravel(), bins=50, cmap='magma')
    plt.colorbar(label='counts')
    plt.xlabel('Hamming distance')
    plt.ylabel('|Δ raw flu|')
    plt.title('Raw: Hamming vs Δfluorescence')
    plt.savefig(out_path.replace('.png','_raw.png'))
    # smoothed
    diffs2 = np.abs(sm_sub[:,None] - sm_sub[None,:])
    plt.figure(figsize=(6,5))
    plt.hist2d(dists.ravel(), diffs2.ravel(), bins=50, cmap='magma')
    plt.colorbar(label='counts')
    plt.xlabel('Hamming distance')
    plt.ylabel('|Δ smooth flu|')
    plt.title('Smoothed: Hamming vs Δfluorescence')
    plt.savefig(out_path.replace('.png','_smooth.png'))


def hill_climb(X, seqs, y, steps=20):
    # adjacency: one-step neighbors
    n, d = X.shape
    visited = []
    # start from WT (mutation_count=0)
    start = np.where(np.sum(X,axis=1)==0)[0][0]
    curr = start
    vals = [y[curr]]
    visited.append(curr)
    for _ in range(steps):
        # find neighbors with Hamming=1
        nbrs = np.where(np.sum(np.abs(X - X[curr]),axis=1)==1)[0]
        # among unvisited choose max y
        nxt = max(nbrs, key=lambda i: y[i])
        curr = nxt
        visited.append(curr)
        vals.append(y[curr])
    return vals


def plot_hill_climb(X, seqs, y_raw, y_smooth, out_path):
    vals_raw = hill_climb(X, seqs, y_raw)
    vals_s = hill_climb(X, seqs, y_smooth)
    plt.figure()
    plt.plot(vals_raw, label='raw')
    plt.plot(vals_s, label='smoothed')
    plt.xlabel('step')
    plt.ylabel('fluorescence')
    plt.title('Hill-Climbing Trajectories')
    plt.legend()
    plt.savefig(out_path)


def bayes_opt_sim(X, seqs, gp_mean, gp_std, k, steps=20):
    # UCB acquisition
    y_pred = gp_mean.copy()
    y_std = gp_std.copy()
    candidate = np.argmax(y_pred + k*y_std)
    vals = [y_pred[candidate]]
    visited = {candidate}
    for _ in range(steps-1):
        # exclude visited
        y_acq = y_pred + k*y_std
        y_acq[list(visited)] = -np.inf
        nxt = np.argmax(y_acq)
        visited.add(nxt)
        vals.append(y_pred[nxt])
    return vals


def plot_bayes_opt(X, seqs, gp_mean, gp_std, k, out_path):
    traj = bayes_opt_sim(X, seqs, gp_mean, gp_std, k)
    plt.figure()
    plt.plot(traj)
    plt.xlabel('proposal step')
    plt.ylabel('predicted mean + k·std')
    plt.title(f'Bayesian Optimization (k={k})')
    plt.savefig(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw_csv', default='../output/processed_gfp_1000.csv')
    p.add_argument('--smoothed_csv', default='../output/processed_gfp_1000_smoothed.csv')
    p.add_argument('--gp_csv', default='../output/processed_gfp_1000_smoothed_gp.csv')
    p.add_argument('--plots_dir', default='../output/plots')
    p.add_argument('--ucb_k', type=float, default=2.0)
    args = p.parse_args()

    out = Path(args.plots_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_raw      = pd.read_csv(args.raw_csv)
    df_smooth   = pd.read_csv(args.smoothed_csv)
    df_gp       = pd.read_csv(args.gp_csv)

    X           = load_vectors(df_raw)
    y_raw       = df_raw['log_fluorescence'].values
    y_smooth    = df_smooth['smooth_fluorescence'].values
    gp_mean     = df_gp['gp_mean'].values
    gp_std      = df_gp['gp_std'].values

    # 1. t-SNE
    plot_tsne(X, y_smooth, out/'tsne.png')
    # 2. 3D PCA & UMAP
    plot_3d_embed(X, y_smooth, 'pca', out/'3d_pca.png')
    plot_3d_embed(X, y_smooth, 'umap', out/'3d_umap.png')
    # 3. Heatmap
    plot_hamming_heatmap(X, y_raw, y_smooth, str(out/'hamming_heatmap.png'))
    # 4. Hill-climbing
    plot_hill_climb(X, df_raw['amino_acid_sequence'], y_raw, y_smooth, out/'hill_climb.png')
    # 5. Bayesian optimization
    plot_bayes_opt(X, df_raw['amino_acid_sequence'], gp_mean, gp_std, args.ucb_k, out/'bayes_opt.png')

if __name__ == '__main__':
    main()
