# init.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
    

def fit_kmeans(y, Ks=[2, 3, 4], display=False):
    """
    Optimal clusters: fit KMeans on (drift, diffusion) features from y, return best K and cluster stats.
    """
    dy = np.diff(y, axis=0).ravel()
    features = np.column_stack([dy, dy ** 2])
    features = StandardScaler().fit_transform(features)

    fit_stats = []
    cluster_stats = {}
    cluster_labels = {}

    # Compute cluster fits and stats
    for K in Ks:
        km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(features)
        labels = km.labels_
        sil = silhouette_score(features, labels)
        fit_stats.append({"K": K, "silhouette": sil})
        cluster_labels[K] = (labels, km.cluster_centers_)

        stats = []
        for k in range(K):
            dy_k = dy[labels == k]
            abs_dy_k = np.abs(dy_k)
            stats.append({
                "cluster": k,
                "mean_drift": np.mean(dy_k),
                "std_drift": np.std(dy_k),
                "mean_diffusion": np.mean(abs_dy_k),
                "std_diffusion": np.std(abs_dy_k)
            })
        cluster_stats[K] = pd.DataFrame(stats)

    # Select best K
    best_K = max(fit_stats, key=lambda x: x['silhouette'])['K']

    if display:
        fig, axes = plt.subplots(len(Ks), 1, figsize=(6, 3 * len(Ks)), sharex=True, sharey=True)
        if len(Ks) == 1:
            axes = [axes]
        for ax, K in zip(axes, Ks):
            labels, centers = cluster_labels[K]
            ax.scatter(features[:, 0], features[:, 1], c=labels, cmap='tab10', s=6, alpha=0.8)
            ax.scatter(centers[:, 0], centers[:, 1], marker='x', c='black', s=80, lw=2)
            sil = next(f['silhouette'] for f in fit_stats if f['K'] == K)
            ax.set_title(f'KMeans K={K} silhouette={sil:.3f}')
            ax.set_xlabel('std Δy (drift)')
            ax.set_ylabel('std (Δy)^2 (diffusion)')
        plt.tight_layout()
        plt.show()

        # Print fit stats
        df_fit = pd.DataFrame(fit_stats).set_index('K')
        print("\nModel fit stats (silhouette):")
        print(df_fit.to_string())

        # Print cluster stats
        for K in Ks:
            print(f"\nCluster stats for K={K}:")
            print(cluster_stats[K].to_string(index=False))

        print(f"\nBest model: K={best_K}")

    return best_K, cluster_stats
