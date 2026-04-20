#!/usr/bin/env python3

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from plotting_style import save_sci_figure, setup_sci_style

column_limit = 12


def load_and_process(file_path):
    data = []
    original_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            cols = line.strip().split()
            if len(cols) < column_limit:
                continue
            selected = cols[3:column_limit]
            try:
                data.append(list(map(float, selected)))
                original_lines.append(line.strip())
            except ValueError:
                continue
    return np.array(data), original_lines


file_path = '6MWT.txt'
setup_sci_style()
X, valid_lines = load_and_process(file_path)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)

with open('6MWT_clustered_data.txt', 'w') as f:
    for line, label in zip(valid_lines, labels):
        f.write(f'{line}\t{label}\n')

print(f'Processing complete! Processed {len(valid_lines)} valid records')
print('Cluster sizes:')
unique, counts = np.unique(labels, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f'Cluster {cluster}: {count} samples')

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(6.8, 4.8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=42, edgecolors='white', linewidths=0.5, alpha=0.9)
ax.set_title('K-means Clustering of 6MWT Data', fontsize=12, pad=12)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
cbar = plt.colorbar(scatter, ax=ax, ticks=range(4), pad=0.02)
cbar.set_label('Cluster', rotation=270, labelpad=12)
save_sci_figure('6MWT_cluster_scatter.png')
plt.show()
