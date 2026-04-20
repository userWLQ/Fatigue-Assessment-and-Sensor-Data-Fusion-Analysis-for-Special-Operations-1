import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from plotting_style import SCI_COLORS, setup_sci_style


COLUMN_LIMIT = 12


def load_and_process(file_path):
    data = []
    original_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            cols = line.strip().split()
            if len(cols) < COLUMN_LIMIT:
                continue
            selected = cols[2:COLUMN_LIMIT]
            try:
                data.append(list(map(float, selected)))
                original_lines.append(line.strip())
            except ValueError:
                continue
    return np.array(data), original_lines


def add_panel_label(ax, label):
    if getattr(ax, "name", "") == "3d":
        ax.text2D(
            -0.08,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight='bold',
            va='top',
            ha='left',
        )
    else:
        ax.text(
            -0.08,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight='bold',
            va='top',
            ha='left',
        )


def main():
    setup_sci_style()
    file_path = 'Simulated_climbing.txt'
    X, valid_lines = load_and_process(file_path)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    with open('Simulated_climbing_clustered_data.txt', 'w') as f:
        for line, label in zip(valid_lines, labels):
            f.write(f'{line}\t{label}\n')

    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_scaled)

    cluster_colors = [
        SCI_COLORS["secondary"],
        SCI_COLORS["accent"],
        SCI_COLORS["highlight"],
        SCI_COLORS["primary"],
    ]
    cluster_cmap = ListedColormap(cluster_colors)
    cluster_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), cluster_cmap.N)

    fig = plt.figure(figsize=(11.6, 4.9))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    scatter_2d = ax1.scatter(
        X_pca_2d[:, 0],
        X_pca_2d[:, 1],
        c=labels,
        cmap=cluster_cmap,
        norm=cluster_norm,
        s=42,
        edgecolors='white',
        linewidths=0.5,
        alpha=0.9,
    )
    ax1.set_title('PCA Clustering (2D)', fontsize=11, pad=8)
    ax1.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0] * 100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1] * 100:.1f}%)')
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    add_panel_label(ax1, 'A')

    scatter_3d = ax2.scatter(
        X_pca_3d[:, 0],
        X_pca_3d[:, 1],
        X_pca_3d[:, 2],
        c=labels,
        cmap=cluster_cmap,
        norm=cluster_norm,
        s=40,
        edgecolor='white',
        linewidth=0.4,
        alpha=0.85,
        depthshade=True,
    )
    ax2.set_title('PCA Clustering (3D)', fontsize=11, pad=8)
    ax2.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0] * 100:.1f}%)', labelpad=8)
    ax2.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1] * 100:.1f}%)', labelpad=8)
    ax2.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2] * 100:.1f}%)', labelpad=6)
    ax2.view_init(elev=24, azim=42)  # type: ignore[attr-defined]
    add_panel_label(ax2, 'B')

    cax = fig.add_axes([0.915, 0.18, 0.018, 0.64])
    cbar = fig.colorbar(
        scatter_3d,
        cax=cax,
        ticks=range(4),
    )
    cbar.set_label('Cluster', rotation=270, labelpad=12)
    cbar.outline.set_visible(False)

    fig.subplots_adjust(left=0.06, right=0.88, top=0.88, bottom=0.14, wspace=0.10)
    fig.savefig('Simulated_climbing_cluster_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


if __name__ == '__main__':
    main()
