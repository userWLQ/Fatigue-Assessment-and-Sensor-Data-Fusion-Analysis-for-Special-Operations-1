import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from plotting_style import SCI_COLORS, setup_sci_style


COLUMN_LIMIT = 12


def read_correlation_data(file_path):
    """Read columns needed for Spearman correlation analysis."""
    cols = [[] for _ in range(12)]

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 14:
                print(f"Skipping line {line_num + 1}: insufficient columns")
                continue

            try:
                for i in range(11):
                    cols[i].append(float(parts[i + 2]))
                cols[11].append(float(parts[13]))
            except (IndexError, ValueError) as e:
                print(f"Skipping line {line_num + 1}: data error - {str(e)}")
                continue

    return cols


def compute_spearman(data):
    """Compute Spearman coefficients against the target column."""
    target = data[-1]
    correlations = []

    for idx in range(11):
        feature = data[idx]
        if len(feature) != len(target):
            raise ValueError(
                f"Data length mismatch (column {idx + 3}: {len(feature)} vs column 13: {len(target)})"
            )
        if len(feature) < 2:
            correlations.append(np.nan)
            continue

        corr, _ = spearmanr(feature, target)
        correlations.append(corr)

    return correlations


def load_cluster_data(file_path):
    """Read columns needed for clustering analysis."""
    data = []
    original_lines = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            cols = line.strip().split()
            if len(cols) < COLUMN_LIMIT:
                continue

            selected = cols[3:COLUMN_LIMIT]
            try:
                data.append(list(map(float, selected)))
                original_lines.append(line.strip())
            except ValueError:
                continue

    return np.array(data), original_lines


def add_panel_label(ax, label):
    ax.text(
        -0.10,
        1.015,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight='bold',
        va='top',
        ha='left',
    )


def get_correlation_label_adjustments(values, pos_offset=0.06, neg_offset=-0.10, min_gap=0.12):
    """Spread label offsets and stagger x positions to reduce overlap."""
    adjustments = []
    prev_y = None
    prev_text_y = None
    prev_sign = None
    for idx, value in enumerate(values):
        base_y = pos_offset if value >= 0 else neg_offset
        text_y = value + base_y
        x_shift = 0.0
        if prev_y is not None and abs(value - prev_y) < min_gap:
            if value >= 0:
                text_y = max(text_y, prev_text_y + min_gap)
            else:
                text_y = min(text_y, prev_text_y - min_gap)
            x_shift = -0.14 if idx % 2 == 0 else 0.14
        elif prev_sign is not None and value >= 0 and prev_sign >= 0 and abs(value - prev_y) < (min_gap * 1.6):
            x_shift = -0.10 if idx % 2 == 0 else 0.10

        # The left-most positive labels tend to crowd together, so add a stronger manual spread.
        if idx in (0, 1, 2, 3) and value > 0:
            manual_x = {0: -0.18, 1: 0.18, 2: -0.10, 3: 0.10}
            manual_y = {0: 0.10, 1: 0.18, 2: 0.06, 3: 0.14}
            x_shift += manual_x.get(idx, 0.0)
            text_y = max(text_y, value + manual_y.get(idx, pos_offset))
        adjustments.append((x_shift, text_y - value))
        prev_y = value
        prev_text_y = text_y
        prev_sign = value
    return adjustments


def main():
    file_path = '6MWT.txt'
    setup_sci_style()

    correlation_data = read_correlation_data(file_path)
    spearman_corrs = compute_spearman(correlation_data)

    cluster_data, valid_lines = load_cluster_data(file_path)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_data)
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    with open('6MWT_clustered_data.txt', 'w') as f:
        for line, label in zip(valid_lines, labels):
            f.write(f'{line}\t{label}\n')

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    cluster_colors = [
        SCI_COLORS["secondary"],
        SCI_COLORS["accent"],
        SCI_COLORS["highlight"],
        SCI_COLORS["primary"],
    ]
    cluster_cmap = ListedColormap(cluster_colors)
    cluster_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), cluster_cmap.N)

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(11.4, 4.4),
        gridspec_kw={'width_ratios': [1, 1.06]},
    )

    x_positions = np.arange(1, 12)
    column_labels = np.arange(3, 14)
    correlation_adjustments = get_correlation_label_adjustments(spearman_corrs)
    ax1.plot(
        x_positions,
        spearman_corrs,
        marker='s',
        markersize=4.5,
        markerfacecolor='white',
        markeredgewidth=1.0,
        color=SCI_COLORS["accent"],
        linewidth=1.6,
        linestyle='-',
        label='Correlation coefficient',
    )
    threshold = 0.6
    for x, y, (x_shift, y_offset) in zip(x_positions, spearman_corrs, correlation_adjustments):
        if abs(y) > threshold:
            va = 'bottom' if y >= 0 else 'top'
            ax1.text(
                x + x_shift,
                y + y_offset,
                f'{y:.2f}',
                ha='center',
                va=va,
                fontsize=8,
                color=SCI_COLORS["secondary"],
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=0.8),
            )
    ax1.axhline(0, color=SCI_COLORS["primary"], linewidth=0.8, linestyle='--')
    ax1.set_xlabel('Column index')
    ax1.set_ylabel(r'Spearman $\rho$', labelpad=12)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(column_labels)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_xlim(0.5, 11.5)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.7)
    ax1.set_yticks(np.linspace(-1.0, 1.0, 5))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.text(
        0.00,
        1.005,
        'Spearman correlation',
        transform=ax1.transAxes,
        ha='left',
        va='bottom',
        fontsize=10,
        fontweight='semibold',
        color=SCI_COLORS["primary"],
    )
    add_panel_label(ax1, 'A')

    scatter = ax2.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap=cluster_cmap,
        norm=cluster_norm,
        s=40,
        edgecolors='white',
        linewidths=0.5,
        alpha=0.92,
    )
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    ax2.grid(True, alpha=0.28)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.text(
        0.00,
        1.005,
        'PCA clustering',
        transform=ax2.transAxes,
        ha='left',
        va='bottom',
        fontsize=10,
        fontweight='semibold',
        color=SCI_COLORS["primary"],
    )
    cbar = fig.colorbar(scatter, ax=ax2, ticks=range(4), pad=0.015, fraction=0.036)
    cbar.set_label('Cluster', rotation=270, labelpad=12)
    cbar.outline.set_visible(False)
    add_panel_label(ax2, 'B')

    fig.subplots_adjust(left=0.10, right=0.95, top=0.87, bottom=0.17, wspace=0.08)
    fig.savefig(
        '6MWT_correlation_cluster_combined.png',
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
    )
    plt.show()


if __name__ == "__main__":
    main()
