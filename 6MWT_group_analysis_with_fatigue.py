import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from plotting_style import SCI_COLORS, save_sci_figure, setup_sci_style


group_mapping = {
    0: 'Good group',
    1: 'Excellent group',
    2: 'Overweight group',
    3: 'Tall group',
}

group_order = [3, 2, 0, 1]
fatigue_scores = {
    'Tall group': 3.0,
    'Overweight group': 2 + 3 / 7,
    'Good group': 2.42,
    'Excellent group': 2.3,
}


def load_group_means():
    data = []
    with open('6MWT_clustered_data.txt', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('['):
                cols = line.split()
                if len(cols) >= 15:
                    try:
                        age = float(cols[2])
                        height = float(cols[3])
                        weight = float(cols[4])
                        laps = float(cols[13])
                        group = int(cols[14])
                        bmi = weight / 2 / (height ** 2)
                        data.append([group, age, height, bmi, laps])
                    except (ValueError, IndexError):
                        continue

    df = pd.DataFrame(data, columns=['Group', 'Age', 'Height', 'BMI', 'Laps'])
    group_means = df.groupby('Group').mean().reset_index().sort_values('Group')

    ordered = []
    for g in group_order:
        group_data = group_means[group_means['Group'] == g]
        if not group_data.empty:
            ordered.append(group_data)

    if not ordered:
        raise ValueError('No valid group data found in 6MWT_clustered_data.txt')

    return pd.concat(ordered).reset_index(drop=True)


def add_panel_label(ax, label):
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
    group_means = load_group_means()

    group_names = group_means['Group'].map(group_mapping)
    laps = group_means['Laps'].to_numpy()
    bmi = group_means['BMI'].to_numpy()
    age = group_means['Age'].to_numpy()
    fatigue = np.array([fatigue_scores[name] for name in group_names])

    x = np.arange(len(group_names))
    width = 0.54

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12.2, 4.7),
        gridspec_kw={'width_ratios': [1.45, 1.0]},
    )

    ax1_right = ax1.twinx()

    bars = ax1.bar(
        x,
        laps,
        width,
        color=SCI_COLORS['primary'],
        alpha=0.88,
        edgecolor='white',
        linewidth=0.8,
        label='Laps',
    )
    bmi_line, = ax1_right.plot(
        x,
        bmi,
        'o-',
        color=SCI_COLORS['accent'],
        linewidth=1.6,
        markersize=5.2,
        label='BMI',
    )
    age_line, = ax1_right.plot(
        x,
        age,
        's--',
        color=SCI_COLORS['secondary'],
        linewidth=1.6,
        markersize=4.8,
        label='Age',
    )

    ax1.set_ylabel('Laps (count)', fontsize=10)
    ax1_right.set_ylabel('BMI and Age', fontsize=10, labelpad=10)
    ax1.set_ylim(0, max(laps) * 1.24)

    secondary_min = min(bmi.min(), age.min())
    secondary_max = max(bmi.max(), age.max())
    secondary_span = secondary_max - secondary_min if secondary_max != secondary_min else 1.0
    ax1_right.set_ylim(secondary_min - secondary_span * 0.18, secondary_max + secondary_span * 0.20)

    for bar in bars:
        height = bar.get_height()
        ax1.annotate(
            f'{height:.2f}',
            xy=(bar.get_x() + bar.get_width() / 2, height * 0.985),
            xytext=(0, -4),
            textcoords='offset points',
            ha='center',
            va='top',
            fontsize=8,
            color='white',
        )

    for i, (b, a) in enumerate(zip(bmi, age)):
        bmi_offset = 6 if i != len(bmi) - 1 else 12
        age_offset = -10 if i != len(age) - 1 else -16
        ax1_right.annotate(
            f'{b:.2f}',
            xy=(i, b),
            xytext=(0, bmi_offset),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=7.6,
            color=SCI_COLORS['accent'],
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.8),
        )
        ax1_right.annotate(
            f'{a:.2f}',
            xy=(i, a),
            xytext=(0, age_offset),
            textcoords='offset points',
            ha='center',
            va='top',
            fontsize=7.6,
            color=SCI_COLORS['secondary'],
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.8),
        )

    ax1.legend(
        [bars, bmi_line, age_line],
        ['Laps', 'BMI', 'Age'],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=True,
        framealpha=0.9,
    )
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax1.grid(True, axis='y')
    ax1.grid(False, axis='x')
    ax1.spines['top'].set_visible(False)
    ax1_right.spines['top'].set_visible(False)
    ax1.set_title('Physical Fitness Summary by Group', fontsize=12, pad=12)
    add_panel_label(ax1, 'A')

    fatigue_bars = ax2.bar(
        x,
        fatigue,
        width,
        color=SCI_COLORS['highlight'],
        alpha=0.82,
        edgecolor='white',
        linewidth=0.8,
        label='Fatigue score',
    )
    ax2.set_ylabel('Fatigue score', fontsize=10)
    ax2.set_xlabel('Group', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(group_names, rotation=12)
    ax2.set_ylim(0, max(fatigue) + 0.7)

    for bar in fatigue_bars:
        height = bar.get_height()
        ax2.annotate(
            f'{height:.2f}'.rstrip('0').rstrip('.'),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=8,
            color=SCI_COLORS['highlight'],
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.82),
        )

    ax2.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=1,
        frameon=True,
        framealpha=0.9,
    )
    ax2.grid(True, axis='y')
    ax2.grid(False, axis='x')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('Self-reported Post-work Fatigue by Group', fontsize=12, pad=12)
    add_panel_label(ax2, 'B')

    fig.subplots_adjust(wspace=0.22, top=0.88)
    save_sci_figure('6MWT_group_analysis_with_fatigue.png')
    plt.show()


if __name__ == '__main__':
    main()
