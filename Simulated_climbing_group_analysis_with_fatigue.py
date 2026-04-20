import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

from plotting_style import SCI_COLORS, save_sci_figure, setup_sci_style


def add_panel_label(ax, label):
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight='bold',
        ha='left',
        va='top',
    )


def load_climbing_group_stats():
    encodings = ['cp949', 'euc-kr', 'gbk', 'latin-1', 'iso-8859-1', 'utf-8']
    data = []

    for encoding in encodings:
        try:
            with open('Simulated_climbing_clustered_3d.txt', 'r', encoding=encoding) as file:
                for line in file:
                    parts = line.strip().split('\t')
                    if len(parts) >= 13:
                        data.append(parts)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if not data:
        with open('Simulated_climbing_clustered_3d.txt', 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) >= 13:
                    data.append(parts)

    columns = [
        'ID', 'Name', 'Age', 'Height', 'Weight',
        'Other1', 'Other2', 'Other3', 'Other4', 'Other5', 'Other6',
        'Climb_Time', 'Group'
    ]
    df = pd.DataFrame(data, columns=columns)

    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Height'] = pd.to_numeric(df['Height'], errors='coerce')
    df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
    df['Climb_Time'] = pd.to_numeric(df['Climb_Time'], errors='coerce')
    df['Group'] = pd.to_numeric(df['Group'], errors='coerce')

    df['Weight_kg'] = df['Weight'] / 2
    df['BMI'] = df['Weight_kg'] / (df['Height'] ** 2)

    group_stats = df.groupby('Group').agg({
        'Climb_Time': 'mean',
        'Age': 'mean',
        'Height': 'mean',
        'BMI': 'mean',
    }).sort_values('Climb_Time')

    group_names = {}
    for i, (group, _) in enumerate(group_stats.iterrows()):
        group_names[group] = ['Poor', 'Fair', 'Good', 'Excellent'][i]

    return group_stats, group_names


def load_fatigue_data(group_names):
    fatigue_by_name = {
        'Excellent': 2.642857143,
        'Good': 2.444444444,
        'Fair': 2.6,
        'Poor': 2.0,
    }
    ordered_names = [group_names[group] for group in group_names]
    ordered_values = [fatigue_by_name[name] for name in ordered_names]
    return ordered_names, ordered_values


def main():
    setup_sci_style()
    group_stats, group_names = load_climbing_group_stats()
    fatigue_labels, fatigue_values = load_fatigue_data(group_names)

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12.6, 4.9),
        gridspec_kw={'width_ratios': [1.7, 1.0]},
    )

    add_panel_label(ax1, 'A')
    add_panel_label(ax2, 'B')

    x = np.arange(len(group_names))
    width = 0.54

    ax1.grid(True, linestyle='--', alpha=0.55, axis='y')
    ax1.set_axisbelow(True)
    bars = ax1.bar(
        x,
        group_stats['Climb_Time'],
        width,
        alpha=0.88,
        color=SCI_COLORS['primary'],
        label='Climb Time',
        edgecolor='white',
        linewidth=0.8,
    )
    ax1.set_xlabel('Group', fontsize=10)
    ax1.set_ylabel('Climb Time (s)', fontsize=10)
    ax1.set_ylim(0, group_stats['Climb_Time'].max() * 1.24)

    for i, value in enumerate(group_stats['Climb_Time']):
        ax1.annotate(
            f'{value:.2f}',
            xy=(i, value * 0.985),
            xytext=(0, -4),
            textcoords='offset points',
            ha='center',
            va='top',
            fontsize=8,
            color='white',
        )

    ax1_right = ax1.twinx()
    marker_size = 48
    age_line = ax1_right.scatter(
        x, group_stats['Age'],
        color=SCI_COLORS['accent'], marker='o', s=marker_size, label='Age', zorder=5
    )
    height_line = ax1_right.scatter(
        x, group_stats['Height'],
        color=SCI_COLORS['highlight'], marker='s', s=marker_size, label='Height', zorder=5
    )
    bmi_line = ax1_right.scatter(
        x, group_stats['BMI'],
        color=SCI_COLORS['secondary'], marker='^', s=marker_size, label='BMI', zorder=5
    )

    ax1_right.plot(x, group_stats['Age'], color=SCI_COLORS['accent'], linestyle='-', linewidth=1.6, alpha=0.8)
    ax1_right.plot(x, group_stats['Height'], color=SCI_COLORS['highlight'], linestyle='-', linewidth=1.6, alpha=0.8)
    ax1_right.plot(x, group_stats['BMI'], color=SCI_COLORS['secondary'], linestyle='-', linewidth=1.6, alpha=0.8)

    for i, (age, height, bmi) in enumerate(zip(group_stats['Age'], group_stats['Height'], group_stats['BMI'])):
        ax1_right.annotate(
            f'{age:.2f}',
            xy=(i, age),
            xytext=(0, 8 if i == 3 else 6),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=7.6,
            color=SCI_COLORS['accent'],
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.8),
        )
        ax1_right.annotate(
            f'{height:.2f}',
            xy=(i, height),
            xytext=(0, -12),
            textcoords='offset points',
            ha='center',
            va='top',
            fontsize=7.6,
            color=SCI_COLORS['highlight'],
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.8),
        )
        ax1_right.annotate(
            f'{bmi:.2f}',
            xy=(i, bmi),
            xytext=(0, -12 if i == 3 else -10),
            textcoords='offset points',
            ha='center',
            va='top',
            fontsize=7.6,
            color=SCI_COLORS['secondary'],
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.8),
        )

    ax1_right.set_ylabel('BMI, Age, and Height', fontsize=10, labelpad=10)
    secondary_min = min(group_stats['Age'].min(), group_stats['Height'].min(), group_stats['BMI'].min())
    secondary_max = max(group_stats['Age'].max(), group_stats['Height'].max(), group_stats['BMI'].max())
    secondary_span = secondary_max - secondary_min if secondary_max != secondary_min else 1.0
    ax1_right.set_ylim(secondary_min - secondary_span * 0.18, secondary_max + secondary_span * 0.2)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{group_names[group]}\n(Group {int(group)})" for group in group_stats.index])
    ax1.set_title('Physical Fitness Summary by Group', fontsize=12, pad=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(
        lines1 + [age_line, height_line, bmi_line],
        labels1 + ['Age', 'Height', 'BMI'],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=True,
        framealpha=0.9,
    )
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax1.spines['top'].set_visible(False)
    ax1_right.spines['top'].set_visible(False)

    fatigue_x = np.arange(len(fatigue_labels))
    fatigue_bars = ax2.bar(
        fatigue_x,
        fatigue_values,
        width=0.54,
        color=SCI_COLORS['highlight'],
        alpha=0.82,
        edgecolor='white',
        linewidth=0.8,
        label='Fatigue score',
    )
    ax2.set_title('Self-reported Post-work Fatigue by Group', fontsize=12, pad=12)
    ax2.set_xlabel('Group', fontsize=10)
    ax2.set_ylabel('Fatigue score', fontsize=10)
    ax2.set_xticks(fatigue_x)
    ax2.set_xticklabels(fatigue_labels, rotation=12)
    ax2.set_ylim(0, max(fatigue_values) + 0.7)
    ax2.grid(True, axis='y')
    ax2.grid(False, axis='x')
    ax2.set_axisbelow(True)

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
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.14, wspace=0.2)
    save_sci_figure('Simulated_climbing_group_analysis_with_fatigue.png')
    plt.show()


if __name__ == '__main__':
    main()
