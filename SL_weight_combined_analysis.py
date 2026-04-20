import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from plotting_style import SCI_COLORS, save_sci_figure, setup_sci_style


def extract_number(text):
    match = re.search(r'(\d+(\.\d+)?)', text)
    return float(match.group(1)) if match else None


def add_panel_label(ax, label):
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight='bold',
        ha='left',
        va='bottom',
    )


def load_group_data():
    group_data = defaultdict(lambda: {'weights': [], 'counts': [], 'ages': [], 'heights': []})

    with open('SL.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            tokens = re.split(r'\s+', line)
            if len(tokens) < 5:
                continue

            age = extract_number(tokens[1])
            height = extract_number(tokens[2])
            weight = extract_number(tokens[3])
            count = extract_number(tokens[4])
            group_id = extract_number(tokens[5]) if len(tokens) >= 6 else None

            if None in (age, height, weight, count, group_id):
                continue

            if height > 10:
                height = height / 100.0

            group_data[group_id]['ages'].append(age)
            group_data[group_id]['heights'].append(height)
            group_data[group_id]['weights'].append(weight)
            group_data[group_id]['counts'].append(count)

    return group_data


def build_group_statistics(group_data):
    group_mapping = {1: 'Poor', 3: 'Fair', 2: 'Good', 4: 'Excellent'}
    group_order = ['Poor', 'Fair', 'Good', 'Excellent']

    group_labels = []
    weight_distributions = []
    weight_means = []
    count_means = []
    age_means = []
    bmi_means = []

    for group_name in group_order:
        group_id = next((gid for gid, name in group_mapping.items() if name == group_name), None)
        if group_id not in group_data or not group_data[group_id]['weights']:
            continue

        data = group_data[group_id]
        weights_kg = [value / 2 for value in data['weights']]
        mean_weight = float(np.mean(weights_kg))
        mean_count = float(np.mean(data['counts']))
        mean_age = float(np.mean(data['ages']))
        bmis = [weight_kg / (height_m ** 2) for weight_kg, height_m in zip(weights_kg, data['heights'])]
        mean_bmi = float(np.mean(bmis))

        group_labels.append(group_name)
        weight_distributions.append(weights_kg)
        weight_means.append(mean_weight)
        count_means.append(mean_count)
        age_means.append(mean_age)
        bmi_means.append(mean_bmi)

    return group_labels, weight_distributions, weight_means, count_means, age_means, bmi_means


def main():
    setup_sci_style()

    group_data = load_group_data()
    group_labels, weight_distributions, weight_means, count_means, age_means, bmi_means = build_group_statistics(group_data)

    weight_margin = max(weight_means) - min(weight_means) if len(weight_means) > 1 else max(weight_means) * 0.1
    count_margin = max(count_means) - min(count_means) if len(count_means) > 1 else max(count_means) * 0.1
    secondary_margin = max(max(age_means), max(bmi_means)) - min(min(age_means), min(bmi_means))
    secondary_margin = secondary_margin if secondary_margin else 1.0

    colors = [
        SCI_COLORS['accent'],
        '#EBCB8B',
        SCI_COLORS['highlight'],
        SCI_COLORS['secondary'],
    ]

    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        figsize=(16.6, 4.8),
        gridspec_kw={'width_ratios': [0.98, 1.14, 1.14]},
    )

    add_panel_label(ax1, 'A')
    add_panel_label(ax2, 'B')
    add_panel_label(ax3, 'C')

    box = ax1.boxplot(
        weight_distributions,
        labels=group_labels,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color='white', linewidth=1.4),
        whiskerprops=dict(color='#4C566A', linewidth=1.0),
        capprops=dict(color='#4C566A', linewidth=1.0),
        flierprops=dict(
            marker='o',
            markersize=4.5,
            markerfacecolor='white',
            markeredgecolor='#4C566A',
            alpha=0.8,
        ),
    )

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.84)
        patch.set_edgecolor('white')
        patch.set_linewidth(0.8)

    for i, values in enumerate(weight_distributions, start=1):
        if not values:
            continue

        q3 = np.percentile(values, 75)
        mean_val = float(np.mean(values))
        ax1.text(
            i,
            q3 + 0.9,
            f'{mean_val:.1f}',
            ha='center',
            va='bottom',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.82, edgecolor='none', pad=1.1),
        )

    ax1.set_title('Weight Distribution by Group', fontsize=11.5, pad=8)
    ax1.set_xlabel('Group', fontsize=10)
    ax1.set_ylabel('Weight (kg)', fontsize=10)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.35)
    ax1.grid(False, axis='x')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2_right = ax2.twinx()
    bars = ax2.bar(
        group_labels,
        weight_means,
        color=SCI_COLORS['primary'],
        alpha=0.84,
        width=0.54,
        edgecolor='white',
        linewidth=0.8,
        label='Mean Weight',
        zorder=1,
    )
    count_line, = ax2_right.plot(
        group_labels,
        count_means,
        'o-',
        color=SCI_COLORS['secondary'],
        linewidth=1.6,
        markersize=5.5,
        label='Mean Count',
        zorder=3,
    )

    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.18,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.82, edgecolor='none', pad=1.1),
        )

    for i, count in enumerate(count_means):
        ax2_right.text(
            i,
            count + 0.28,
            f'{count:.2f}',
            ha='center',
            va='bottom',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.82, edgecolor='none', pad=1.1),
        )

    ax2.set_title('Mean Completed Count and Weight by Group', fontsize=11.5, pad=8)
    ax2.set_xlabel('Group', fontsize=10)
    ax2.set_ylabel('Weight (kg)', fontsize=10)
    ax2_right.set_ylabel('Completed Count', fontsize=10)
    ax2.set_ylim(min(weight_means) - weight_margin * 0.35, max(weight_means) + weight_margin * 0.55)
    ax2_right.set_ylim(0, max(count_means) + max(2.0, count_margin * 0.7))
    ax2.grid(True, axis='y', linestyle='--', alpha=0.35)
    ax2.grid(False, axis='x')
    ax2.spines['top'].set_visible(False)
    ax2_right.spines['top'].set_visible(False)

    lines1, labels1 = ax2.get_legend_handles_labels()
    ax2.legend(
        lines1 + [count_line],
        labels1 + ['Mean Count'],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=True,
        framealpha=0.9,
    )

    ax3_right = ax3.twinx()
    count_bars = ax3.bar(
        group_labels,
        count_means,
        color=SCI_COLORS['primary'],
        alpha=0.84,
        width=0.54,
        edgecolor='white',
        linewidth=0.8,
        label='Mean Count',
        zorder=1,
    )
    age_line, = ax3_right.plot(
        group_labels,
        age_means,
        's--',
        color=SCI_COLORS['secondary'],
        linewidth=1.6,
        markersize=5.0,
        label='Mean Age',
        zorder=3,
    )
    bmi_line, = ax3_right.plot(
        group_labels,
        bmi_means,
        'o-',
        color=SCI_COLORS['accent'],
        linewidth=1.6,
        markersize=5.5,
        label='Mean BMI',
        zorder=3,
    )

    for bar in count_bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.2,
            f'{height:.1f}',
            ha='center',
            va='bottom',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.82, edgecolor='none', pad=1.1),
        )

    for i, (age_val, bmi_val) in enumerate(zip(age_means, bmi_means)):
        age_offset = -0.15 if i == 0 else 0.2
        age_va = 'top' if i == 0 else 'bottom'
        ax3_right.text(
            i,
            age_val + age_offset,
            f'{age_val:.1f}',
            ha='center',
            va=age_va,
            fontsize=8,
            color=SCI_COLORS['secondary'],
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.0),
        )
        ax3_right.text(
            i,
            bmi_val - 0.35,
            f'{bmi_val:.1f}',
            ha='center',
            va='top',
            fontsize=8,
            color=SCI_COLORS['accent'],
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.0),
        )

    ax3.set_title('Completed Count, Age, and BMI by Group', fontsize=11.5, pad=8)
    ax3.set_xlabel('Group', fontsize=10)
    ax3.set_ylabel('Completed Count', fontsize=10)
    ax3_right.set_ylabel('Age / BMI', fontsize=10)
    ax3.set_ylim(min(count_means) - count_margin * 0.25, max(count_means) + max(1.5, count_margin * 0.45))
    max_secondary = max(max(age_means), max(bmi_means))
    min_secondary = min(min(age_means), min(bmi_means))
    ax3_right.set_ylim(min_secondary - secondary_margin * 0.15, max_secondary + secondary_margin * 0.22)
    ax3.grid(True, axis='y', linestyle='--', alpha=0.35)
    ax3.grid(False, axis='x')
    ax3.spines['top'].set_visible(False)
    ax3_right.spines['top'].set_visible(False)
    ax3_lines, ax3_labels = ax3.get_legend_handles_labels()
    ax3.legend(
        ax3_lines + [age_line, bmi_line],
        ax3_labels + ['Mean Age', 'Mean BMI'],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=True,
        framealpha=0.9,
    )

    fig.subplots_adjust(left=0.055, right=0.975, top=0.88, bottom=0.13, wspace=0.24)
    save_sci_figure('SL_weight_combined_analysis.png')
    plt.show()


if __name__ == '__main__':
    main()
