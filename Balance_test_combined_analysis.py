import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from plotting_style import SCI_COLORS, setup_sci_style


def read_trial_columns(file_path):
    first_trial, second_trial = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                first_trial.append(float(parts[5]))
                second_trial.append(float(parts[6]))
            except (IndexError, ValueError):
                continue
    return first_trial, second_trial


def calculate_quartiles(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered = [x for x in data if lower_bound <= x <= upper_bound]
    return {'q1': q1, 'median': np.median(data), 'q3': q3, 'filtered': filtered}


def polynomial(x, *coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))


def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def generate_equation(model_name, params):
    if 'polynomial' in model_name:
        order = len(params) - 1
        terms = []
        for i, p in enumerate(params):
            exp = order - i
            term = f"{p:.2e}x^{exp}" if exp > 1 else f"{p:.2e}x" if exp == 1 else f"{p:.2e}"
            terms.append(term)
        return f"$y = {' + '.join(terms)}$"
    if 'gaussian' in model_name:
        return f"$y = {params[0]:.2e}e^{{-(x-{params[1]:.2f})^2/(2\\cdot{params[2]:.2f}^2)}}$"
    return f"Fitted curve: {model_name}"


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
    first_trial, second_trial = read_trial_columns('Balance_Test.txt')
    trial_stats = [calculate_quartiles(first_trial), calculate_quartiles(second_trial)]

    mean_data = np.genfromtxt('Balance_test_Mean.txt').flatten()
    num_bins = 25
    counts, bin_edges = np.histogram(mean_data, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    models = {
        'Quadratic polynomial': (lambda x, a, b, c: polynomial(x, a, b, c)),
        'Normal distribution': gaussian,
        'Quartic polynomial': (lambda x, a, b, c, d, e: polynomial(x, a, b, c, d, e)),
    }
    best_model = None
    best_r2 = -np.inf
    line_colors = [SCI_COLORS['secondary'], SCI_COLORS['highlight'], '#8FBCBB']

    for (name, model), color in zip(models.items(), line_colors):
        try:
            params, _ = curve_fit(model, bin_centers, counts, maxfev=5000)
            y_pred = model(bin_centers, *params)
            ss_res = np.sum((counts - y_pred) ** 2)
            ss_tot = np.sum((counts - np.mean(counts)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            x_fit = np.linspace(min(bin_centers), max(bin_centers), 300)
            y_fit = model(x_fit, *params)
            if r2 > best_r2:
                best_r2 = r2
                best_model = (name, params, x_fit, y_fit)
        except Exception as e:
            print(f"{name} fitting failed: {str(e)}")

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.5),
        gridspec_kw={'width_ratios': [0.95, 1.15]},
    )

    box = ax1.boxplot(
        [trial_stats[0]['filtered'], trial_stats[1]['filtered']],
        positions=[1, 2],
        widths=0.5,
        patch_artist=True,
        labels=['First trial', 'Second trial'],
        showfliers=False,
        medianprops=dict(color=SCI_COLORS['accent'], linewidth=1.6),
        whiskerprops=dict(color=SCI_COLORS['primary'], linewidth=1.2),
        capprops=dict(color=SCI_COLORS['primary'], linewidth=1.2),
    )
    for patch, color in zip(box['boxes'], [SCI_COLORS['primary'], SCI_COLORS['secondary']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
        patch.set_edgecolor('white')
        patch.set_linewidth(0.8)

    for pos, stats in zip([1, 2], trial_stats):
        ax1.plot([pos - 0.24, pos + 0.24], [stats['q1']] * 2, color=SCI_COLORS['highlight'], linewidth=1.3, linestyle='--')
        ax1.plot([pos - 0.24, pos + 0.24], [stats['q3']] * 2, color=SCI_COLORS['secondary'], linewidth=1.3, linestyle='--')
        ax1.text(pos + 0.30, stats['median'], f"Median\n{stats['median']:.2f}", ha='left', va='center', fontsize=7.5, bbox=dict(facecolor='white', alpha=0.84, edgecolor='none', pad=0.9))
    ax1.set_title('Eyes-closed Single-leg Stance Trials', fontsize=11, pad=8)
    ax1.set_ylabel('Duration')
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.grid(False, axis='x')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    add_panel_label(ax1, 'A')

    ax2.bar(
        bin_centers,
        counts,
        width=bin_edges[1] - bin_edges[0],
        alpha=0.74,
        edgecolor='white',
        linewidth=0.7,
        color=SCI_COLORS['primary'],
        label='Observed frequency',
    )
    for (name, model), color in zip(models.items(), line_colors):
        try:
            params, _ = curve_fit(model, bin_centers, counts, maxfev=5000)
            x_fit = np.linspace(min(bin_centers), max(bin_centers), 300)
            y_fit = model(x_fit, *params)
            y_pred = model(bin_centers, *params)
            ss_res = np.sum((counts - y_pred) ** 2)
            ss_tot = np.sum((counts - np.mean(counts)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            ax2.plot(x_fit, y_fit, '--', lw=1.25, alpha=0.88, color=color, label=f'{name} (R2={r2:.3f})')
        except Exception:
            continue

    if best_model:
        name, params, x_fit, y_fit = best_model
        ax2.plot(x_fit, y_fit, color=SCI_COLORS['accent'], lw=2.2, label=f'Best fit: {name} (R2={best_r2:.3f})')
        equation = generate_equation(name.lower(), params)
        ax2.text(
            0.98,
            0.62,
            equation,
            transform=ax2.transAxes,
            fontsize=8.2,
            ha='right',
            va='top',
            bbox=dict(facecolor='white', alpha=0.88, edgecolor='none', pad=1.0),
        )

    ax2.set_title('Distribution of Mean Trial Durations', fontsize=11, pad=8)
    ax2.set_xlabel('Averaged duration')
    ax2.set_ylabel('Count')
    ax2.legend(loc='upper right', fontsize=7.4)
    ax2.grid(True, alpha=0.28)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    add_panel_label(ax2, 'B')

    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.16, wspace=0.22)
    fig.savefig('Balance_test_trials_and_fitting_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


if __name__ == '__main__':
    main()
