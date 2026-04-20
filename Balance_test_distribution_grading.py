import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from plotting_style import SCI_COLORS, save_sci_figure, setup_sci_style

mu = 31.2
sigma = 25.97
x_min = 0
x_max = 100

total_area = norm.cdf(x_max, mu, sigma) - norm.cdf(x_min, mu, sigma)
quantiles = [x_min]
for i in range(1, 4):
    target_cdf = norm.cdf(x_min, mu, sigma) + total_area * i / 4
    quantiles.append(norm.ppf(target_cdf, mu, sigma))
quantiles.append(x_max)

x = np.linspace(x_min, x_max, 1000)
y = norm.pdf(x, mu, sigma)
ymax = y.max()

setup_sci_style()
fig, ax = plt.subplots(figsize=(7.4, 4.9))
ax.plot(x, y, color='black', linewidth=1.8, label='Probability density')
colors = [SCI_COLORS['accent'], '#EBCB8B', SCI_COLORS['highlight'], SCI_COLORS['secondary']]
labels = ['Poor', 'Fair', 'Good', 'Excellent']

for i in range(4):
    start = max(quantiles[i], x_min)
    end = min(quantiles[i + 1], x_max)
    mask = (x >= start) & (x <= end)
    ax.fill_between(x[mask], y[mask], color=colors[i], alpha=0.58, label=labels[i])
    center = (start + end) / 2
    ax.text(center, ymax * 0.14, labels[i], ha='center', va='center', fontsize=8.5)

for q in quantiles[1:-1]:
    ax.axvline(q, color=SCI_COLORS['primary'], linestyle='--', alpha=0.72, linewidth=1.0)
    ax.text(q, ymax * 0.88, f'{q:.1f}', ha='center', va='top', rotation=90, fontsize=7.6, bbox=dict(facecolor='white', alpha=0.86, edgecolor='none', pad=0.8))

stats_text = f'mu = {mu:.1f}\nsigma = {sigma:.2f}\nEffective area = {total_area * 100:.1f}%'
ax.text(
    0.975,
    0.78,
    stats_text,
    transform=ax.transAxes,
    ha='right',
    va='top',
    fontsize=8.5,
    bbox=dict(facecolor='white', alpha=0.88, edgecolor='none', pad=1.2),
)

ax.set_title('Normal Distribution Based Ability Grading', fontsize=12, pad=12)
ax.set_xlabel('Ability value', fontsize=10)
ax.set_ylabel('Probability density', fontsize=10)
ax.grid(True, linestyle=':', alpha=0.55)
ax.set_xlim(x_min, x_max)
ax.set_ylim(0, ymax * 1.08)
ax.legend(
    loc='upper right',
    bbox_to_anchor=(0.98, 0.98),
    ncol=2,
    frameon=True,
    framealpha=0.92,
    borderaxespad=0.0,
)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
save_sci_figure('Balance_Test_normal_distribution_grading.png')
plt.show()
