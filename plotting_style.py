import matplotlib.pyplot as plt


SCI_COLORS = {
    "primary": "#4C566A",
    "secondary": "#5E81AC",
    "accent": "#BF616A",
    "highlight": "#A3BE8C",
    "neutral": "#D8DEE9",
}


def setup_sci_style():
    """Apply a consistent journal-style Matplotlib theme."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.titleweight": "semibold",
        "axes.edgecolor": "#4A4A4A",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.5,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
    })


def save_sci_figure(path):
    """Save current figure with consistent export settings."""
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
