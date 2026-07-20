import numpy as np

def plot_panel(ax, x_axis, series, title, ylabel, color):
    """Plot each trial faintly plus a thick mean line with std band."""
    series = np.asarray(series)

    # Faint individual trials
    for row in series:
        ax.plot(x_axis, row, color=color, alpha=0.15, linewidth=1)

    # Thick averaged line
    mean = series.mean(axis=0)
    std = series.std(axis=0)
    ax.fill_between(x_axis, mean - std, mean + std, color=color, alpha=0.15)
    ax.plot(x_axis, mean, color=color, linewidth=2.5, label='mean')

    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(ylabel)
    ax.margins(x=0)