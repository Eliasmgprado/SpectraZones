import seaborn as sns
import copy
from matplotlib import gridspec, pyplot as plt
from matplotlib.colors import LogNorm
from spectraZones.tools.plot_clusters import cm2inch
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
import numpy as np


def perc_fmt(v, min_val=None):
    v = v * 100

    if min_val is not None:
        if v < min_val:
            return f"<{min_val:.1f}"
    if v == 0:
        return ""
    elif v < 1:
        return f"{v:.1f}"
    else:
        return f"{v:.0f}"


def plot_min_litho_heatmap(
    min_count,
    xticklabel,
    yticklabel,
    bar_sizes,
    k_colors,
    figname="",
    figsize=(15, 19),
    min_val=0.5,
    fs=6,
    show_fig=True,
    dpi=500,
    title_size=5,
    ax_title_size=5,
    ax_tick_size=3,
    legend_size=7,
):
    """
    Plot a heatmap showing the percentage of samples in which a specific mineral
    was identified for each cluster/lithology.

    Parameters
    ----------
    min_count : np.ndarray
        Array of mineral counts for each cluster.
    xticklabel : list
        List of cluster labels.
    yticklabel : list
        List of mineral labels.
    bar_sizes : dict
        Dictionary of cluster sizes.
    k_colors : dict
        Dictionary of cluster colors.
    figname : str, optional
        Name of the output file, by default "".
    figsize : tuple, optional
        Figure size, by default (15, 19).
    min_val : float, optional
        Minimum value to display, by default 0.5.
    fs : int, optional
        Font size, by default 6.
    show_fig : bool, optional
        Show the figure, by default True.
    dpi : int, optional
        Dots per inch, by default 500.

    """

    fmt = lambda x, pos: "{:.0%}".format(x)
    fmt_func_v = np.vectorize(perc_fmt)

    bar_counts = np.array(list(bar_sizes.values()))
    bar_keys = list(bar_sizes.keys())

    with plt.rc_context(
        {
            "axes.titlesize": title_size,
            "axes.titlelocation": "left",
            "axes.labelsize": ax_title_size,
            "xtick.labelsize": ax_tick_size,
            "ytick.labelsize": ax_tick_size,
            "figure.facecolor": "none",
            "figure.dpi": dpi,
            "legend.fontsize": legend_size,
            "legend.title_fontsize": legend_size,
            "figure.figsize": cm2inch(figsize),
            "mathtext.default": "regular",
            "figure.subplot.hspace": 0,
        }
    ):
        sns.set_style("dark")
        sns.set_context("paper")

        cmap = copy.copy(plt.get_cmap("rainbow"))
        cmap.set_under("#c6c6d2")

        fig = plt.figure(figsize=cm2inch(figsize), dpi=350)

        outer = gridspec.GridSpec(2, 1, hspace=0.05, height_ratios=[60, 1])
        grid_spec = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[0], hspace=0, height_ratios=[min_count.shape[0], 1]
        )

        ax = plt.Subplot(fig, grid_spec[0])
        fig.add_subplot(ax)
        cbar_ax = plt.Subplot(fig, outer[1])
        fig.add_subplot(cbar_ax)

        cbar_ax.tick_params(labelsize=fs)
        cbar_ax.xaxis.label.set_size(fs)

        ax_ = sns.heatmap(
            min_count,
            annot=fmt_func_v(min_count, min_val),
            xticklabels=xticklabel,
            yticklabels=yticklabel,
            cbar_kws={
                "label": "% of samples from cluster k",
                "format": FuncFormatter(fmt),
                "aspect": 40,
                "orientation": "horizontal",
            },
            annot_kws={"size": fs, "weight": "bold"},
            robust=True,
            linewidth=0.5,
            vmin=min_val / 100,
            cmap=cmap,
            fmt="",
            cbar_ax=cbar_ax,
            ax=ax,
        )

        ax_.xaxis.set_label_position("top")
        ax_.xaxis.tick_top()

        ax_.tick_params(axis="y", which="major", left=True, width=1, length=4)
        ax_.tick_params(axis="x", which="major")

        ix = 0
        trans = transforms.blended_transform_factory(ax_.transData, ax_.transAxes)
        for i, count in enumerate(bar_counts):
            color = k_colors[bar_keys[i]]
            rect_w = count
            ax_.add_patch(
                Rectangle(
                    (ix, 1),
                    rect_w,
                    0.01,
                    ec="white",
                    fc=color,
                    lw=1,
                    transform=trans,
                    clip_on=False,
                )
            )
            ix += rect_w

        ax_.set_axisbelow(False)

        for label in ax_.get_xticklabels():
            label.set_rotation(60)
            label.set_ha("left")
            label.set_fontsize(fs)

        for label in ax_.get_yticklabels():
            label.set_fontsize(fs)

        # CLuster bars

        ax_bar = plt.Subplot(fig, grid_spec[1], sharex=ax)
        fig.add_subplot(ax_bar)

        for i, (k, count) in enumerate(bar_sizes.items()):
            left = None
            if i > 0:
                left = [np.sum(bar_counts[:i])]
            ax_bar.barh(["Cluster"], [count], color=k_colors[k], left=left)

        ax_bar.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_bar.tick_params(axis="y", which="major", left=False, labelleft=False)
        ax_bar.set_xlabel("Cluster", fontsize=fs)

        for i, p in enumerate(ax_bar.patches):
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            ax_bar.text(
                x + width / 2,
                y + height / 2,
                bar_keys[i],
                horizontalalignment="center",
                verticalalignment="center",
                color="white",
                weight="bold",
                fontsize=fs,
            )

        outer.tight_layout(fig, pad=0.5)
        fig.set_tight_layout({"pad": 0.0})
        if figname != "":
            plt.savefig(figname, dpi=dpi, facecolor="w", edgecolor="w")

        if show_fig:
            plt.show()

        return ax_
