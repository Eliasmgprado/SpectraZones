import copy
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
from matplotlib import pyplot as plt
from spectraZones.tools.plot_clusters import cm2inch
import matplotlib.gridspec as gridspec

from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np


def perc_fmt(v):
    v = v * 100
    #     print(v)
    if v == 0:
        return ""
    elif v < 1:
        return f"{float(str(v)[:3]):.1f}%"
    else:
        return f"{v:.0f}%"


fmt_func_v = np.vectorize(perc_fmt)


def plot_cluster_contMat_targetBP(
    plot_cont_mat,
    cont_mat_xtick,
    cont_mat_ytick,
    cont_mat_ylabel,
    orig_cont_mat,
    bp_data,
    bp_x,
    bp_y,
    bp_ylabel,
    txt_color="k",
    figsize=(16, 15),
    figname="",
    cbar_label="% samples from cluster $k$",
    dpi=350,
):
    """
    Plots a contingency matrix and a boxplot of the target variable for each cluster.

    Parameters
    ----------
    plot_cont_mat : np.array
        Normalized contingency matrix to be plotted.
        Created by the function `sklearn.metrics.cluster.contingency_matrix` and normalized.
    cont_mat_xtick : list
        List of strings with the xtick labels.
    cont_mat_ytick : list
        List of strings with the ytick labels.
    cont_mat_ylabel : str
        Y axis label of the contingency matrix.
    orig_cont_mat : np.array
        Original contingency matrix.
        Created by the function `sklearn.metrics.cluster.contingency_matrix` without normalization.
    bp_data : pd.DataFrame
        DataFrame with the data to be plotted in the boxplot.
    bp_x : str
        Column name of the x-axis of the boxplot (from bp_data).
    bp_y : str
        Column name of the y-axis of the boxplot (from bp_data).
    bp_ylabel : str
        Y axis label of the boxplot.
    txt_color : str, optional
        Text color of the plot. The default is 'k'.
    figsize : tuple, optional
        Figure size. The default is (16, 15).
    figname : str, optional
        Name of the file to save the figure. The default is "" (not save).
    cbar_label : str, optional
        Label of the colorbar. The default is "% samples from cluster $k$".
    dpi : int, optional
        Resolution of the figure. The default is 350.
    """

    with plt.rc_context(
        {
            "text.color": txt_color,
            "axes.titlesize": 7,
            "axes.titlelocation": "left",
            "axes.labelsize": 7,
            "xtick.color": txt_color,
            "ytick.color": txt_color,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.facecolor": "none",
            "figure.dpi": 350,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "figure.figsize": cm2inch(figsize),
            "mathtext.default": "regular",
            "figure.subplot.hspace": 0,
            #             'text.usetex': True
        }
    ):

        fig = plt.figure()

        charts_gs = gridspec.GridSpec(
            2, 2, hspace=0, wspace=0, width_ratios=[5, 2], height_ratios=[2, 1]
        )
        colorbars_gs = gridspec.GridSpecFromSubplotSpec(
            1, 9, subplot_spec=charts_gs[0, 1], hspace=0
        )

        contMat_gs = charts_gs[0, 0]
        contMat_ax = plt.Subplot(fig, contMat_gs)
        fig.add_subplot(contMat_ax)
        contMat_cbar_gs = colorbars_gs[1]
        contMat_cbar_ax = plt.Subplot(fig, contMat_cbar_gs)
        fig.add_subplot(contMat_cbar_ax)

        targetBP_gs = charts_gs[1, 0]
        targetBP_ax = plt.Subplot(fig, targetBP_gs)
        fig.add_subplot(targetBP_ax)

        #       CONTIGENCY MATRIX
        fmt = lambda x, pos: "{:.0%}".format(x)

        sns.set_style("dark")
        sns.set_context("paper")

        cmap = copy.copy(plt.get_cmap("rainbow"))
        cmap.set_under("#eaeaf2")

        contMat_ax = sns.heatmap(
            plot_cont_mat,
            annot=fmt_func_v(plot_cont_mat),
            xticklabels=cont_mat_xtick,
            yticklabels=cont_mat_ytick,
            cbar_kws={"label": cbar_label, "format": FuncFormatter(fmt)},
            annot_kws={"fontsize": 6, "weight": "bold"},
            #     robust=True,
            linewidth=0.5,
            #     norm=LogNorm(vmin=10),
            vmin=0.00001,
            cmap=cmap,
            fmt="",
            ax=contMat_ax,
            cbar_ax=contMat_cbar_ax,
        )

        contMat_ax.set(xlabel=r"Cluster", ylabel=cont_mat_ylabel)
        contMat_ax.xaxis.set_label_position("top")
        contMat_ax.xaxis.tick_top()

        max_cluster = np.argmax(orig_cont_mat, axis=0)
        # print(max_cluster.shape)
        # print(orig_cont_mat.shape)

        for i, max_c in enumerate(max_cluster):
            contMat_ax.add_patch(
                Rectangle((i, max_c), 1, 1, ec="black", fc="none", lw=1)
            )

        #       Target Box Plot

        sns.set_style("darkgrid")
        sns.set_context("paper")

        targetBP_ax = sns.boxplot(
            data=bp_data,
            x=bp_x,
            y=bp_y,
            width=0.1,
            showcaps=False,
            boxprops={"facecolor": "black"},
            showfliers=False,
            medianprops={"color": "orange", "linewidth": 2},
            meanprops={
                "markeredgecolor": "k",
                "linewidth": 4,
                "marker": "o",
                "markerfacecolor": "w",
                "markersize": 4,
            },
            showmeans=True,
            ax=targetBP_ax,
        )

        targetBP_ax.margins(y=0.04)
        targetBP_ax.minorticks_on()
        targetBP_ax.grid(visible=True, which="major", axis="y", lw=1.5)
        targetBP_ax.grid(visible=True, which="minor", axis="y", lw=0.8)
        targetBP_ax.set_xlabel(r"Cluster")
        targetBP_ax.set_ylabel(bp_ylabel)
        targetBP_ax.tick_params(
            which="major",
            axis="both",
            direction="out",
            length=6,
            width=1,
            colors="k",
            left=True,
            bottom=True,
        )
        yticks = targetBP_ax.get_yticks()
        targetBP_ax.set_yticks(yticks[1:-2])
        targetBP_ax.yaxis.set_minor_locator(AutoMinorLocator(4))

        targetBP_ax.spines["left"].set_visible(False)
        targetBP_ax.spines["bottom"].set_visible(False)

        if figname != "":
            plt.savefig(
                figname, dpi=dpi, facecolor="w", edgecolor="w", bbox_inches="tight"
            )

        plt.show()
