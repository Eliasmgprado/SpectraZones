from spectraZones.tools.plot_clusters import cm2inch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np


def make_color_matrix(som_shape, predictions, data, data_fn=lambda x: x, max_fn=np.max):
    """
    Assigns data values to the SOM nodes.

    Parameters
    ----------
    som_shape : tuple
        Shape of the SOM.
    predictions : np.ndarray
        Predictions of the SOM.
    data : np.ndarray
        Data to be assigned to the SOM nodes.
    data_fn : function, optional
        Function to be applied to the data. The default is lambda x: x.
    max_fn : function, optional
        Function to get the selected node value. The default is np.max.

    Returns
    -------
    np.ndarray
        SOM nodes with the assigned values.
    """

    color_mat = np.zeros(som_shape)

    for i in range(som_shape[0]):
        for j in range(som_shape[1]):
            node_idxs = np.where((predictions[:, 0] == i) & (predictions[:, 1] == j))
            if len(node_idxs[0]) > 0:
                color_mat[i, j] = max_fn(data_fn(data[node_idxs[0]]))
    return color_mat


def plot_som_results(
    umat,
    target_umat,
    clusters_umat,
    figsize=(19, 10),
    txt_color="k",
    cmap=cm.viridis,
    target_cmap=cm.rainbow,
    clusters_cmap=cm.rainbow,
    figname="",
    dpi=350,
    title_size=8,
    ax_title_size=7,
    ax_tick_size=7,
    legend_size=7,
):
    """
    Plot the U-matrix, clusters, and target values.

    Parameters
    ----------
    umat : np.ndarray
        U-matrix of the SOM.
    target_umat : np.ndarray
        Target values for the SOM.
    clusters_umat : np.ndarray
        Clusters of the SOM.
    figsize : tuple, optional
        Figure size. The default is (19, 10).
    txt_color : str, optional
        Text color. The default is "k".
    cmap : matplotlib.colors.Colormap, optional
        Colormap for the U-matrix. The default is cm.viridis.
    target_cmap : matplotlib.colors.Colormap, optional
        Colormap for the target values. The default is cm.rainbow.
    clusters_cmap : matplotlib.colors.Colormap, optional
        Colormap for the clusters. The default is cm.rainbow.
    figname : str, optional
        Figure name. The default is "" (image is not saved).
    dpi : int, optional
        Dots per inch. The default is 350.
    title_size : int, optional
        Title size. The default is 8.
    ax_title_size : int, optional
        Axis title size. The default is 7.
    ax_tick_size : int, optional
        Axis tick size. The default is 7.
    legend_size : int, optional
        Legend size. The default is 7.
    """

    with plt.rc_context(
        {
            "text.color": txt_color,
            "axes.titlesize": title_size,
            "axes.titlelocation": "left",
            "axes.labelsize": ax_title_size,
            "xtick.color": txt_color,
            "ytick.color": txt_color,
            "xtick.labelsize": ax_tick_size,
            "ytick.labelsize": ax_tick_size,
            "figure.facecolor": "none",
            "figure.dpi": dpi,
            "legend.fontsize": legend_size,
            "figure.figsize": cm2inch(figsize),
            "mathtext.default": "regular",
            "figure.subplot.wspace": 0.3,
        }
    ):

        fig, axs = plt.subplots(1, 3, sharey=True)
        plt.subplots_adjust(wspace=0.05)

        for i, (ax, data, cmap_, title, lab_) in enumerate(
            zip(
                axs,
                [umat, clusters_umat, target_umat],
                [cmap, clusters_cmap, target_cmap],
                ["U-matrix", "Clusters", "Cu ppm"],
                ["a", "b", "c"],
            )
        ):
            if data is None:
                continue

            if i == 1:
                karg = {"vmin": np.min(data) - 0.5, "vmax": np.max(data) + 0.5}
                cbar_kargs = {"ticks": np.arange(np.min(data), np.max(data) + 1)}
            else:
                karg = {}
                cbar_kargs = {}

            plt_ = ax.matshow(data, cmap=cmap_, **karg)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, loc="center")

            plt.setp(ax.spines.values(), color="k", linewidth=0.5)

            divider = make_axes_locatable(ax)
            cax = divider.new_vertical(size="5%", pad=0.1, pack_start=True)
            fig.add_axes(cax)

            cbar = plt.colorbar(
                plt_, cax=cax, orientation="horizontal", drawedges=False, **cbar_kargs
            )
            cbar.ax.tick_params(
                axis="x",
                which="major",
                length=3,
                width=0.5,
                labelsize=legend_size,
                direction="out",
            )
            cbar.outline.set_linewidth(0.5)
            cbar.outline.set_color("k")

            ax.text(
                0,
                1.01,
                lab_,
                transform=ax.transAxes,
                fontsize="9",
                va="bottom",
                fontfamily="roboto",
                fontweight="bold",
            )

        if figname != "":
            plt.savefig(
                figname, dpi=dpi, facecolor="w", edgecolor="w", bbox_inches="tight"
            )
        plt.show()


#
