from sklearn.decomposition import PCA
from spectraZones.tools.plot_clusters import cm2inch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from textwrap import fill
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import numpy as np


def print_cluster_scatter_results(
    arr,
    cluster_labels,
    lito_labels,
    arr_lito,
    arr_clust,
    color_field=None,
    pca_trans=None,
    points=None,
    points_labels=None,
    symbols=None,
    cmap=None,
    field_cmap=None,
    lito_cmap=None,
    figsize=(19, 14),
    txt_color="k",
    figname="",
    lito_names={},
    pcs=[0, 1],
    xlim=[-1, 1.3],
    ylim=[-0.5, 1.1],
    color_field_vlim=None,
):
    """

    This function generates a scatter plot for cluster analysis results,
    visualizing different clusters and lithologies in a PCA-transformed space.

    Parameters
    ----------
    arr : np.array
        Array with the color field data features (3rd plot).
    cluster_labels : np.array
        Array with the cluster labels (1fs plot).
    lito_labels : np.array
        Array with the lithology labels (2nd plot).
    arr_lito : np.array
        Array with the lithology data (2nd plot).
    arr_clust : np.array
        Array with the cluster data (1fs plot).
    color_field : np.array
        Array with the color field data (3rd plot).
    pca_trans: PCA
        PCA transformation object.
    points : np.array
        Array with the points to be plotted.
    points_labels : np.array
        Array with the points labels.
    symbols : dict
        Dictionary with the symbols to be plotted.
    cmap : matplotlib colormap
        Colormap for the cluster plot.
    field_cmap : matplotlib colormap
        Colormap for the field plot.
    lito_cmap : matplotlib colormap
        Colormap for the lithology plot.
    figsize : tuple
        Figure size.
        Default is (19, 14).
    txt_color : str
        Text color.
        Default is black.
    figname : str
        Figure name.
    lito_names : dict
        Dictionary with the lithology names.
    pcs : list
        List with the principal components to be plotted.
        Default is [0, 1].

    """

    if cmap is None:
        cmap = cm.Dark2

    if field_cmap is None:
        field_cmap = cm.rainbow

    plt.style.use("classic")

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
            "figure.dpi": 600,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "figure.figsize": cm2inch(figsize),
            "mathtext.default": "regular",
            "figure.subplot.hspace": 0,
        }
    ):

        fig = plt.figure()
        outer = gridspec.GridSpec(2, 1, hspace=0.1, height_ratios=[5, 1.5])

        charts_gs = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer[0], hspace=0, wspace=0
        )

        legend_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[1], hspace=0.2
        )

        colorbars_gs = gridspec.GridSpecFromSubplotSpec(
            8, 1, subplot_spec=legend_gs[1], hspace=0
        )

        #         fig, axs = plt.subplots(4, 2, sharex=True, sharey=True)
        fig.patch.set_facecolor("none")

        n_clusters = len(np.unique(cluster_labels))

        if pca_trans is None:
            pca = PCA(n_components=3)
            pca_trans = pca.fit(arr)
        arr_ = pca_trans.transform(arr)
        arr_lito_ = pca_trans.transform(arr_lito)
        arr_clust_ = pca_trans.transform(arr_clust)

        point_leg = None

        plots = []

        chart_axes = np.empty(shape=(3), dtype=object)

        for i, (gs, dt, c, cmap_, title, lab_, isCat, hasMarker) in enumerate(
            zip(
                charts_gs,
                [arr_clust_, arr_lito_, arr_],
                [
                    cluster_labels.astype(float),
                    lito_labels.astype(float),
                    color_field.astype(float),
                ],
                [cmap, lito_cmap, field_cmap],
                ["Clusters", "Lithologies", "Cu ppm"],
                ["a", "b", "c"],
                [1, 1, 0],
                [0, 0, 0],
            )
        ):

            ax = plt.Subplot(fig, gs, sharex=chart_axes[0])
            fig.add_subplot(ax)
            chart_axes[i] = ax

            plt.setp(ax.spines.values(), color="k", linewidth=0.5)

            if isCat:
                karg = {"vmin": np.min(0) - 0.5, "vmax": np.max(cmap_.N - 1) + 0.5}
            else:
                karg = {}
                if color_field_vlim is not None:
                    karg = {"vmin": color_field_vlim[0], "vmax": color_field_vlim[1]}
            plt_ = ax.scatter(
                dt[:, pcs[0]],
                dt[:, pcs[1]],
                marker=".",
                s=3,
                lw=0,
                alpha=0.6,
                cmap=cmap_,
                c=c,
                edgecolor="k",
                **karg,
            )

            ax.set_aspect("equal")
            ax.set_facecolor("none")

            plots.append(plt_)

            # Draw points
            if points is not None and hasMarker:
                points_ = pca_trans.transform(points)
                if points_labels is not None:

                    if isinstance(points_labels[0], str):
                        colmap = {
                            label: n for n, label in enumerate(set(points_labels))
                        }
                        pts_colors = [colmap[label] for label in points_labels]
                    else:
                        pts_colors = points_labels

                    point_colors = pts_colors
                else:
                    point_colors = [0 for i in range(points_.shape[0])]

                if symbols is not None:
                    unique_ = np.unique(points_labels)
                    for label in unique_:
                        if label not in symbols:
                            print(f"Error Missing label on symbols for: {label}")
                            continue
                        idxs_ = np.where(points_labels == label)
                        marker = symbols[label]

                        scatter = ax.scatter(
                            points_[idxs_, pcs[0]],
                            points_[idxs_, pcs[1]],
                            marker=marker["marker"],
                            s=marker["size"] / 5,
                            lw=marker["lw"] / 3,
                            alpha=1,
                            c=marker["color"],
                            edgecolor=marker["edgecolor"],
                        )
                if points_labels is not None:
                    labels = list(set(points_labels))
                    labels.sort()
                else:
                    labels = list(set(point_colors))

                if symbols is not None:

                    legend_elements = [
                        Line2D(
                            [0],
                            [0],
                            marker=symbols[label]["marker"],
                            label=label,
                            markerfacecolor=symbols[label]["color"],
                            markersize=4,
                            markeredgewidth=symbols[label]["lw"] / 2,
                            markeredgecolor=symbols[label]["edgecolor"],
                            lw=0,
                            alpha=0.8,
                        )
                        for label in labels
                    ]
                else:
                    legend_elements = [
                        Line2D(
                            [0],
                            [0],
                            marker=".",
                            label=labels[i],
                            markerfacecolor=cm.tab20(list(set(point_colors))[i]),
                            markersize=15,
                            lw=0,
                            alpha=1,
                        )
                        for i in range(len(labels))
                    ]

                handles, labels = scatter.legend_elements()

                if points_labels is not None:
                    labels = set(points_labels)

            #             if i % 2  == 0:
            ax.set_ylabel(f"PC{pcs[1]+1} Feature space")
            ax.tick_params(
                axis="y",
                which="major",
                length=3,
                width=0.5,
                labelsize="6",
                direction="inout",
                top=False,
                right=False,
            )
            ax.tick_params(
                axis="x",
                which="major",
                length=3,
                width=0.5,
                labelsize="6",
                direction="inout",
                top=False,
                right=False,
            )

            #             if i % 2  == 1:
            #                 ax.tick_params(axis='y', which='major', right=False, labelleft=False)

            if i < 2:
                ax.tick_params(axis="x", which="major", labelbottom=False)
            if i > 1:
                ax.set_xlabel(f"PC{pcs[0]+1} Feature space")
                ax.tick_params(axis="x", which="major", labelbottom=True, top=True)
            if i > 0:
                ax.tick_params(axis="x", which="major", top=True)

            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            ax.text(
                0.02,
                0.98,
                lab_,
                transform=ax.transAxes,
                fontsize="9",
                va="top",
                fontfamily="roboto",
                fontweight="bold",
            )

        if lito_names:
            legend_elements_ = [
                Patch(
                    facecolor=lito_cmap(lito_names[lito]),
                    edgecolor="none",
                    label=fill(str(lito), 20),
                )
                for lito in list(lito_names.keys())
            ]

            leg_axs = plt.Subplot(fig, legend_gs[0])
            fig.add_subplot(leg_axs)
            leg_axs.axis("off")

            point_leg = leg_axs.legend(
                handles=legend_elements_,
                title="Lithologies",
                loc="upper left",
                fontsize="6",
                bbox_to_anchor=[0, 1],
                numpoints=1,
                ncol=4,
                frameon=False,
            )
            point_leg._legend_box.align = "left"
        if points is not None:
            leg_axs = plt.Subplot(fig, legend_gs[1])
            fig.add_subplot(leg_axs)
            leg_axs.axis("off")
            point_leg = leg_axs.legend(
                handles=legend_elements,
                title="Minerals",
                loc="upper left",
                fontsize="6",
                bbox_to_anchor=[0, 1.08],
                numpoints=1,
                ncol=5,
                frameon=False,
            )
            point_leg._legend_box.align = "left"

        top_bar_cax = plt.Subplot(fig, colorbars_gs[0])
        bot_bar_cax = plt.Subplot(fig, colorbars_gs[6])
        fig.add_subplot(top_bar_cax)
        fig.add_subplot(bot_bar_cax)

        # sorted unique values of cluster labels
        unique_clusters = np.unique(cluster_labels.astype(float))
        unique_clusters.sort()

        # Compute boundaries (assumes ticks are in increasing order)
        boundaries = [unique_clusters[0] - 0.5]
        boundaries += [
            (unique_clusters[i] + unique_clusters[i + 1]) / 2
            for i in range(len(unique_clusters) - 1)
        ]
        boundaries += [unique_clusters[-1] + 0.5]

        # Determine the overall range (based on your boundaries)
        vmin, vmax = boundaries[0], boundaries[-1]

        filtered_colors = [
            plots[0].cmap(tick / plots[0].cmap.N) for tick in unique_clusters
        ]

        norm = mcolors.BoundaryNorm(
            boundaries=boundaries, ncolors=len(unique_clusters), clip=True
        )

        cbar_tick_spacing = 1 / len(unique_clusters)

        cbar_kargs = {
            "ticks": np.arange(
                0 + cbar_tick_spacing / 2,
                1 + cbar_tick_spacing / 2,
                cbar_tick_spacing,
            )
        }

        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LinearSegmentedColormap

        k_cmap_new = LinearSegmentedColormap.from_list(
            "cbar_k", filtered_colors, len(filtered_colors)
        )

        # Create a new ScalarMappable with your filtered cmap and norm
        sm = ScalarMappable(cmap=k_cmap_new)
        sm.set_array([])

        cbar = plt.colorbar(
            sm,
            label="Clusters",
            cax=top_bar_cax,
            drawedges=False,
            orientation="horizontal",
            # norm=norm,
            # cmap=filtered_cmap,
            **cbar_kargs,
        )
        cbar.ax.set_xticklabels(unique_clusters.astype(int))
        cbar.ax.tick_params(
            axis="x", which="major", length=3, width=0.5, labelsize="6", direction="out"
        )
        # cbar.ax.set_xticks(np.arange(19))
        cbar.outline.set_linewidth(0.5)
        cbar.outline.set_color("k")

        cbar = plt.colorbar(
            plots[2],
            label="Cu ppm",
            cax=bot_bar_cax,
            drawedges=False,
            orientation="horizontal",
        )
        cbar.ax.tick_params(
            axis="x", which="major", length=3, width=0.5, labelsize="6", direction="out"
        )
        cbar.outline.set_linewidth(0.5)
        cbar.outline.set_color("k")

        if figname != "":
            plt.savefig(
                figname, dpi=350, facecolor="w", edgecolor="w", bbox_inches="tight"
            )
        plt.show()
