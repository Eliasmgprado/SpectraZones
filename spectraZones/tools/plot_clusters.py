import os

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from sklearn.decomposition import PCA
from textwrap import fill


dirname = os.path.dirname(__file__)

pkl_path = os.path.join(dirname, "../pkl/")

import pickle

with open(os.path.join(pkl_path, "lit_intcolor.pkl"), "rb") as handle:
    lit_intcolor_dict = pickle.load(handle)

with open(os.path.join(pkl_path, "lit_geoclass.pkl"), "rb") as handle:
    lit_geoclass_dict = pickle.load(handle)

with open(os.path.join(pkl_path, "lit_geoclass_eng.pkl"), "rb") as handle:
    lit_geoclass_dict_eng = pickle.load(handle)


def unit_str_format(str_):
    return (
        str_.replace("_PCT", " (%)")
        .replace("_pct", " (%)")
        .replace("_PPM", " (ppm)")
        .replace("_ppm", " (ppm)")
        .replace("_PPB", " (ppb)")
        .replace("_ppb", " (ppb)")
    )


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def plot_dh_curves(
    df_input,
    dh,
    dh_col="HOLEID",
    from_col="FROM",
    to_col="TO",
    lito_cols=["LITO"],
    lito_cols_labels=[],
    data_cols=["Al2O3"],
    data_cols_labels=[],
    color_dict=[],
    eng=False,
    figname="",
    figsize=(19, 22),
    grey_one=False,
    posneg_color=True,
    limit_max=[],
    limit_min=[],
    legend_cols=1,
    fig=None,
    gridSpec=None,
    depth_range=None,
    hide_y_ticks=False,
    show=True,
):
    txt_color = "k"
    #     fontP = FontProperties()
    #     fontP.set_size('xx-small')

    n_cols = len(lito_cols) + len(data_cols)

    if depth_range is None:
        y_lim = [df[to_col].max(), df[from_col].min()]
    else:
        y_lim = depth_range
        print(y_lim)

    plt.style.use("fivethirtyeight")
    with plt.rc_context(
        {
            "text.color": txt_color,
            "axes.titlesize": 6,
            "axes.titlelocation": "left",
            "axes.labelsize": 7,
            "xtick.color": txt_color,
            "ytick.color": txt_color,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.facecolor": "none",
            "figure.dpi": 350,
            "legend.fontsize": 7,
            "figure.figsize": cm2inch(figsize),
            "mathtext.default": "regular",
            "figure.subplot.wspace": 0.3,
        }
    ):

        if fig is None:
            fig = plt.figure()

            fig.tight_layout()
            fig.patch.set_facecolor("none")

            fig.suptitle(dh, fontsize=8)

        dt_axs = []

        if gridSpec is None:

            grid_spec = gridspec.GridSpec(
                1,
                n_cols,
                wspace=0,
                width_ratios=[1] * len(lito_cols) + [5] * len(data_cols),
            )
        else:
            grid_spec = gridspec.GridSpecFromSubplotSpec(
                1,
                n_cols,
                subplot_spec=gridSpec,
                wspace=0,
                width_ratios=[1] * len(lito_cols) + [5] * len(data_cols),
            )

        df = df_input[df_input[dh_col] == dh]

        ## LITO CODE
        for i in range(len(lito_cols)):

            if i >= len(color_dict):

                uniq_litos = df[lito_cols[i]].sort_values().unique()

                if len(uniq_litos) > 10:
                    colors_ = plt.get_cmap("tab20").colors
                else:
                    colors_ = plt.get_cmap("tab10").colors

                color_dict.append({str(k): c for c, k in zip(colors_, uniq_litos)})

            ax = plt.Subplot(fig, grid_spec[i])
            ax.set_facecolor("none")

            for axis in ["right"]:
                ax.spines[axis].set_visible(False)

            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(0.5)
                ax.spines[axis].set_color("k")

            W = 1
            H_sum = df[from_col].min()
            #             trans = transforms.blended_transform_factory(ax.transAxes,ax.transData)

            # SIMPLIFY DATA
            temp = df.shift().bfill()[lito_cols[i]]
            simplified_bfill = temp.infer_objects(copy=False)

            simplify_groups = (
                (df[lito_cols[i]] != simplified_bfill).cumsum().rename("group_simplify")
            )
            simplified_df = (
                df.groupby([lito_cols[i], simplify_groups])[[from_col, to_col]]
                .agg(x=(from_col, "min"), y=(to_col, "max"))
                .reset_index()
                .rename(columns={"x": from_col, "y": to_col})
                .sort_values(by=from_col)
            )
            print(lito_cols[i])
            for From, To, Lito in simplified_df[
                [from_col, to_col, lito_cols[i]]
            ].values:

                H = To - From
                if str(Lito) != "nan":
                    ax.add_patch(
                        Rectangle(
                            xy=(0, From),
                            width=W,
                            height=H,
                            linewidth=1,
                            color=color_dict[i][str(Lito)],
                            fill=True,
                        )
                    )
                H_sum += H

            ax.set_xlim([0, W])

            ax.set_ylim(y_lim)
            ax.set_xticks([0.5])
            if len(lito_cols_labels) == len(lito_cols) and len(lito_cols_labels) > 0:
                ax.set_xticklabels(
                    [lito_cols_labels[i]], rotation=45, ha="right", fontsize=6
                )
            else:
                ax.set_xticklabels([lito_cols[i]], rotation=45, ha="right", fontsize=6)
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            ax.minorticks_on()

            if i > 0 or hide_y_ticks:
                ax.set_yticklabels([])
            else:
                #                 ax.set_title(dh)
                ax.set_ylabel("Depth (m)")

            fig.add_subplot(ax)

        ### DATA CURVE

        for i, j in zip(range(len(lito_cols), n_cols), range(len(data_cols))):

            x_min = None
            x_max = None

            if x_min is None:
                x_min = df[data_cols[j]].min()
            elif x_min > df[data_cols[j]].min():
                x_min = df[data_cols[j]].min()

            if x_max is None:
                x_max = df[data_cols[j]].max()
            elif x_max < df[data_cols[j]].max():
                x_max = df[data_cols[j]].max()

            ax = plt.Subplot(fig, grid_spec[i])
            #             ax = fig.add_subplot(grid_spec[i])
            ax.set_facecolor("none")

            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(0.5)
                ax.spines[axis].set_color("k")

            signal = df[data_cols[j]].astype(float).values

            if posneg_color or grey_one:
                pos_signal = signal.copy()
                neg_signal = signal.copy()

            if grey_one:
                grey_signal = signal.copy()

                pos_signal[pos_signal <= 1] = np.nan
                neg_signal[neg_signal >= -1] = np.nan
                grey_signal[grey_signal < -1] = np.nan
                grey_signal[grey_signal > 1] = np.nan

            elif posneg_color:
                #             print(pos_signal[pos_signal <= 0])
                pos_signal[pos_signal <= 0] = np.nan
                neg_signal[neg_signal > 0] = np.nan

            if posneg_color:
                ax.plot(
                    pos_signal,
                    [(from_ + to_) / 2 for from_, to_ in df[[from_col, to_col]].values],
                    lw=0.7,
                )
                ax.plot(
                    neg_signal,
                    [(from_ + to_) / 2 for from_, to_ in df[[from_col, to_col]].values],
                    lw=0.7,
                    color="r",
                )
            else:
                ax.plot(
                    signal,
                    [(from_ + to_) / 2 for from_, to_ in df[[from_col, to_col]].values],
                    lw=0.7,
                )
            if grey_one:
                ax.plot(
                    grey_signal,
                    [(from_ + to_) / 2 for from_, to_ in df[[from_col, to_col]].values],
                    lw=0.7,
                    color="grey",
                )

            ax.set_ylim(y_lim)
            ax.set_yticklabels([])
            ax.tick_params(axis="x", which="major", length=3, width=0.5)
            ax.minorticks_on()
            ax.yaxis.grid(
                visible=True,
                which="minor",
                color="#c4c4c4",
                linestyle="--",
                linewidth=0.2,
            )
            ax.xaxis.grid(
                visible=True,
                which="major",
                color="#c4c4c4",
                linestyle="-",
                linewidth=0.8,
            )
            ax.xaxis.grid(
                visible=True,
                which="minor",
                color="#c4c4c4",
                linestyle="--",
                linewidth=0.2,
            )

            ax.xaxis.set_major_locator(
                ticker.MaxNLocator(3, steps=[1, 2, 5], min_n_ticks=2)
            )
            ax.xaxis.set_minor_locator(ticker.MaxNLocator(5, steps=[1, 2, 5]))

            if len(data_cols_labels) == len(data_cols) and len(data_cols_labels) > 0:
                #                 ax.set_xlabel(data_cols_labels[j], rotation = 45, ha='right')
                ax.set_xlabel(data_cols_labels[j], rotation=0, ha="center", fontsize=8)
            else:
                #                 ax.set_xlabel(unit_str_format(data_cols[j]), rotation = 45, ha='right')
                ax.title.set_text(unit_str_format(data_cols[j]), fontsize=8)

            delta_ = (x_max - x_min) / 20

            ax_max = x_max + delta_
            ax_min = x_min - delta_

            if len(limit_max) == len(data_cols):
                if limit_max[j] is not None:
                    if limit_max[j][0] is not None:
                        ax_max = max(limit_max[j][0], ax_max)

                    if limit_max[j][1] is not None:
                        ax_max = min(ax_max, limit_max[j][1])

            if len(limit_min) == len(data_cols):
                if limit_min[j] is not None:
                    if limit_min[j][0] is not None:
                        ax_min = max(limit_min[j][0], ax_min)

                    if limit_min[j][1] is not None:
                        ax_min = min(ax_min, limit_min[j][1])

            ax.set_xlim([ax_min, ax_max])

            plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

            fig.add_subplot(ax)

        if gridSpec is None:
            legend_elements = []

            for i, color_d, lito_col in zip(
                range(len(lito_cols)), color_dict, lito_cols
            ):

                if (
                    len(lito_cols_labels) == len(lito_cols)
                    and len(lito_cols_labels) > 0
                ):
                    legend_elements += [
                        Patch(
                            facecolor="none",
                            edgecolor="none",
                            label=lito_cols_labels[i],
                        )
                    ]
                else:
                    legend_elements += [
                        Patch(facecolor="none", edgecolor="none", label=lito_col)
                    ]

                if lito_col == "LIT_INTCOLOR":
                    legend_elements += [
                        Patch(
                            facecolor=color_d[lito],
                            edgecolor="none",
                            label=fill(lito + f" ({lit_intcolor_dict[lito]})", 20),
                        )
                        for lito in color_d
                        if str(lito) != "nan"
                    ]
                elif lito_col == "LIT_GEOCLASS":
                    if eng:
                        dict_ = lit_geoclass_dict_eng
                    else:
                        dict_ = lit_geoclass_dict

                    legend_elements += [
                        Patch(
                            facecolor=color_d[lito],
                            edgecolor="none",
                            label=fill(lito + f" ({dict_[lito]})", 20),
                        )
                        for lito in color_d
                        if str(lito) != "nan"
                    ]
                else:

                    legend_elements += [
                        Patch(
                            facecolor=color_d[lito],
                            edgecolor="none",
                            label=fill(str(lito), 20),
                        )
                        for lito in color_d
                        if str(lito) != "nan"
                    ]

            fig.legend(
                handles=legend_elements,
                bbox_to_anchor=(0.95, 0.98),
                loc="upper left",
                ncol=legend_cols,
            )

            if figname != "":
                plt.savefig(
                    figname, dpi=350, facecolor="w", edgecolor="w", bbox_inches="tight"
                )

            if show:
                plt.show()


def plot_all_dh(
    df_input,
    drill_holes,
    title="",
    dh_col="HOLEID",
    from_col="FROM",
    to_col="TO",
    lito_cols=["LITO"],
    lito_cols_labels=[],
    data_cols=["Al2O3"],
    data_cols_labels=[],
    color_dict=[],
    eng=False,
    figname="",
    figsize=(19, 22),
    grey_one=False,
    posneg_color=True,
    limit_max=[],
    limit_min=[],
    legend_ncols=1,
    idxs_legend_cols=[],
    label_legend_cols=[],
    fig=None,
    dpi=350,
):
    """
    Plot all drill holes data in a single figure.

    Parameters
    ----------
    df_input : pd.DataFrame
        Input DataFrame with the data.
    drill_holes : list
        List of drill holes to plot.
    title : str, optional
        Figure title. The default is "".
    dh_col : str, optional
        Drill hole column. The default is "HOLEID".
    from_col : str, optional
        From column. The default is "FROM".
    to_col : str, optional
        To column. The default is "TO".
    lito_cols : list, optional
        List of lithology columns. The default is ["LITO"].
    lito_cols_labels : list, optional
        List of lithology columns labels. The default is [].
    data_cols : list, optional
        List of data columns. The default is ["Al2O3"].
    data_cols_labels : list, optional
        List of data columns labels. The default is [].
    color_dict : list, optional
        List of color dictionaries, for each litho_col. The default is [].
    eng : bool, optional
        Use English lithology names. The default is False.
    figname : str, optional
        Figure name. The default is "".
    figsize : tuple, optional
        Figure size. The default is (19, 22).
    grey_one : bool, optional
        Use grey color for values between -1 and 1. The default is False.
    posneg_color : bool, optional
        Use different colors for positive and negative values. The default is True.
    limit_max : list, optional
        Maximum limits for each data column. The default is [].
    limit_min : list, optional
        Minimum limits for each data column. The default is [].
    legend_ncols : int, optional
        Number of columns for the legend. The default is 1.
    idxs_legend_cols : list, optional
        Indexes of the columns to show in the legend. The default is [].
    label_legend_cols : list, optional
        Labels for the legend columns. The default is [].
    fig : plt.figure, optional
        Figure instance. The default is None.
    dpi : int, optional
        Dots per inch. The default is 350.
    """

    plt.rc("text", usetex=False)
    txt_color = "k"

    n_cols = len(drill_holes)

    y_lim = [df_input[to_col].max(), df_input[from_col].min()]

    plt.style.use("fivethirtyeight")
    with plt.rc_context(
        {
            "text.color": txt_color,
            "axes.titlesize": 6,
            "axes.titlelocation": "left",
            "axes.labelsize": 7,
            "xtick.color": txt_color,
            "ytick.color": txt_color,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.facecolor": "none",
            "figure.dpi": dpi,
            "legend.fontsize": 7,
            "figure.figsize": cm2inch(figsize),
            "mathtext.default": "regular",
            "figure.subplot.wspace": 0.3,
        }
    ):

        fig = plt.figure(dpi=350)

        fig.tight_layout()
        fig.patch.set_facecolor("none")

        fig.suptitle(title, fontsize=8)

        gs = gridspec.GridSpec(1, n_cols, figure=fig)

        for dh, col in zip(drill_holes, range(n_cols)):
            print(dh)
            plot_dh_curves(
                df_input,
                dh,
                dh_col=dh_col,
                from_col=from_col,
                to_col=to_col,
                lito_cols=lito_cols,
                lito_cols_labels=lito_cols_labels,
                data_cols=data_cols,
                data_cols_labels=data_cols_labels,
                color_dict=color_dict,
                eng=eng,
                figname="",
                figsize=figsize,
                grey_one=grey_one,
                posneg_color=posneg_color,
                limit_max=limit_max,
                limit_min=limit_min,
                legend_cols=1,
                fig=fig,
                gridSpec=gs[col],
                depth_range=y_lim,
                hide_y_ticks=(col > 0),
                show=False,
            )

            ax_title = fig.add_subplot(gs[col])
            ax_title.axis("off")
            ax_title.set_title(dh, fontsize=8, loc="center")

            legend_elements = []

            leg_color_dicts = [color_dict[leg_idx] for leg_idx in idxs_legend_cols]
            leg_list = [len(leg_color_d.keys()) for leg_color_d in leg_color_dicts]
            legend_rows = max(leg_list)

            if len(label_legend_cols) == 0:
                label_legend_cols = lito_cols_labels + ["Cluster"]

            for label, color_d in zip(
                label_legend_cols,
                leg_color_dicts,
            ):

                current_elemets = []

                current_elemets += [
                    Patch(facecolor="none", edgecolor="none", label=label)
                ]
                current_elemets += [
                    Patch(
                        facecolor=color_d[lito],
                        edgecolor="none",
                        label=fill(str(lito), 20),
                    )
                    for lito in color_d
                    if str(lito) != "nan"
                ]

                if len(current_elemets) < legend_rows + 1:
                    diff = legend_rows + 1 - len(current_elemets)
                    current_elemets += [
                        Patch(facecolor="none", edgecolor="none", label="")
                        for i in range(diff)
                    ]
                legend_elements += current_elemets

        leg = fig.legend(
            handles=legend_elements,
            bbox_to_anchor=(0.95, 0.90),
            loc="upper left",
            ncol=legend_ncols,
            frameon=False,
        )
        txts = leg.get_texts()

        # here we create the distinct instance
        txts[0]._fontproperties = txts[1]._fontproperties.copy()
        txts[int(len(txts) / 2)]._fontproperties = txts[1]._fontproperties.copy()
        txts[0].set_weight("bold")
        txts[int(len(txts) / 2)].set_weight("bold")

        if figname != "":
            plt.savefig(
                figname, dpi=dpi, facecolor="w", edgecolor="w", bbox_inches="tight"
            )

        plt.show()
