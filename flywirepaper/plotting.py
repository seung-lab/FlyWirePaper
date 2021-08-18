import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpl_patches
import pandas as pd
import seaborn as sns
import warnings
from scipy import stats
import matplotlib.ticker as plticker

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

plt.rc("font", family="sans-serif")
plt.rcParams["font.sans-serif"] = "Helvetica"

warnings.filterwarnings("ignore")

colors = [[100, 100, 100], [213, 94, 0], [0, 114, 178], [0, 0, 0]]

colors = np.array(colors) / 255


def comp_cumplot(
    data_dict,
    norm_dict={},
    xlabel=None,
    xlog=False,
    path=None,
    xrange=None,
    ylabel=None,
    figsize=(6, 6),
    yrange=None,
    xlabel_fontsize=None,
    color_list=None,
    linestyles=None,
    label_order=None,
    categorical=False,
):
    if color_list is None:
        color_list = colors

    if linestyles is None:
        linestyles = ["-"] * len(data_dict)
    elif isinstance(linestyles, str):
        linestyles = [linestyles] * len(data_dict)

    fig = plt.figure(figsize=figsize, facecolor="white")
    plt.tick_params(length=10, width=1.5, labelsize=20)
    plt.axes().spines["bottom"].set_linewidth(1.5)
    plt.axes().spines["left"].set_linewidth(1.5)
    plt.axes().spines["right"].set_linewidth(0)
    plt.axes().spines["top"].set_linewidth(0)

    if label_order is None:
        label_order = list(sorted(data_dict.keys()))

    for i_k, k in enumerate(label_order):
        if k in norm_dict:
            norm = norm_dict[k]
        else:
            norm = len(data_dict[k])

        if categorical:
            marker = "o"
            counts_t = []
            for n in np.unique(data_dict[k]):
                counts_t.append([n, np.sum(np.array(data_dict[k]) == n)])

            counts_t = np.array(counts_t)
            for i in range(1, len(counts_t)):
                counts_t[i, 1] += counts_t[i - 1, 1]
        else:
            marker = None
            sorting = np.argsort(data_dict[k])
            counts_t = np.array(
                [data_dict[k][sorting], np.cumsum(np.ones_like(data_dict[k]))]
            ).T

        plt.plot(
            counts_t[:, 0],
            counts_t[:, 1] / norm,
            marker=marker,
            markersize=7,
            label=k,
            lw=3,
            color=color_list[i_k],
            ls=linestyles[i_k],
        )

    if xlog:
        plt.xscale("log")
        plt.minorticks_off()

    if xlabel_fontsize is None:
        xlabel_fontsize = 24
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)

    if ylabel is None:
        plt.ylabel("Cumulative ratio", fontsize=24)
    else:
        plt.ylabel(ylabel, fontsize=24)

    if len(data_dict) > 1:
        plt.legend(fontsize=18, frameon=False)

    if not xrange is None:
        plt.xlim(*xrange)

    if not yrange is None:
        plt.ylim(*yrange)

    sns.despine(fig=fig, offset=15, trim=False)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.savefig(f"{path[:-4]}.pdf", bbox_inches="tight")

    plt.show()


def scatter_plot(
    x,
    y,
    xlabel=None,
    ylabel=None,
    xlog=False,
    ylog=False,
    xerr=None,
    yerr=None,
    alpha=1.0,
    color="k",
    marker_size=5,
    xrange=None,
    yrange=None,
    path=None,
    kde_bw=None,
    figsize=(6, 6),
    no_points=False,
    plot_diagonal=False,
    despine=True,
    mirror_data=False,
    mirror_color=".5",
    half_and_half=False,
):
    x = np.array(x)
    y = np.array(y)

    fig = plt.figure(figsize=figsize, facecolor="white")
    plt.tick_params(length=10, width=1.5, labelsize=22)

    plt.axes().spines["bottom"].set_linewidth(1.5)
    plt.axes().spines["left"].set_linewidth(1.5)
    plt.axes().spines["right"].set_linewidth(0)
    plt.axes().spines["top"].set_linewidth(0)

    if xlog:
        plt.xscale("log")
        x_s = np.log10(x)
    else:
        x_s = x

    if ylog:
        plt.yscale("log")
        y_s = np.log10(y)
    else:
        y_s = y

    if kde_bw:
        xx, yy, zz = kde2D(x_s, y_s, kde_bw)

        if xlog:
            xx = 10 ** xx
        if ylog:
            yy = 10 ** yy

        if half_and_half:
            m = xx < yy
            zz[m] = 0

        plt.pcolormesh(xx, yy, zz, cmap="gist_heat_r", clim=[0, 0.5])
        color = "k"
        marker_size = 3

    if not no_points:
        if half_and_half:
            t = x.copy()
            x = y
            y = t

            m = x < y
            x = x[m]
            y = y[m]

        plt.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            lw=2,
            markeredgewidth=0,
            color=color,
            fmt="o",
            ms=marker_size,
            alpha=alpha,
        )

        if mirror_data:
            plt.errorbar(
                y,
                x,
                xerr=yerr,
                yerr=xerr,
                lw=2,
                markeredgewidth=0,
                color=mirror_color,
                fmt="o",
                ms=marker_size,
                alpha=alpha,
            )

    if xrange is not None:
        plt.xlim(*xrange)

    xrange = plt.xlim()

    if yrange is not None:
        plt.ylim(*yrange)

    yrange = plt.ylim()

    if plot_diagonal:
        plt.plot(xrange, yrange, c="k", ls="-", lw=1.5)

    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)

    ax = fig.gca()

    if despine:
        sns.despine(fig=fig, offset=15, trim=False)
    plt.tight_layout()

    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.savefig(f"{path[:-4]}.pdf", bbox_inches="tight")
    plt.show()


def scatter_w_binstats_plot(
    x,
    y,
    xlabel=None,
    ylabel=None,
    xlog=False,
    ylog=False,
    xerr=None,
    yerr=None,
    alpha=1.0,
    color="k",
    marker_size=5,
    xrange=None,
    yrange=None,
    path=None,
    figsize=(6, 6),
    plot_diagonal=False,
    despine=True,
):
    x = np.array(x)
    y = np.array(y)

    fig = plt.figure(figsize=figsize, facecolor="white")
    plt.tick_params(length=10, width=1.5, labelsize=26, axis="both", which="major")
    plt.tick_params(length=0, width=0, labelsize=26, axis="both", which="minor")

    plt.axes().spines["bottom"].set_linewidth(1.5)
    plt.axes().spines["left"].set_linewidth(1.5)
    plt.axes().spines["right"].set_linewidth(0)
    plt.axes().spines["top"].set_linewidth(0)

    if xlog:
        plt.xscale("log")
        x_s = np.log10(x)
    else:
        x_s = x

    if ylog:
        plt.yscale("log")
        y_s = np.log10(y)
    else:
        y_s = y

    plt.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        lw=2,
        markeredgewidth=0,
        color=color,
        fmt="o",
        ms=marker_size,
        alpha=alpha,
    )

    if xrange is not None:
        plt.xlim(*xrange)

    xrange = plt.xlim()

    if yrange is not None:
        plt.ylim(*yrange)

    yrange = plt.ylim()

    bin_mean, bin_x_bb, _ = stats.binned_statistic(
        x_s, y_s, statistic="mean", bins=10, range=np.log10(xrange)
    )
    bin_std, _, _ = stats.binned_statistic(
        x_s, y_s, statistic="std", bins=10, range=np.log10(xrange)
    )
    bin_x = np.convolve(bin_x_bb, [0.5, 0.5], mode="valid")

    plt.plot(10 ** bin_x, 10 ** bin_mean, c=[0.7, 0.1, 0.1], lw=3, zorder=100)
    fill_plot = plt.fill_between(
        10 ** bin_x,
        10 ** (bin_mean - bin_std),
        10 ** (bin_mean + bin_std),
        alpha=0.2,
        color=[0.7, 0.1, 0.1],
        lw=0,
        zorder=101,
    )

    if plot_diagonal:
        plt.plot(xrange, yrange, c="k", ls="-", lw=1.5)

    plt.xlabel(xlabel, fontsize=28)
    plt.ylabel(ylabel, fontsize=28)

    ax = fig.gca()
    loc = plticker.LogLocator(
        base=10, numticks=3
    )  # this locator puts ticks at regular intervals
    loc = plticker.LogLocator(
        base=10, numticks=3
    )  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    if despine:
        sns.despine(fig=fig, offset=15, trim=False)
    plt.tight_layout()

    if path is not None:
        plt.savefig(path, dpi=300)
        plt.savefig(f"{path[:-4]}.pdf")
    plt.show()


def hist_plot(
    data,
    xlabel=None,
    logbins=False,
    path=None,
    nbins=25,
    cumulative=False,
    normalize=True,
    shade_color="k",
    edgecolor="k",
    data_range=None,
    rwidth=1.0,
    no_bins=False,
    y_log=False,
    figsize=(6, 6),
    ylabel=None,
    yrange=None,
    step=False,
    orientation="vertical",
    horizontal_lines=[],
    vertical_lines=[],
    n_bins_ticks=5,
):

    fig = plt.figure(figsize=figsize, facecolor="white")
    plt.tick_params(length=10, width=1.5, labelsize=22, which="major")
    plt.tick_params(length=0, width=0, labelsize=22, which="minor")
    plt.axes().spines["bottom"].set_linewidth(1.5)
    plt.axes().spines["left"].set_linewidth(1.5)
    plt.axes().spines["right"].set_linewidth(0)
    plt.axes().spines["top"].set_linewidth(0)

    if data_range is None:
        data_range = [np.min(data), np.max(data)]

    if logbins:
        bins = 10 ** np.linspace(
            np.log10(data_range[0]), np.log10(data_range[1]), nbins + 1
        )
        plt.xscale("log")
    else:
        bins = np.linspace(*data_range, nbins + 1)

    if normalize:
        weights = np.ones(len(data)) / len(data)
    else:
        weights = None

    if not no_bins:
        if step:
            histtype = "step"
            lw = 4
        else:
            histtype = "bar"
            lw = 4

        n, bins, _ = plt.hist(
            data,
            bins=bins,
            color=shade_color,
            weights=weights,
            rwidth=rwidth,
            histtype=histtype,
            edgecolor=edgecolor,
            lw=lw,
            orientation=orientation,
            cumulative=cumulative,
        )

    plt.xlabel(xlabel, fontsize=24)

    if normalize:
        if ylabel is None:
            ylabel = "Frequency"
        ax = fig.gca()
        ax.yaxis.set_major_locator(
            MaxNLocator(integer=False, nbins=n_bins_ticks, min_n_ticks=2)
        )
    else:
        if ylabel is None:
            ylabel = "Count"
        ax = fig.gca()
        ax.yaxis.set_major_locator(
            MaxNLocator(integer=True, nbins=n_bins_ticks, min_n_ticks=2)
        )

    plt.ylabel(ylabel, fontsize=24)

    if not yrange is None:
        plt.ylim(*yrange)
    else:
        plt.ylim(0, plt.ylim()[1])

    if len(vertical_lines) > 0:
        plt.vlines(vertical_lines, 0, plt.ylim()[1], [0.8, 0.4, 0.4], ls="--", lw=3)
    if len(horizontal_lines):
        plt.hlines(0, horizontal_lines, plt.xlim()[1], [0.8, 0.4, 0.4], ls="--", lw=3)

    if y_log:
        plt.yscale("log")

    sns.despine(fig=fig, offset=15, trim=False)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300)
        plt.savefig(f"{path[:-4]}.pdf")

    plt.show()


def comp_hist_plot(
    data_dict,
    xlabel=None,
    xlabel_fontsize=None,
    logbins=False,
    nbins=25,
    path=None,
    figsize=None,
    kde_bins=1000,
    kde_bw=None,
    data_range=None,
    y_log=False,
    dual_axis=False,
    normalize=True,
    color_list=None,
    show_legend=True,
    tick_label_size=None,
    yrange=None,
    label_order=None,
):
    if color_list is None:
        color_list = colors

    if figsize is None:
        if dual_axis:
            assert len(data_dict) == 2
            figsize = (7, 6)
        else:
            figsize = (6, 6)

    fig = plt.figure(figsize=figsize, facecolor="white")
    ax1 = plt.axes()

    plt.tick_params(length=10, width=1.5, labelsize=22)
    plt.axes().spines["bottom"].set_linewidth(2)
    plt.axes().spines["left"].set_linewidth(1.5)
    if dual_axis:
        plt.axes().spines["right"].set_linewidth(1.5)
    else:
        plt.axes().spines["right"].set_linewidth(0)
    plt.axes().spines["top"].set_linewidth(0)

    data = []
    for k in data_dict:
        data.extend(data_dict[k])

    if data_range is None:
        data_range = [np.min(data), np.max(data)]

    if logbins:
        bins = 10 ** np.linspace(
            np.log10(data_range[0]), np.log10(data_range[1]), nbins
        )
        plt.xscale("log")
        xs = np.linspace(np.log10(data_range[0]), np.log10(data_range[1]), kde_bins)
    else:
        bins = np.linspace(*data_range, nbins + 1)
        xs = np.linspace(np.min(data), np.max(data), kde_bins)

    if label_order is None:
        label_order = list(sorted(data_dict.keys()))

    for i_k, k in enumerate(label_order):

        if normalize:
            weights = np.ones(len(data_dict[k])) / len(data_dict[k])
        else:
            weights = None

        if dual_axis:
            if i_k == 0:
                ax = ax1
                font_color = "k"
            else:
                ax = ax1.twinx()
                ax.tick_params(length=8, width=1.5, labelsize=16)
                font_color = color_list[i_k]

            ax.hist(
                data_dict[k],
                bins=bins,
                density=None,
                histtype="step",
                lw=4,
                weights=weights,
                color=color_list[i_k],
                alpha=1,
            )

            ax.set_ylabel(f"{k} count", fontsize=24, color=font_color)
        else:
            ax1.hist(
                data_dict[k],
                bins=bins,
                label=str(k),
                density=None,
                histtype="step",
                lw=4,
                weights=weights,
                color=color_list[i_k],
                alpha=1,
            )

    if show_legend:
        handles, labels = ax1.get_legend_handles_labels()
        new_handles = [Line2D([0], [0], c=h.get_edgecolor()) for h in handles]
        plt.legend(
            handles=new_handles, frameon=False, fontsize=18, loc="best", labels=labels
        )

    ax1.set_xlabel(xlabel, fontsize=24)

    if not dual_axis:
        if normalize:
            ylabel = "Frequency"
        else:
            ylabel = "Count"

        plt.ylabel(ylabel, fontsize=24)

    if normalize:
        ax1.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=5, min_n_ticks=2))
    else:
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5, min_n_ticks=2))

    if not logbins:
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=2))

    if y_log:
        plt.yscale("log")

    if not yrange is None:
        plt.ylim(*yrange)
    else:
        plt.ylim(0, plt.ylim()[1])

    if tick_label_size:
        plt.xticks(fontsize=tick_label_size)
        plt.yticks(fontsize=tick_label_size)

    sns.despine(fig=fig, offset=15, trim=False)
    plt.tight_layout()

    if path is not None:
        plt.savefig(path, dpi=300)
        plt.savefig(f"{path[:-4]}.pdf")

    plt.show()


def box_plot(
    data_dict,
    figsize=(6, 6),
    color_list=None,
    xlabel=None,
    ylabel=None,
    path=None,
    yrange=None,
):
    if color_list is None:
        color_list = colors
    x = []
    y = []

    for k in data_dict:
        x.extend([k] * len(data_dict[k]))
        y.extend(data_dict[k])

    fig = plt.figure(figsize=figsize, facecolor="white")
    plt.tick_params(length=10, width=1.5, labelsize=22)

    plt.axes().spines["bottom"].set_linewidth(1.5)
    plt.axes().spines["left"].set_linewidth(1.5)
    plt.axes().spines["right"].set_linewidth(0)
    plt.axes().spines["top"].set_linewidth(0)

    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=24)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=24)

    data = {"x": x, "y": y}
    ax = sns.boxplot(
        x="x",
        y="y",
        data=data,
        showfliers=False,
        color="w",
        linewidth=2.5,
        saturation=1,
    )

    for i, artist in enumerate(ax.artists):
        col = color_list[i]
        artist.set_edgecolor("k")
        artist.set_facecolor(col)

        for j in range(i * 6, i * 6 + 6):
            try:
                line = ax.lines[j]
                line.set_color("k")
                line.set_mfc("k")
                line.set_mec("k")
            except:
                pass

    ax = fig.gca()
    if yrange is not None:
        ax.set_ylim(*yrange)

    plt.tight_layout()

    sns.despine(fig=fig, offset=15, trim=False)

    plt.tight_layout()

    if path is not None:
        plt.savefig(path, dpi=300)
        plt.savefig(f"{path[:-4]}.pdf")

    plt.show()


def bar_plot(
    x,
    y,
    xlabel=None,
    ylabel=None,
    ylog=False,
    figsize=(6, 6),
    xlabel_rotation=0,
    yerr=None,
    alpha=1.0,
    color=".2",
    yrange=None,
    tick_label_size=None,
    path=None,
):
    y = np.array(y)

    fig = plt.figure(figsize=figsize, facecolor="white")
    plt.tick_params(length=10, width=1.5, labelsize=20)

    plt.axes().spines["bottom"].set_linewidth(1.5)
    plt.axes().spines["left"].set_linewidth(1.5)
    plt.axes().spines["right"].set_linewidth(0)
    plt.axes().spines["top"].set_linewidth(0)

    if ylog:
        plt.yscale("log")

    if x is None:
        x = np.arange(len(y))
        no_x_labels = True
    else:
        no_x_labels = False

    plt.bar(np.arange(len(x)), y, yerr=yerr, color=color, alpha=alpha)
    if yrange is not None:
        plt.ylim(*yrange)

    yrange = plt.ylim()

    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)

    sns.despine(fig=fig, offset=15, trim=False)
    if no_x_labels:
        plt.axes().get_xaxis().set_ticks([])
    else:
        plt.axes().get_xaxis().set_ticks(np.arange(len(x)))
        plt.axes().set_xticklabels(
            x, rotation=xlabel_rotation, ha="center", fontsize=20
        )

    if tick_label_size:
        plt.xticks(fontsize=tick_label_size)
        plt.yticks(fontsize=tick_label_size)

    plt.tight_layout()

    if path is not None:
        plt.savefig(path, dpi=300)
        plt.savefig(f"{path[:-4]}.pdf")
    plt.show()


def comp_scatter_plot(
    data_dict,
    xlabel=None,
    ylabel=None,
    xlog=False,
    ylog=False,
    xerr_dict=None,
    fmt="o",
    figsize=(6, 6),
    marker_size=6,
    yerr_dict=None,
    alpha=1.0,
    xrange=None,
    yrange=None,
    color_list=None,
    show_legend=True,
    path=None,
):
    if color_list is None:
        color_list = colors

    fig = plt.figure(figsize=figsize, facecolor="white")
    plt.tick_params(length=10, width=1.5, labelsize=20)
    plt.axes().spines["bottom"].set_linewidth(1.5)
    plt.axes().spines["left"].set_linewidth(1.5)
    plt.axes().spines["right"].set_linewidth(0)
    plt.axes().spines["top"].set_linewidth(0)

    for i_k, k in enumerate(data_dict):
        if xerr_dict is not None and k in xerr_dict:
            xerr = xerr_dict[k]
        else:
            xerr = None

        if yerr_dict is not None and k in yerr_dict:
            yerr = yerr_dict[k]
        else:
            yerr = None

        data = data_dict[k]
        # data_y.extend(data_dict[k])
        # data_xerr.extend(data_dict[k])
        # data_y_err.extend(data_dict[k])
        # data_colors.extend([colors[i_k]] * len(data_dict[k]))

        plt.errorbar(
            data[:, 0],
            data[:, 1],
            markersize=marker_size,
            xerr=xerr,
            yerr=yerr,
            lw=2,
            fmt=fmt,
            alpha=alpha,
            label=k,
            color=color_list[i_k],
        )

    if xrange is not None:
        plt.xlim(*xrange)

    if yrange is not None:
        plt.ylim(*yrange)

    if xlog:
        plt.xscale("log")
    if ylog:
        plt.yscale("log")

    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)

    if show_legend:
        plt.legend(frameon=False, fontsize=18)

    sns.despine(fig=fig, offset=15, trim=False)
    plt.tight_layout()

    if path is not None:
        plt.savefig(path, dpi=300)
        plt.savefig(f"{path[:-4]}.pdf")
    plt.show()


def swarmplot(
    df,
    x_key,
    y_key,
    hue_key=None,
    xlabel=None,
    ylog=False,
    path=None,
    ylabel=None,
    figsize=(6, 6),
    yrange=None,
    color_list=None,
    show_legend=True,
):
    if color_list is None:
        color_list = colors

    fig = plt.figure(figsize=figsize, facecolor="white")
    plt.tick_params(length=10, width=1.5, labelsize=20)
    plt.axes().spines["bottom"].set_linewidth(1.5)
    plt.axes().spines["left"].set_linewidth(1.5)
    plt.axes().spines["right"].set_linewidth(0)
    plt.axes().spines["top"].set_linewidth(0)

    g = sns.swarmplot(
        data=df, x=x_key, y=y_key, hue=hue_key, size=10, palette=color_list
    )
    g.legend_.remove()

    if ylog:
        plt.yscale("log")
        plt.minorticks_off()

    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)

    if show_legend:
        plt.legend(fontsize=18, frameon=False)

    if not yrange is None:
        plt.ylim(*yrange)

    sns.despine(fig=fig, offset=15, trim=False)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300)
        plt.savefig(f"{path[:-4]}.pdf")

    plt.show()


def plot_matrix(
    data,
    cell_types_x=None,
    cell_types_y=None,
    combined_cell_types_x=None,
    combined_cell_types_y=None,
    is_symmetric=True,
    color_dict={},
    figsize=(16, 14),
    cb_label="# synapses",
    show_legend=True,
    cmap="Greys",
    min_n_syn=0,
    max_syn=20,
    path=None,
):
    def get_name_bounds(names):
        starts = [0]
        u_names = [names[0]]
        temp_name = names[0]
        for i_name in range(1, len(names)):
            if temp_name == names[i_name]:
                continue
            else:
                starts.append(i_name)
                u_names.append(names[i_name])
                temp_name = names[i_name]

        starts.append(len(names))

        starts = np.array(starts)

        return u_names, starts

    fig = plt.figure(figsize=figsize, facecolor="white")
    plt.tick_params(length=0, width=0, labelsize=0)
    ax = plt.gca()

    plt.axes().spines["bottom"].set_linewidth(0)
    plt.axes().spines["left"].set_linewidth(0)
    plt.axes().spines["right"].set_linewidth(0)
    plt.axes().spines["top"].set_linewidth(0)

    # Core plot
    cmap_d = plt.cm.get_cmap(cmap, max_syn + 1 - min_n_syn)
    im = plt.imshow(
        data, cmap=cmap_d, vmax=max_syn + 0.5, aspect="auto", vmin=min_n_syn - 0.5
    )

    # Divider lines
    if cell_types_x is not None:
        coarse_lines_x = get_name_bounds([ct.split(",")[0] for ct in cell_types_x])
        fine_lines_x = get_name_bounds(cell_types_x)

        if is_symmetric:
            coarse_lines_y = coarse_lines_x
            fine_lines_y = fine_lines_x
        else:
            coarse_lines_y = get_name_bounds([ct.split(",")[0] for ct in cell_types_y])
            fine_lines_y = get_name_bounds(cell_types_y)

        plt.hlines(
            y=fine_lines_y[1][1:-1] - 0.5,
            xmin=-3,
            xmax=data.shape[1] - 0.5,
            color=".5",
            lw=1,
        )
        plt.vlines(
            x=fine_lines_x[1][1:-1] - 0.5,
            ymin=-0.5,
            ymax=data.shape[0] + 2,
            color=".5",
            lw=1,
        )

        plt.hlines(
            y=coarse_lines_y[1][1:-1] - 0.5,
            xmin=-3,
            xmax=data.shape[1] - 0.5,
            color="k",
            lw=2,
        )
        plt.vlines(
            x=coarse_lines_x[1][1:-1] - 0.5,
            ymin=-0.5,
            ymax=data.shape[0] + 2,
            color="k",
            lw=2,
        )

    # Cell type color blocks
    if combined_cell_types_x is not None:
        ct_blocks_x = get_name_bounds(
            [ct.split(",")[0] for ct in combined_cell_types_x]
        )

        if is_symmetric:
            ct_blocks_y = ct_blocks_x
        else:
            ct_blocks_y = get_name_bounds(
                [ct.split(",")[0] for ct in combined_cell_types_y]
            )

        for i_start in range(len(ct_blocks_x[0])):
            name_sub = ct_blocks_x[0][i_start]
            start_sub = ct_blocks_x[1][i_start]
            end_sub = ct_blocks_x[1][i_start + 1]

            rect = mpl_patches.Rectangle(
                (start_sub - 0.5, data.shape[0]),
                end_sub - start_sub,
                2,
                linewidth=0,
                facecolor=color_dict[name_sub],
                edgecolor=None,
            )
            ax.add_patch(rect)

        for i_start in range(len(ct_blocks_y[0])):
            name_sub = ct_blocks_y[0][i_start]
            start_sub = ct_blocks_y[1][i_start]
            end_sub = ct_blocks_y[1][i_start + 1]

            rect = mpl_patches.Rectangle(
                (-3, start_sub - 0.5),
                2,
                end_sub - start_sub,
                linewidth=0,
                facecolor=color_dict[name_sub],
                edgecolor=None,
            )
            ax.add_patch(rect)

    fig.tight_layout()

    # Colorbar
    if show_legend:
        cbar = plt.colorbar(im, shrink=0.35, orientation="vertical")
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=22, length=0)
        cbar.ax.set_ylabel(cb_label, fontsize=24)

        yticks = np.array(cbar.ax.get_yticks(), dtype=np.int).astype(np.str)

        if int(yticks[-1]) < np.max(np.array(data)):
            yticks[-1] = f"{yticks[-1]}-{np.max(np.array(data))}"
        if 0 < int(yticks[0]):
            yticks[0] = f"0-{yticks[0]}"
        cbar.ax.set_yticklabels(yticks, fontsize=22)

    if path is not None:
        plt.savefig(path, dpi=300)
        plt.savefig(f"{path.replace('.png', '.pdf')}")
    plt.show()
