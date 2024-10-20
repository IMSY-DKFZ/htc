# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib import ticker
from matplotlib.lines import Line2D

from htc.evaluation.ranking import BootstrapRanking
from htc_projects.context.settings_context import settings_context


def ranking_figure(ranking: BootstrapRanking, task_mapping: dict[str, str]):
    df_counts = ranking.counts
    df_statistics = ranking.statistics

    # ordering of x labels according to mean ranking in task:
    n_cols = 2
    n_rows = 3
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(19, 11), sharey=True, constrained_layout=True)

    alg_short = {
        "organ_transplantation": "OT",
        "random_erasing": "RE",
        "cut_mix": "CM",
        "hide_and_seek": "HS",
        "jigsaw": "JI",
        "elastic": "EL",
        "baseline": "BA",
    }

    scenario_mapping = {
        "isolation_0": "(I)",
        "isolation_cloth": "(I)",
        "masks_isolation": "(I)",
        "removal_0": "(II)",
        "removal_cloth": "(II)",
        "glove": "(III)",
    }

    for t, task in enumerate(task_mapping.keys()):
        row = t // n_cols
        col = t % n_cols

        df_task = df_counts.query("task == @task")
        df_statistics_task = df_statistics.query("task == @task")

        # Scatter plot
        alg_order = list(df_statistics_task.algorithm.values)  # determine ordering
        alg_positions = list(range(len(alg_order)))
        df_task = df_task.sort_values(by="algorithm", key=lambda column: column.map(lambda e: alg_order.index(e)))
        sns.scatterplot(
            df_task,
            x="algorithm",
            y="rank",
            hue="algorithm",
            palette=settings_context.augmentation_colors,
            size="count",
            sizes=(1, df_task["count"].max()),
            ax=ax[row, col],
        )

        # Confidence intervals, median and mean rank
        for _, series in df_statistics_task.iterrows():
            ax[row, col].vlines(
                alg_order.index(series["algorithm"]), series["min_CI"], series["max_CI"], colors="black", alpha=0.5
            )
            ax[row, col].scatter(
                alg_order.index(series["algorithm"]), series["median_rank"], marker="x", s=800, color="black"
            )
            ax[row, col].scatter(
                alg_order.index(series["algorithm"]), series["mean_rank"], marker="D", s=100, color="gray", alpha=0.7
            )

        # Formatting
        ax[row, col].xaxis.set_major_locator(ticker.FixedLocator(alg_positions))
        ax[row, col].set_xticklabels([alg_short[a] for a in alg_order], rotation=90)
        ax[row, col].get_legend().set_visible(False)
        ax[row, col].set_yticks(list(range(1, len(alg_order) + 1)))
        ax[row, col].set_ylim([0.4, len(alg_order) + 0.6])
        ax[row, col].set_title(f"{task_mapping[task]} {scenario_mapping[task]}")
        for i, a in enumerate(alg_order):
            ax[row, col].get_xticklabels()[i].set_color(settings_context.augmentation_colors[a])

        if row < n_rows - 1:
            ax[row, col].xaxis.label.set_visible(False)

    # Generating the legend:
    handles, labels = ax[1, 0].get_legend_handles_labels()

    new_handles = []
    new_labels = []
    for l, h in zip(labels, handles, strict=True):
        try:
            new_labels.append(f"{int(l) / 10:.0f} %")
            new_handles.append(h)
        except ValueError:
            pass

    plt.figlegend(
        handles=new_handles,
        labels=new_labels,
        loc="upper center",
        borderaxespad=0.7,
        ncol=5,
        labelspacing=0.0,
        prop={"size": 20},
        bbox_to_anchor=(0.76, 1.065),
    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="x",
            color="black",
            label="median rank",
            markerfacecolor="black",
            markersize=12,
            linewidth=0,
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="gray",
            label="mean rank",
            markerfacecolor="gray",
            markersize=12,
            linewidth=0,
            alpha=0.7,
        ),
        Line2D(
            [0],
            [0],
            marker="|",
            color="black",
            linestyle="None",
            label="95 % confidence interval",
            markerfacecolor="gray",
            markersize=12,
            linewidth=12,
            alpha=0.5,
        ),
    ]

    plt.figlegend(
        handles=legend_elements,
        loc="upper center",
        borderaxespad=0.7,
        ncol=3,
        labelspacing=0.0,
        prop={"size": 20},
        bbox_to_anchor=(0.27, 1.065),
    )

    return fig


def ranking_legend(algorithms: list[str], mean_ranks: list[float], images: dict[str, torch.Tensor]):
    assert len(algorithms) == len(mean_ranks) == len(images)

    fig, ax = plt.subplots(ncols=len(images), figsize=(16, 8))
    for i, (name, rank) in enumerate(zip(algorithms, mean_ranks, strict=True)):
        ax[i].imshow(images[name])
        ax[i].axis("off")

        bbox = ax[i].get_tightbbox(fig.canvas.get_renderer())
        x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
        fig.add_artist(
            plt.Rectangle(
                (x0, y0), width, height, edgecolor=settings_context.augmentation_colors[name], linewidth=4, fill=False
            )
        )

        ax[i].set_title(f"{rank:0.1f}", fontsize=22)

    return fig
