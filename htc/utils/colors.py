# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import colorsys
from pprint import pprint

import numpy as np
from matplotlib.colors import to_hex, to_rgb
from scipy.spatial import distance

from htc.settings import settings
from htc.tivita.DatasetSettings import DatasetSettings
from htc.utils.helper_functions import sort_labels


def generate_distinct_colors(n_colors: int, existing_colors: list[tuple] = None) -> list[tuple]:
    """
    Generates distinct random colors by maximizing the distance between the colors.

    Args:
        n_colors: Number of random colors to choose.
        existing_colors: Existing colors as (r,g,b) @ [0,1] tuples. Useful if additional colors are needed for an existing mapping.

    Returns: Generated colors as rgba tuples.
    """
    np.random.seed(1337)
    if existing_colors is None:
        existing_colors = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
        n_colors -= 2

    for _ in range(n_colors):
        random_colors = []
        n_colors = 1000

        for _ in range(n_colors):
            color = [np.random.rand(1)[0], np.random.randint(30, 80) / 100, np.random.randint(50, 90) / 100]
            color = colorsys.hls_to_rgb(*color)

            random_colors.append(color)

        if not existing_colors:
            selected_color = random_colors[0]
        else:
            # Calculate the distance from each existing colors to all random colors
            distances = distance.cdist(np.array(existing_colors), np.array(random_colors), "euclidean")

            # For each random color, the closest distance to all existing colors is relevant
            distances = np.min(distances, axis=0)
            assert len(distances) == n_colors

            # Take the random color with the highest minimal distance (which is hopefully a distinct color)
            min_index = np.argmax(distances)

            selected_color = random_colors[min_index]

        existing_colors.append(selected_color)

    existing_colors = list(reversed(existing_colors))

    return existing_colors


def unique_labels() -> list[str]:
    settings_semantic = DatasetSettings(settings.data_dirs.semantic)
    settings_masks = DatasetSettings(settings.data_dirs.masks)
    labels = [
        label
        for label, label_index in settings_semantic["label_mapping"].items()
        if label_index <= settings_semantic["last_valid_label_index"]
    ]
    labels += [
        label
        for label, label_index in settings_masks["label_mapping"].items()
        if label_index <= settings_masks["last_valid_label_index"]
    ]
    labels = list(set(labels))

    return labels


def color_organs() -> dict[str, tuple]:
    labels = unique_labels()
    return dict(zip(labels, generate_distinct_colors(len(labels))))


def color_organs_extending() -> dict[str, tuple]:
    labels = unique_labels()
    labels = [label for label in labels if label not in settings.label_colors.keys()]

    if len(labels) == 0:
        return {}
    else:
        existing_colors = list(settings.label_colors.values())
        existing_colors = [to_rgb(c) for c in existing_colors]
        new_colors = generate_distinct_colors(len(labels), existing_colors)

        return dict(zip(labels, new_colors))


if __name__ == "__main__":
    color_mapping = color_organs()
    color_mapping = {l: to_hex(c).upper() for l, c in color_mapping.items()}
    color_mapping = sort_labels(color_mapping)

    pprint(color_mapping, sort_dicts=False)
