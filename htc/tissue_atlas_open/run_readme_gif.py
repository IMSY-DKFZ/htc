# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from rich.progress import track

from htc.cpp import map_label_image
from htc.settings import settings
from htc.tissue_atlas.settings_atlas import settings_atlas
from htc.tivita.DataPath import DataPath
from htc.utils.LabelMapping import LabelMapping

if __name__ == "__main__":
    # Select the example image
    path = DataPath.from_image_name("P088#2021_04_19_11_48_27")
    label_name = "peritoneum"

    # RGB with mask
    mapping_path = LabelMapping.from_path(path)
    label_index = mapping_path.name_to_index(label_name)
    label_mask = path.read_segmentation("polygon#annotator1") == label_index
    label_color = settings_atlas.label_colors[label_name]

    mapping_seg = LabelMapping(
        {"background": 0, "label": 1},
        zero_is_invalid=True,
        label_colors={"background": "#ffffff00", "label": label_color},
    )
    mask = map_label_image(label_mask.astype(np.int64), mapping_seg)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    ax.imshow(path.read_rgb_reconstructed())
    ax.imshow(mask, alpha=0.5, interpolation="none")
    ax.set_title(settings_atlas.labels_paper_renaming.get(label_name, label_name))
    ax.set_axis_off()
    rgb_file = settings.results_dir / "open_data" / "example_rgb.png"
    fig.savefig(rgb_file, bbox_inches="tight")
    plt.close(fig)

    # HSI gif
    imgs_channels = []
    cube = path.read_cube()

    # We are using the jet colormap to visualize visible range
    # The start index is a rough match to the color for 500 nm
    jet_start_index = 0.45
    cmap_jet = matplotlib.colormaps.get_cmap("jet")

    for c in track(range(cube.shape[2])):
        # Map wavelengths to colors based on the jet colormap
        wavelength = c * 5 + 500
        if wavelength > 780:
            # We don't have colors anymore in the NIR, so we just use gray instead
            color = (0.4, 0.4, 0.4, 1.0)
        else:
            cmap_index = (wavelength - 500) / (780 - 500)  # [0;1]
            cmap_index = cmap_index * (1.0 - jet_start_index) + jet_start_index  # [0.45;1]
            color = cmap_jet(cmap_index)

        # We map the reflectance values always from black to the current color
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("hsi", [(0, 0, 0, 1.0), color])

        fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
        ax.imshow(cube[:, :, c], cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"{wavelength:0.0f} nm")
        ax.set_axis_off()

        # Convert to PIL image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        imgs_channels.append(Image.open(buf).convert("RGB"))
        plt.close(fig)

    gif_file = settings.results_dir / "open_data" / "example_hsi.gif"
    imgs_channels[0].save(
        fp=gif_file,
        format="GIF",
        append_images=imgs_channels[1:],
        save_all=True,
        duration=200,
        loop=0,
        optimize=True,
    )
