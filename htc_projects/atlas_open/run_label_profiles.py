# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import itertools
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from htc.cpp import map_label_image
from htc.fonts.set_font import set_font
from htc.settings import settings
from htc.tivita.colorscale import tivita_colorscale
from htc.tivita.DataPath import DataPath
from htc.tivita.hsi import tivita_wavelengths
from htc.utils.helper_functions import median_table
from htc.utils.LabelMapping import LabelMapping
from htc.utils.parallel import p_map
from htc_projects.atlas.settings_atlas import settings_atlas
from htc_projects.atlas_open.profiles import logo_on_fig


def image_data(path: DataPath, label_name: str) -> tuple[list[dict], list[dict]]:
    mapping_path = LabelMapping.from_path(path)
    cube = path.read_cube()

    rows_params = []
    rows_spectra = []

    for annotation_name, labels in path.read_segmentation(annotation_name="all").items():
        # General info about the image
        label_index = mapping_path.name_to_index(label_name)
        label_mask = labels == label_index

        # Parameter values directly as percentage
        sto2 = path.compute_sto2(cube) * 100
        nir = path.compute_nir(cube) * 100
        twi = path.compute_twi(cube) * 100
        ohi = path.compute_ohi(cube) * 100
        tli = path.compute_tli(cube) * 100
        thi = path.compute_thi(cube) * 100

        # For the line plots
        spectra = np.median(cube[label_mask], axis=0)
        with np.errstate(invalid="ignore"):
            absorption = -np.log(spectra)
        div1 = np.gradient(spectra, axis=-1)
        div2 = np.gradient(div1, axis=-1)

        rows_params.append(
            path.image_name_typed()
            | {
                "annotation_name": annotation_name,
                "label_name": label_name,
                "StO2": np.median(sto2.data[label_mask]),
                "NIR": np.median(nir.data[label_mask]),
                "TWI": np.median(twi.data[label_mask]),
                "OHI": np.median(ohi.data[label_mask]),
                "TLI": np.median(tli.data[label_mask]),
                "THI": np.median(thi.data[label_mask]),
            }
        )

        rows_spectra.append(
            path.image_name_typed()
            | {
                "annotation_name": annotation_name,
                "label_name": label_name,
                "absorption": absorption,
                "spectra": spectra,
                "div1": div1,
                "div2": div2,
            }
        )

    return rows_params, rows_spectra


def save_label_page(
    label_name: str, label_number: str, df_params: pd.DataFrame, df_spectra: pd.DataFrame, target_dir: Path
) -> None:
    set_font(12)

    df_params = df_params.query("label_name == @label_name")
    df_spectra = df_spectra.query("label_name == @label_name")
    assert df_params["label_name"].unique() == df_spectra["label_name"].unique()
    assert df_params["label_name"].nunique() == 1

    example_images = {
        "stomach": DataPath.from_image_name("P092#2021_04_27_12_08_58"),
        "small_bowel": DataPath.from_image_name("P086#2021_04_15_12_22_07"),
        "colon": DataPath.from_image_name("P087#2021_04_16_11_51_49"),
        "liver": DataPath.from_image_name("P086#2021_04_15_11_20_00"),
        "gallbladder": DataPath.from_image_name("P087#2021_04_16_09_22_06"),
        "pancreas": DataPath.from_image_name("P088#2021_04_19_13_37_59"),
        "kidney": DataPath.from_image_name("P086#2021_04_15_10_24_06"),
        "spleen": DataPath.from_image_name("P090#2021_04_22_10_45_12"),
        "bladder": DataPath.from_image_name("P087#2021_04_16_12_28_04"),
        "omentum": DataPath.from_image_name("P087#2021_04_16_11_23_37"),
        "lung": DataPath.from_image_name("P086#2021_04_15_18_26_19"),
        "heart": DataPath.from_image_name("P086#2021_04_15_18_09_11"),
        "cartilage": DataPath.from_image_name("P090#2021_04_22_13_55_20"),
        "bone": DataPath.from_image_name("P086#2021_04_15_19_18_22"),
        "skin": DataPath.from_image_name("P089#2021_04_21_14_44_19"),
        "muscle": DataPath.from_image_name("P088#2021_04_19_13_16_10"),
        "peritoneum": DataPath.from_image_name("P088#2021_04_19_11_48_27"),
        "major_vein": DataPath.from_image_name("P093#2021_04_28_16_05_31"),
        "kidney_with_Gerotas_fascia": DataPath.from_image_name("P086#2021_04_15_09_49_53"),
        "bile_fluid": DataPath.from_image_name("P087#2021_04_16_10_36_02"),
    }

    # General info about the image
    path = example_images[label_name]
    mapping_path = LabelMapping.from_path(path)
    label_color = settings_atlas.label_colors[label_name]
    label_index = mapping_path.name_to_index(label_name)
    label_mask = path.read_segmentation("polygon#annotator1") == label_index
    cube = path.read_cube()

    # Parameter values directly as percentage
    sto2 = path.compute_sto2(cube) * 100
    nir = path.compute_nir(cube) * 100
    twi = path.compute_twi(cube) * 100
    ohi = path.compute_ohi(cube) * 100
    tli = path.compute_tli(cube) * 100
    thi = path.compute_thi(cube) * 100

    # Actual plotting
    fig, axes = plt.subplots(4, 3, figsize=(12, 12), constrained_layout=True)
    cmap = tivita_colorscale("matplotlib")
    x = tivita_wavelengths()

    def std_plot(data: np.ndarray, ax) -> None:
        mean = np.median(data, axis=0)
        std = np.std(data, axis=0)

        ax.plot(x, mean, c=label_color)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=label_color)

    # RGB
    ax = axes[0, 0]
    ax.imshow(path.read_rgb_reconstructed())
    mapping_seg = LabelMapping(
        {"background": 0, "label": 1},
        zero_is_invalid=True,
        label_colors={"background": "#ffffff00", "label": label_color},
    )
    mask = map_label_image(label_mask.astype(np.int64), mapping_seg)
    ax.imshow(mask, alpha=0.5, interpolation="none")
    ax.set_title("RGB")

    # Box plot
    ax = axes[0, 1]
    sns.boxplot(
        data=df_params.melt(id_vars=["label_name", "subject_name"]).replace({"StO2": "StO$_2$"}),
        x="variable",
        y="value",
        color=label_color,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "8",
        },
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("index value [a.u.]")
    ax.set_title("Parameter Values")

    # Absorption
    ax = axes[0, 2]
    std_plot(np.stack(df_spectra["absorption"]), ax)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("absorption [a.u.]")
    ax.set_title("Absorbance")

    # Reflection
    ax = axes[1, 0]
    std_plot(np.stack(df_spectra["spectra"]), ax)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("reflectance [a.u.]")
    ax.set_title("Reflectance")

    # 1st derivative
    ax = axes[1, 1]
    std_plot(np.stack(df_spectra["div1"]), ax)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("derivative [a.u.]", labelpad=0)
    ax.set_title("Reflectance – 1$^{\\mathrm{st}}$ Derivative")

    # 2st derivative
    ax = axes[1, 2]
    std_plot(np.stack(df_spectra["div2"]), ax)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("derivative [a.u.]", labelpad=0)
    ax.set_title("Reflectance – 2$^{\\mathrm{nd}}$ Derivative")

    # StO2
    ax = axes[2, 0]
    im = ax.imshow(sto2, cmap=cmap, vmin=0, vmax=100)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    ax.set_facecolor("black")
    ax.set_title("Oxygenation – StO$_2$")

    # NIR
    ax = axes[2, 1]
    im = ax.imshow(nir, cmap=cmap, vmin=0, vmax=100)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    ax.set_facecolor("black")
    ax.set_title("NIR Perfusion Index")

    # THI
    ax = axes[2, 2]
    im = ax.imshow(thi, cmap=cmap, vmin=0, vmax=100)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    ax.set_facecolor("black")
    ax.set_title("Tissue Hemoglobin Index – THI")

    # TWI
    ax = axes[3, 0]
    im = ax.imshow(twi, cmap=cmap, vmin=0, vmax=100)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    ax.set_facecolor("black")
    ax.set_title("Tissue Water Index – TWI")

    # OHI
    ax = axes[3, 1]
    im = ax.imshow(ohi, cmap=cmap, vmin=0, vmax=100)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    ax.set_facecolor("black")
    ax.set_title("Organ Hemoglobin Index – OHI")

    # TLI
    ax = axes[3, 2]
    im = ax.imshow(tli, cmap=cmap, vmin=0, vmax=100)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    ax.set_facecolor("black")
    ax.set_title("Tissue Lipid Index – TLI")

    binary = logo_on_fig(fig, label_name, delta_left=56, delta_top=29)
    plt.close(fig)
    file = target_dir / f"label_profile_{label_number}_{label_name}.pdf"
    file.write_bytes(binary.getbuffer())


def profile_labels(target_dir: Path) -> None:
    df = median_table(dataset_name="HeiPorSPECTRAL", annotation_name="all")

    paths = [DataPath.from_image_name(row["image_name"]) for i, row in df.iterrows()]
    label_names = df["label_name"].values
    label_numbers = [paths[0].dataset_settings["label_ordering"][l] for l in label_names]

    # Collect all the image data we need for aggregating
    res = p_map(image_data, paths, label_names)
    df_params = pd.DataFrame(list(itertools.chain(*[x[0] for x in res])))
    df_spectra = pd.DataFrame(list(itertools.chain(*[x[1] for x in res])))

    # Aggregate according to the hierarchical structure of the data
    param_names = ["StO2", "NIR", "TWI", "OHI", "TLI", "THI"]
    df_params = df_params.groupby(["label_name", "subject_name", "timestamp"], as_index=False)[param_names].mean()
    df_params = df_params.groupby(["label_name", "subject_name"], as_index=False)[param_names].mean()

    spectra_names = ["absorption", "spectra", "div1", "div2"]
    df_spectra = df_spectra.groupby(["label_name", "subject_name", "timestamp"], as_index=False)[spectra_names].mean()
    df_spectra = df_spectra.groupby(["label_name", "subject_name"], as_index=False)[spectra_names].mean()

    p_map(
        partial(save_label_page, df_params=df_params, df_spectra=df_spectra, target_dir=target_dir),
        label_names,
        label_numbers,
    )


if __name__ == "__main__":
    # Similar to the profile pages in the intermediates folder, but globally aggregated for each label (e.g. subject variation instead of pixel variation for the median spectra)
    target_dir = settings.results_dir / "open_data"
    target_dir.mkdir(parents=True, exist_ok=True)
    profile_labels(target_dir)
