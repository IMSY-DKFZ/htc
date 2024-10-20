# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import io
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
from htc.tivita.DatasetSettings import DatasetSettings
from htc.tivita.hsi import tivita_wavelengths
from htc.utils.helper_functions import sort_labels
from htc.utils.import_extra import requires_extra
from htc.utils.LabelMapping import LabelMapping
from htc.utils.parallel import p_map
from htc_projects.atlas.settings_atlas import settings_atlas

try:
    from pypdf import PdfMerger, PdfReader, PdfWriter, Transformation

    _missing_library = ""
except ImportError:
    _missing_library = "pypdf"


@requires_extra(_missing_library)
def logo_on_fig(fig, label_name: str, delta_left: int, delta_top: int) -> io.BytesIO:
    """
    Places a label symbol logo on the PDF figure.

    Args:
        fig: matplotlib figure object. The figure will be converted to a PDF.
        label_name: Name of the label.
        delta_left: Shift of the logo on the figure PDF from the left.
        delta_top: Shift of the logo on the figure PDF from the top.

    Returns: New PDF as Bytes.
    """
    # Add the label logo to the pdf (based on https://pypdf.readthedocs.io/en/latest/user/add-watermark.html)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", bbox_inches="tight")

    dsettings = DatasetSettings(settings.data_dirs["HeiPorSPECTRAL"])
    label_number = dsettings["label_ordering"][label_name]
    logo_reader = PdfReader(
        settings.data_dirs["HeiPorSPECTRAL"] / "extra_label_symbols" / f"Cat_{label_number}_{label_name}.pdf"
    )

    content_reader = PdfReader(buffer)
    content_page = content_reader.pages[0]
    mediabox = content_page.mediabox

    logo_scaling = 0.6
    logo_page = logo_reader.pages[0]
    logo_height = float(logo_page.mediabox.height)

    # Make the logo the same size as the figure so that we can translate the logo to our target position
    logo_page.mediabox = mediabox

    # Position our logo on the figure page
    logo_page.add_transformation(Transformation().scale(logo_scaling))
    logo_page.add_transformation(
        Transformation().translate(
            ty=float(mediabox.height) - logo_height * logo_scaling - delta_top,
            tx=delta_left,
        )
    )

    # Add the logo on the figure PDF
    content_page.merge_page(logo_page)
    content_page.mediabox = mediabox

    buffer = io.BytesIO()
    writer = PdfWriter()
    writer.add_page(content_page)
    writer.write(buffer)

    return buffer


def create_profile_page(path: DataPath, annotation_name: str, label_name: str) -> io.BytesIO:
    set_font(12)

    # General info about the image
    mapping_path = LabelMapping.from_path(path)
    label_number = path.dataset_settings["label_ordering"][label_name]
    label_index = mapping_path.name_to_index(label_name)
    label_color = settings_atlas.label_colors[label_name]
    label_mask = path.read_segmentation(annotation_name) == label_index
    cube = path.read_cube()

    # Parameter values directly as percentage
    sto2 = path.compute_sto2(cube) * 100
    nir = path.compute_nir(cube) * 100
    twi = path.compute_twi(cube) * 100
    ohi = path.compute_ohi(cube) * 100
    tli = path.compute_tli(cube) * 100
    thi = path.compute_thi(cube) * 100
    rows = {
        "StO2": sto2.data[label_mask],
        "NIR": nir.data[label_mask],
        "TWI": twi.data[label_mask],
        "OHI": ohi.data[label_mask],
        "TLI": tli.data[label_mask],
        "THI": thi.data[label_mask],
    }
    df_params = pd.DataFrame(rows).melt()

    # For the line plots
    spectra = cube[label_mask]
    with np.errstate(invalid="ignore"):
        absorption = -np.log(spectra)
    div1 = np.gradient(spectra, axis=-1)
    div2 = np.gradient(div1, axis=-1)

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
        data=df_params.replace({"StO2": "StO$_2$"}),
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
    std_plot(absorption, ax)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("absorption [a.u.]")
    ax.set_title("Absorbance")

    # Reflection
    ax = axes[1, 0]
    std_plot(spectra, ax)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("reflectance [a.u.]")
    ax.set_title("Reflectance")

    # 1st derivative
    ax = axes[1, 1]
    std_plot(div1, ax)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("derivative [a.u.]", labelpad=0)
    ax.set_title("Reflectance – 1$^{\\mathrm{st}}$ Derivative")

    # 2st derivative
    ax = axes[1, 2]
    std_plot(div2, ax)
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

    title = r"$\bf{" + path.image_name().replace("#", r"\#").replace("_", r"\_") + "}$\n"
    situs = path.meta(f"label_meta/{label_name}/situs")
    angle = path.meta(f"label_meta/{label_name}/angle")
    repetition = path.meta(f"label_meta/{label_name}/repetition")
    title += (
        f"label={label_number}_{label_name}, annotation={annotation_name}, situs={situs}, angle={angle}°,"
        f" repetition={repetition}"
    )
    fig.suptitle(title)

    binary = logo_on_fig(fig, label_name, delta_left=4, delta_top=4)
    plt.close(fig)

    return binary


@requires_extra(_missing_library)
def profile_pages(file: Path, label_name: str, annotation_name: str, paths: list[DataPath]) -> None:
    """
    Create a profile PDF file with a page per image for the given label and annotator.

    Args:
        file: Location where the PDF file should be stored.
        label_name: Name of the label which should be included in the PDF.
        annotation_name: Name of the annotator from which the polygon annotations are selected.
        paths: List of image paths which are considered for the profile PDF.
    """
    # Collect some meta information about the paths for the sorting
    rows = []
    for p in paths:
        if label_name in p.annotated_labels(annotation_name):
            assert annotation_name in p.meta("annotation_name")
            label_meta = p.meta(f"label_meta/{label_name}")
            rows.append({
                "annotation_name": annotation_name,
                "situs": label_meta["situs"],
                "angle": label_meta["angle"],
                "repetition": label_meta["repetition"],
                "path": p,
            })

    # Sort the dataframe which determines the order of the pages in the resulting PDF
    df = pd.DataFrame(rows)
    df = sort_labels(
        df, sorting_cols=["annotation_name", "situs", "angle", "repetition"], dataset_name="HeiPorSPECTRAL"
    )

    pages = p_map(
        partial(create_profile_page, label_name=label_name, annotation_name=annotation_name),
        df["path"].tolist(),
        task_name=f"Profiles for {annotation_name}@{label_name}",
    )

    merger = PdfMerger()
    for page in pages:
        merger.append(page)

    with file.open("wb") as f:
        merger.write(f)
