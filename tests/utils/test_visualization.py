# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import difflib
import time
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.visualization import compress_html, create_overview_document


def test_compress_html(tmp_path: Path) -> None:
    def compress_fig(fig: go.Figure, base_dir: Path) -> tuple[Path, Path, Path]:
        file_original = base_dir / "original.html"
        fig.write_html(file_original, include_plotlyjs="cdn", div_id="fig")
        assert file_original.exists()

        file_fig = base_dir / "fig.html"
        compress_html(file_fig, fig)
        assert file_fig.exists()

        file_html = base_dir / "html.html"
        compress_html(file_html, fig.to_html(include_plotlyjs="cdn", div_id="fig"))
        assert file_html.exists()

        return file_original, file_fig, file_html

    # Small file, default Plotly html is used
    base_dir = tmp_path / "small"
    base_dir.mkdir(parents=True, exist_ok=True)
    fig = px.scatter(x=[0, 1, 2], y=[0, 1, 4])
    file_original, file_fig, file_html = compress_fig(fig, base_dir)
    assert file_fig.stat().st_size == file_original.stat().st_size
    assert file_html.stat().st_size == file_original.stat().st_size

    # Large file, compressed file is used
    base_dir = tmp_path / "large"
    base_dir.mkdir(parents=True, exist_ok=True)
    img = np.arange(150**2).reshape((150, 150))
    fig = px.imshow(img)
    file_original, file_fig, file_html = compress_fig(fig, base_dir)
    assert file_fig.stat().st_size < file_original.stat().st_size
    assert file_html.stat().st_size < file_original.stat().st_size

    # Output should be reproducible
    _, file_fig, file_html = compress_fig(fig, base_dir)
    with file_fig.open() as f1, file_html.open() as f2:
        content_fig = f1.read()
        content_html = f2.read()

    # Trigger different modification time (which should not influence the output)
    time.sleep(1)
    _, file_fig, file_html = compress_fig(fig, base_dir)
    with file_fig.open() as f1, file_html.open() as f2:
        content_fig2 = f1.read()
        content_html2 = f2.read()

    assert content_fig == content_fig2
    assert content_html == content_html2

    html = compress_html(file=None, fig_or_html=fig.to_html(include_plotlyjs="cdn", div_id="fig"))
    assert html == content_html2


def test_create_overview_document(tmp_path: Path) -> None:
    def create_navigation_link(label_name: str, label_order: str, image_path: DataPath) -> str:
        return f"{quote_plus(image_path.image_name())}.html"

    navigation_paths = list(DataPath.iterate(settings.data_dirs.semantic))
    navigation_paths += list(DataPath.iterate(settings.data_dirs.semantic / "context_experiments"))
    path = DataPath.from_image_name("P043#2019_12_20_12_38_35")

    compress_html(
        tmp_path / f"{path.image_name()}.html",
        create_overview_document(
            path, navigation_paths=navigation_paths, navigation_link_callback=create_navigation_link
        ),
    )

    # Check whether the new segmentation figure is similar to the existing one in the data repo
    with (
        (tmp_path / f"{path.image_name()}.html").open() as f_new,
        (path.intermediates_dir / "view_all" / f"{path.image_name()}.html").open() as f_old,
    ):
        content_new = f_new.read()
        content_old = f_old.read()

    diff_ratio = difflib.SequenceMatcher(None, content_new, content_old).quick_ratio()
    assert diff_ratio > 0.99
