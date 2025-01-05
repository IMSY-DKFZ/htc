# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np

from htc.tivita.DataPath import DataPath
from htc.utils.paths import all_masks_paths
from htc_projects.atlas.settings_atlas import settings_atlas
from htc_projects.atlas.tables import median_cam_table, standardized_recordings


def test_median_cam_table() -> None:
    df = median_cam_table()

    assert np.all(df["camera_name"].unique() == settings_atlas.valid_cameras), "Invalid yellow filters"
    assert df["image_name"].nunique() == settings_atlas.n_images
    assert df["subject_name"].nunique() == settings_atlas.n_subjects

    paths = all_masks_paths()
    atlas_images = set()

    for path in paths:
        paper_tags = path.meta("paper_tags")
        if paper_tags is not None and settings_atlas.paper_tag in paper_tags:
            atlas_images.add(path.image_name())

            labels_folder = path.annotated_labels()
            labels_df = df.query("image_name == @path.image_name()")["label_name"].unique()
            assert all(label in labels_folder for label in labels_df)

    assert atlas_images.issuperset(set(df["image_name"].unique())), (
        "The table must include all images of the tissue atlas paper"
    )


def test_standardized_recordings() -> None:
    df = standardized_recordings()
    assert set(df["label_name"]) == set(settings_atlas.labels)
    assert df["image_name"].nunique() == settings_atlas.n_images_standardized
    assert df["subject_name"].nunique() == settings_atlas.n_subjects_standardized

    # Standardized recordings have situs, angle and repetition attributes
    for i, row in df.iterrows():
        path = DataPath.from_image_name(row["image_name"])
        assert path.meta(f"label_meta/{row['label_name']}/situs") is not None
        assert path.meta(f"label_meta/{row['label_name']}/angle") is not None
        assert path.meta(f"label_meta/{row['label_name']}/repetition") is not None
