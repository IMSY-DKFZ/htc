# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
import re
from collections.abc import Iterator

import jsonschema
import numpy as np
import pytest

from htc.models.image.DatasetImage import DatasetImage
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.metadata import generate_metadata_table
from htc.utils.Config import Config
from htc.utils.helper_functions import median_table
from htc.utils.LabelMapping import LabelMapping
from htc.utils.sqldf import sqldf


# General tests for some of our structured datasets
@pytest.mark.slow
class TestDataset:
    @pytest.fixture(scope="class")
    def paths_atlas(self) -> Iterator[list[DataPath]]:
        # This may take a few seconds, so we only want to run it once per test session
        atlas_dir = (
            settings.datasets.network_data
            / "2020_07_23_hyperspectral_MIC_organ_database/data/Catalogization_tissue_atlas"
        )
        yield list(DataPath.iterate(atlas_dir))

    def test_label_mapping(self) -> None:
        path = DataPath.from_image_name("P043#2019_12_20_12_38_35")
        label_mapping = {"blue_cloth": 10, "heart": 20, "lung": 30}

        dataset = DatasetImage([path], train=False, config=Config({"label_mapping": label_mapping}))
        seg = next(iter(dataset))["labels"]
        assert all(np.unique(seg) == list(label_mapping.values()))

        original_mapping = path.dataset_settings["label_mapping"]
        original_seg = path.read_segmentation()

        for name in label_mapping.keys():
            assert all(seg[original_seg == original_mapping[name]] == label_mapping[name])

    def test_labels_available(self) -> None:
        paths = list(DataPath.iterate(settings.data_dirs.semantic))
        n_labels = [len(p.annotated_labels()) for p in paths]

        assert all(l > 0 for l in n_labels)

    def test_unique_timestamp(self) -> None:
        paths_seg = list(DataPath.iterate(settings.data_dirs.semantic))
        paths_masks = list(DataPath.iterate(settings.data_dirs.masks))
        timestamps_seg = [p.timestamp for p in paths_seg]
        timestamps_masks = [p.timestamp for p in paths_masks]

        assert sorted(set(timestamps_seg)) == sorted(timestamps_seg)
        assert sorted(set(timestamps_masks)) == sorted(timestamps_masks)

        assert len(set(timestamps_seg).intersection(set(timestamps_masks))) == 0

        # Check that every timestamp has the correct format
        for timestamp in timestamps_seg + timestamps_masks:
            match = re.search(r"^\d\d\d\d(?:_\d\d){5}$", timestamp)
            assert match is not None, timestamp

    def test_overlap(self) -> None:
        paths_seg = list(DataPath.iterate(settings.data_dirs.semantic))
        paths_masks = list(DataPath.iterate(settings.data_dirs.masks))
        paths_overlap = list(DataPath.iterate(settings.data_dirs.masks / "overlap"))

        # Overlap images must be in the semantic but not in the masks folder
        for path in paths_overlap:
            assert len([p for p in paths_seg if p.timestamp == path.timestamp]) == 1
            assert len([p for p in paths_masks if p.timestamp == path.timestamp]) == 0

        # Semantic images must not be in the masks folder
        for path in paths_seg:
            assert len([p for p in paths_masks if p.timestamp == path.timestamp]) == 0

        # Masks images must not be in the semantic folder
        for path in paths_masks:
            assert len([p for p in paths_seg if p.timestamp == path.timestamp]) == 0

    @pytest.mark.parametrize(
        "dataset_name", ["2021_02_05_Tivita_multiorgan_semantic", "2021_02_05_Tivita_multiorgan_masks"]
    )
    def test_subject_name_match(self, dataset_name: str, paths_atlas: list[DataPath]) -> None:
        # We want to make sure that every image in our dataset is assigned to the same pig as in the paths_atlas
        paths = DataPath.iterate(settings.data_dirs[dataset_name])
        n_match = 0
        n_no_match = 0

        for path in paths:
            match_single = [p for p in paths_atlas if p.timestamp == path.timestamp]
            if len(match_single) >= 1:
                n_match += 1
                assert len({m.timestamp for m in match_single}) == 1, "matched exp ids must be unique"
                match_single = match_single[0]

                assert path.subject_name == match_single.subject_name, (
                    f"The path {path} has a different subject_name than the path {match_single}"
                )
            else:
                n_no_match += 1
                assert dataset_name == "2021_02_05_Tivita_multiorgan_semantic", (
                    f"Could not find a match for the path {path} in the tissue atlas but the path is from the masks"
                    " dataset where every file must also be in the tissue atlas"
                )

        assert n_match > n_no_match

    def test_unique_subject_name(self, paths_atlas: list[DataPath]) -> None:
        # Test whether the mapping P002_OP002_2018_08_06_Experiment1 --> P002 is unique
        assert len({p.subject_name for p in paths_atlas}) == len({p.subject_folder for p in paths_atlas})

        id_mapping = {}
        for p in paths_atlas:
            if p.subject_name in id_mapping:
                assert id_mapping[p.subject_name] == p.subject_folder
            else:
                id_mapping[p.subject_name] = p.subject_folder

    @pytest.mark.parametrize(
        "dataset_name",
        [
            "2021_02_05_Tivita_multiorgan_masks",
            "2021_07_26_Tivita_multiorgan_human",
            "2023_12_07_Tivita_multiorgan_rat",
        ],
    )
    def test_annotation_meta(self, dataset_name: str) -> None:
        data_dir = settings.data_dirs[dataset_name]
        paths = list(DataPath.iterate(data_dir))
        if (overlap_path := data_dir / "overlap").exists():
            paths += list(DataPath.iterate(overlap_path))

        # Load the default schema
        with (data_dir / "meta.schema").open() as f:
            schema = json.load(f)

        for path in paths:
            assert path.annotation_meta_path().exists()
            annotation_meta = path.read_annotation_meta()
            assert annotation_meta is not None

            # Make sure the annotation_meta has the correct file format
            try:
                jsonschema.validate(instance=annotation_meta, schema=schema)
            except jsonschema.exceptions.ValidationError as e:
                raise AssertionError(
                    f"The annotation_meta file {path.annotation_meta_path()} has an invalid format"
                ) from e

            # Check that the camera_name has the correct prefix
            meta = path.read_camera_meta()
            if meta is not None and "Camera_CamID" in meta:
                assert path.meta("camera_name").startswith(meta["Camera_CamID"]), (
                    "The name of the yellow filter must start with the camera_name"
                )
            else:
                assert path.meta("camera_name").startswith("unknown")

            # Check that the image_labels is as superset of the annotated labels
            labels = path.annotated_labels()
            assert set(labels).issubset(set(annotation_meta["image_labels"])), (
                f"Every label mask file must also occur in the meta labels: {path}"
            )

    def test_camera_name(self) -> None:
        path_wrong = DataPath.from_image_name(
            "P084#2021_03_21_21_14_20"
        )  # The switch of the yellow filter was between the recordings of P084 and P085
        path_correct = DataPath.from_image_name("P085#2021_04_10_10_47_08")

        assert "wrong" in path_wrong.meta("camera_name")
        assert "correct" in path_correct.meta("camera_name")

    @pytest.mark.parametrize("camera_key", ["Camera_CamID", "camera_name"])
    def test_camera_consistency(self, camera_key: str) -> None:
        paths = list(DataPath.iterate(settings.data_dirs.semantic))
        paths += list(DataPath.iterate(settings.data_dirs.masks))
        paths += list(DataPath.iterate(settings.data_dirs.masks / "overlap"))

        df = generate_metadata_table(paths)
        df["subject_name"] = [v.split("#")[0] for v in df["image_name"].values]
        errors = []

        # All images per subject must use the same camera
        for subject_name in df["subject_name"].unique():
            cam_names = df.query("subject_name == @subject_name")[camera_key].unique()
            if len(cam_names) != 1:
                df_count = sqldf(f"""
                    SELECT subject_name, {camera_key}, COUNT(*) AS n_images, GROUP_CONCAT(image_name) AS image_names
                    FROM df
                    WHERE subject_name = '{subject_name}'
                    GROUP BY {camera_key}
                """)
                errors.append(df_count)
            else:
                assert cam_names is not None

        assert len(errors) == 0, f"All images per pig should have been acquired with the same camera:\n{errors}"

    @pytest.mark.parametrize(
        "dataset_name", ["2021_02_05_Tivita_multiorgan_semantic", "2021_02_05_Tivita_multiorgan_masks"]
    )
    def test_timestamp(self, dataset_name: str) -> None:
        paths = list(DataPath.iterate(settings.data_dirs[dataset_name]))
        if (overlap_dir := settings.data_dirs[dataset_name] / "overlap").exists():
            paths += list(DataPath.iterate(overlap_dir))

        for p in paths:
            # Make sure we get the datetime info for every image
            assert p.timestamp.startswith(str(p.datetime().year))

            # Make sure every file starts with the timestamp
            for file in p().rglob("*"):
                if file.is_dir():
                    continue

                assert file.name.startswith(p.timestamp)

    def test_kidney_dataset(self) -> None:
        paths = list(DataPath.iterate(settings.data_dirs.kidney, annotation_name="semantic#primary"))
        paths += list(DataPath.iterate(settings.data_dirs.kidney / "overlap", annotation_name="semantic#primary"))

        df = median_table("2023_04_22_Tivita_multiorgan_kidney", annotation_name="semantic#primary")

        assert len(paths) > 0 and len(df) > 0
        assert len({p.image_name() for p in paths}) == len(paths)
        assert sorted([p.image_name() for p in paths]) == sorted(df["image_name"].unique().tolist())

        len_before = len(paths)
        paths += list(DataPath.iterate(settings.data_dirs.kidney, annotation_name="perfusion#primary"))
        assert len(paths) > len_before

        mapping = LabelMapping.from_path(paths[0])
        for p in paths:
            seg = p.read_segmentation()
            assert seg.shape == (480, 640)
            assert mapping.is_index_valid(seg).all()

    def test_human_dataset(self) -> None:
        paths = list(DataPath.iterate(settings.data_dirs.human, annotation_name="semantic#primary"))

        df = median_table("2021_07_26_Tivita_multiorgan_human", annotation_name="semantic#primary")

        assert len(paths) > 0 and len(df) > 0
        assert len({p.image_name() for p in paths}) == len(paths)
        assert sorted([p.image_name() for p in paths]) == sorted(df["image_name"].unique().tolist())

        mapping = LabelMapping.from_path(paths[0])
        for p in paths:
            seg = p.read_segmentation()
            assert seg.shape == (480, 640)
            assert mapping.is_index_valid(seg).all()
