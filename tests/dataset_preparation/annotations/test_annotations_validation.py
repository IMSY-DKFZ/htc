# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from htc import DataPath, LabelMapping
from htc.dataset_preparation.annotations.run_rebuild_nrrd import RebuildAnnotations
from htc.dataset_preparation.annotations.run_validate_nrrd import ValidateAnnotations
from htc.utils.mitk.mitk_masks import segmentation_to_nrrd


class TestAnnotationsValidation:
    def test_annotations_validation(
        self, tmp_path: Path, make_tmp_example_data: Callable, caplog: pytest.LogCaptureFixture
    ) -> None:
        tmp_example_dataset = make_tmp_example_data(n_images=4)

        paths = list(DataPath.iterate(tmp_example_dataset / "data"))

        nrrd_path = tmp_path / "nrrd_files"
        nrrd_path.mkdir(exist_ok=True)
        mapping = LabelMapping(mapping_name_index={"stomach": 1, "liver": 2}, zero_is_invalid=True)

        # including a small region of a different label
        seg_mask_shape = paths[0].dataset_settings["spatial_shape"]
        seg_mask = np.ones(shape=seg_mask_shape, dtype=int)
        seg_mask[0:2, 0:2] = 2
        nrrd_file = nrrd_path / f"{paths[0].image_name()}.nrrd"
        segmentation_to_nrrd(nrrd_file=nrrd_file, mask=seg_mask, mapping_mask=mapping)

        # including unlabeled pixels
        # also in this case the labels stomach and liver are part of the metadata but not the mask
        # this case includes a segmentation mask which is completely empty i.e. unlabeled
        seg_mask_shape = paths[1].dataset_settings["spatial_shape"]
        seg_mask = np.zeros(shape=seg_mask_shape, dtype=int)
        nrrd_file = nrrd_path / f"{paths[1].image_name()}.nrrd"
        segmentation_to_nrrd(nrrd_file=nrrd_file, mask=seg_mask, mapping_mask=mapping, mask_labels_only=False)

        # including an unknown label in the mapping i.e. a label not in the dataset_labels.json
        # also in this case the labels liver and ureter are part of the metadata but not the mask
        seg_mask_shape = paths[2].dataset_settings["spatial_shape"]
        seg_mask = np.ones(shape=seg_mask_shape, dtype=int)
        nrrd_file = nrrd_path / f"{paths[2].image_name()}.nrrd"
        mapping_unknown = LabelMapping(mapping_name_index={"stomach": 1, "liver": 2, "ureter": 3}, zero_is_invalid=True)
        segmentation_to_nrrd(nrrd_file=nrrd_file, mask=seg_mask, mapping_mask=mapping_unknown, mask_labels_only=False)

        nrrd_file_missing = nrrd_path / "P144#2023_02_07_10_43_28.nrrd"
        mapping = LabelMapping(mapping_name_index={"stomach": 1}, zero_is_invalid=True)
        segmentation_to_nrrd(nrrd_file=nrrd_file_missing, mask=seg_mask, mapping_mask=mapping)

        dataset_labels = [{"name": "stomach", "color": "#0475FF"}, {"name": "liver", "color": "#ED00D2"}]

        with (tmp_path / "dataset_labels.json").open("w") as f:
            json.dump(dataset_labels, f, indent=4)

        validate_annotations = ValidateAnnotations(
            nrrd_path_primary=nrrd_path,
            nrrd_path_secondary=nrrd_path,
            annotation_type="semantic",
            dataset_labels_path=tmp_path / "dataset_labels.json",
            small_regions_threshold=5,
        )
        validate_annotations.process_header()

        assert "Could not find the image name P144#2023_02_07_10_43_28" in caplog.text

        assert validate_annotations.minimum_threshold_region == [paths[0].image_name()]
        assert list(validate_annotations.missing_semantic_pixels.keys()) == [paths[1].image_name()]
        assert validate_annotations.empty_label_instances == {paths[1].image_name()}
        assert validate_annotations.missing_labels_in_mask == {paths[1].image_name(), paths[2].image_name()}
        assert paths[2].image_name() in validate_annotations.unknown_label_name.keys()
        assert validate_annotations.missing_annotations_primary == {paths[3].image_name()}

        # adding a nrrd file which is not part of the temporary dataset
        with pytest.raises(AssertionError, match="Data directory mismatch.*"):
            seg_mask = np.ones(shape=seg_mask_shape, dtype=int)
            nrrd_file_missing = nrrd_path / "R002#2023_09_19_10_14_28#0202-00118.nrrd"
            mapping = LabelMapping(mapping_name_index={"stomach": 1}, zero_is_invalid=True)
            segmentation_to_nrrd(nrrd_file=nrrd_file_missing, mask=seg_mask, mapping_mask=mapping)

            ValidateAnnotations(
                nrrd_path_primary=nrrd_path,
                nrrd_path_secondary=nrrd_path,
                annotation_type="semantic",
                dataset_labels_path=tmp_path / "dataset_labels.json",
                small_regions_threshold=5,
            )

    def test_annotations_rebuild(self, tmp_path: Path, make_tmp_example_data: Callable) -> None:
        tmp_example_dataset = make_tmp_example_data(n_images=4)

        paths = list(DataPath.iterate(tmp_example_dataset / "data"))

        nrrd_path = tmp_path / "nrrd_files"
        nrrd_path.mkdir(exist_ok=True)
        mapping = LabelMapping(mapping_name_index={"stomach": 1, "liver": 2}, zero_is_invalid=True)
        dataset_labels = [{"name": "stomach", "color": "#0475FF"}, {"name": "liver", "color": "#ED00D2"}]

        with (tmp_path / "dataset_labels.json").open("w") as f:
            json.dump(dataset_labels, f, indent=4)

        # including a small region of a different label
        seg_mask_shape = paths[0].dataset_settings["spatial_shape"]
        seg_mask = np.ones(shape=seg_mask_shape, dtype=int)
        seg_mask[0:2, 0:2] = 2
        nrrd_file = nrrd_path / f"{paths[0].image_name()}.nrrd"
        segmentation_to_nrrd(nrrd_file=nrrd_file, mask=seg_mask, mapping_mask=mapping)

        # add a multilayered segmentation mask, where one of the layer is completely empty
        mapping = LabelMapping(mapping_name_index={"stomach": 1}, zero_is_invalid=True)
        seg_mask_shape = (2, *paths[1].dataset_settings["spatial_shape"])
        seg_mask = np.ones(shape=seg_mask_shape, dtype=int)
        seg_mask[1, ...] = 0
        nrrd_file = nrrd_path / f"{paths[1].image_name()}.nrrd"
        segmentation_to_nrrd(nrrd_file=nrrd_file, mask=seg_mask, mapping_mask=mapping)

        # fix the small region present in the segmentation mask which has been added
        rebuild_annotations = RebuildAnnotations(
            nrrd_path_primary=nrrd_path,
            nrrd_path_secondary=nrrd_path,
            annotation_type="semantic",
            dataset_labels_path=tmp_path / "dataset_labels.json",
            small_regions_threshold=5,
        )
        rebuild_annotations.process_header()
        rebuild_annotations.rebuild_small_regions()
        rebuild_annotations.rebuild_empty_layers()

        # run the validation script, to find out if the small region segmentation mask has indeed been fixed
        validate_annotations = ValidateAnnotations(
            nrrd_path_primary=nrrd_path,
            nrrd_path_secondary=nrrd_path,
            annotation_type="semantic",
            dataset_labels_path=tmp_path / "dataset_labels.json",
            small_regions_threshold=5,
        )
        validate_annotations.process_header()

        assert len(validate_annotations.minimum_threshold_region) == 0
        assert len(validate_annotations.empty_label_instances) == 0
