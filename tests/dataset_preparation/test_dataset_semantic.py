# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import filecmp
from collections.abc import Callable
from multiprocessing import set_start_method

import numpy as np
import pandas as pd
import pytest
import torch
from pandas.testing import assert_frame_equal

from htc.cpp import segmentation_mask
from htc.dataset_preparation.run_dataset_semantic import DatasetGeneratorSemantic
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.blosc_compression import decompress_file
from htc.utils.helper_functions import median_table


class TestDataGeneratorSemantic:
    def test_segmentation_mask(self) -> None:
        label_image = torch.tensor(
            [[[0, 0, 0], [0, 0, 0], [0, 0, 1]], [[0, 0, 2], [0, 0, 3], [0, 0, 3]]], dtype=torch.uint8
        )
        color_mapping = {
            (0, 0, 0): 1,
            (0, 0, 1): 2,
            (0, 0, 2): 10,
            (0, 0, 3): 20,
        }
        label_mask = segmentation_mask(label_image, color_mapping)

        assert torch.all(label_mask == torch.tensor([[1, 1, 2], [10, 20, 20]], dtype=torch.uint8))

        label_image[1, 2, :] = torch.tensor([0, 0, 4], dtype=torch.uint8)
        with pytest.raises(AssertionError, match="not part of the mapping.*(0, 0, 4)"):
            segmentation_mask(label_image, color_mapping)

    def test_data_generation(self, make_tmp_example_data: Callable) -> None:
        set_start_method("spawn", force=True)  # Avoid timeouts due to multiprocessing
        median = median_table("2021_02_05_Tivita_multiorgan_semantic", annotation_name="all")
        median_context = median_table(
            "2021_02_05_Tivita_multiorgan_semantic#context_experiments", annotation_name="all"
        )

        tmp_example_dataset = make_tmp_example_data(
            paths=[
                DataPath.from_image_name("P041#2019_12_14_13_33_30"),
                DataPath.from_image_name("P068#2020_07_20_18_15_37"),
                DataPath.from_image_name("P144#2023_02_07_10_43_28"),
            ]
        )
        tmp_intermediates = tmp_example_dataset / "intermediates"

        paths = list(DataPath.iterate(tmp_example_dataset / "data"))
        paths += list(DataPath.iterate(tmp_example_dataset / "data" / "context_experiments"))
        gen = DatasetGeneratorSemantic(output_path=tmp_example_dataset)
        assert paths == gen.paths

        # The make_tmp_example_data already runs the meta file generation function
        gen.median_spectra_table()
        gen.preprocessed_files()
        gen.view_all()
        gen.view_annotators()

        for p in paths:
            rel_paths = [
                f"segmentations/{p.image_name()}.blosc",
                f"view_all/{p.image_name()}.html",
                f"preprocessing/L1/{p.image_name()}.blosc",
                f"preprocessing/parameter_images/{p.image_name()}.blosc",
            ]

            segmentations = decompress_file(tmp_intermediates / rel_paths[0])
            for seg in segmentations.values():
                assert seg.dtype == np.uint8
                assert [label in p.dataset_settings["label_mapping"].values() for label in np.unique(seg)]

            if len(segmentations) > 1:
                rel_paths.append(
                    f"view_annotators/{p.image_name()}.html",
                )

            for rel_path in rel_paths:
                file_tmp = tmp_intermediates / rel_path
                assert file_tmp.exists()
                assert filecmp.cmp(file_tmp, settings.intermediates_dir_all / rel_path, shallow=False)

        # The function now uses the temporarily created median spectra table
        tmp_median = median_table("2021_02_05_Tivita_multiorgan_semantic", annotation_name="all")
        tmp_median_context = median_table(
            "2021_02_05_Tivita_multiorgan_semantic#context_experiments", annotation_name="all"
        )
        tmp_meta = pd.read_feather(tmp_intermediates / "tables" / "2021_02_05_Tivita_multiorgan_semantic@meta.feather")
        meta = pd.read_feather(
            settings.intermediates_dir_all / "tables" / "2021_02_05_Tivita_multiorgan_semantic@meta.feather"
        )

        img_names = [p.image_name() for p in paths]
        median = median.query("image_name in @img_names").reset_index(drop=True)
        median_context = median_context.query("image_name in @img_names").reset_index(drop=True)
        meta = meta.query("image_name in @img_names").reset_index(drop=True)

        assert_frame_equal(median, tmp_median, atol=1e-3)
        assert_frame_equal(median_context, tmp_median_context, atol=1e-3)
        assert_frame_equal(meta, tmp_meta)
