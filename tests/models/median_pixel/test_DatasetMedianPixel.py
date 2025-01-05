# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import logging
import warnings
from collections.abc import Callable

import numpy as np
import torch
from pytest import LogCaptureFixture

from htc.models.data.DataSpecification import DataSpecification
from htc.models.median_pixel.DatasetMedianPixel import DatasetMedianPixel
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping
from htc.utils.sqldf import sqldf
from htc_projects.atlas.tables import median_cam_table


class TestDatasetMedianPixel:
    def test_atlas(self, caplog: LogCaptureFixture) -> None:
        specs = DataSpecification(
            settings.htc_projects_dir / "atlas/data/tissue-atlas_loocv_test-8_seed-0_cam-118.json"
        )
        specs.activate_test_set()
        paths = specs.paths()
        config = Config.from_model_name("default", "median_pixel")

        df = median_cam_table()
        # Same order as defined by the paths
        img_names = [p.image_name() for p in paths]
        df = df.set_index("image_name").loc[img_names].reset_index()

        dataset = DatasetMedianPixel(paths, train=False, config=config)
        assert len(dataset) == len(df), "Every entry in the table should be used"
        assert [p.image_name() for p in dataset.paths] == df["image_name"].tolist()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable", category=UserWarning)
            assert torch.all(
                torch.isclose(dataset[0]["features"], torch.from_numpy(df.iloc[0]["median_normalized_spectrum"]).half())
            )

        # Iterate once over the dataset and check the labels
        original_mapping = LabelMapping.from_path(paths[0])
        label_mapping = LabelMapping.from_config(config)
        labels_dataset = []
        labels_df = []
        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample["image_name"] == df.iloc[i]["image_name"]

            labels_dataset.append(label_mapping.index_to_name(sample["labels"].item()))
            labels_df.append(original_mapping.index_to_name(df.iloc[i]["label_index"]))

        assert sorted(labels_dataset) == sorted(labels_df)

        for name in ["train", "val", "test"]:
            dataset_part = DatasetMedianPixel(specs.paths(name), train=False, config=config)
            image_names = [p.image_name() for p in specs.paths(name)]
            assert len(image_names) > 0
            assert len(dataset_part) == len(df.query("image_name in @image_names"))

        # Now we mark one label as invalid
        spleen_label_index = label_mapping.name_to_index("spleen")
        del label_mapping.mapping_index_name[spleen_label_index]
        label_mapping.mapping_index_name[settings.label_index_thresh] = "spleen"
        label_mapping.mapping_name_index["spleen"] = settings.label_index_thresh

        df_selection = df[["image_name", "label_name"]]
        df_spleen_only = sqldf("""
            SELECT image_name, COUNT(DISTINCT label_name) AS n_labels
            FROM df_selection
            GROUP BY image_name
            -- Images which contain only spleen (these images will be removed by the dataset)
            HAVING label_name == 'spleen' AND n_labels = 1
        """)

        dataset2 = DatasetMedianPixel(paths, train=False, config=config)

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 1
        assert f"{len(df_spleen_only)} image_names are not used" in warning_records[0].msg

        assert len(dataset) - len(df.query('label_name == "spleen"')) == len(dataset2)
        assert len(dataset2.labels.unique()) == len(dataset.labels.unique()) - 1
        assert spleen_label_index not in dataset2.labels.unique()

    def test_dataset_mix(self) -> None:
        paths = [
            DataPath.from_image_name("P070#2020_07_25_00_32_20"),  # semantic: heart, muscle, lung
            DataPath.from_image_name("P053#2020_03_06_17_15_27"),  # masks: muscle
            DataPath.from_image_name("P072#2020_08_08_18_05_43#overlap"),  # masks: heart
        ]
        config = Config({
            "input/normalization": "L1",
            "label_mapping": {"muscle": 0, "heart": 1, "lung": settings.label_index_thresh},
        })
        dataset = DatasetMedianPixel(paths, train=False, config=config)

        assert len(dataset) == 4
        assert dataset[0]["image_name"] == "P070#2020_07_25_00_32_20" and dataset[0]["labels"] == 1
        assert dataset[1]["image_name"] == "P070#2020_07_25_00_32_20" and dataset[1]["labels"] == 0
        assert dataset[2]["image_name"] == "P053#2020_03_06_17_15_27" and dataset[2]["labels"] == 0
        assert dataset[3]["image_name"] == "P072#2020_08_08_18_05_43#overlap" and dataset[3]["labels"] == 1

    def test_meta_and_image_labels(self, check_sepsis_data_accessible: Callable) -> None:
        check_sepsis_data_accessible()

        path = DataPath.from_image_name("S438#2023_10_02_19_23_03")
        config = Config({
            "task": "classification",
            "label_mapping": LabelMapping({"finger": 0}, last_valid_label_index=0),
            "input/normalization": "L1",
            "input/image_labels": [
                {
                    "meta_attributes": ["sepsis_status"],
                    "image_label_mapping": "htc_projects.sepsis_icu.settings_sepsis_icu>sepsis_label_mapping",
                }
            ],
            "input/meta/attributes": [{"name": "age"}, {"name": "sex", "mapping": {"male": 0, "female": 1}}],
        })
        sample = DatasetMedianPixel([path], train=False, config=config)[0]
        assert sample["image_name"] == "S438#2023_10_02_19_23_03"
        assert sample["image_labels"] == 0  # intermediates
        assert sample["labels"] == 0

        meta = sample["meta"].tolist()
        assert len(meta) == 2
        assert meta[0] == path.meta("age")
        assert meta[1] == config["input/meta/attributes"][1]["mapping"][path.meta("sex")]

    def test_feature_columns(self, check_sepsis_data_accessible: Callable) -> None:
        check_sepsis_data_accessible()

        path0 = DataPath.from_image_name("S001#2022_10_24_13_49_45")
        path1 = DataPath.from_image_name("S005#2022_10_24_14_44_17")
        config = Config({
            "task": "classification",
            "label_mapping": LabelMapping({"finger": 0}, last_valid_label_index=0),
            "input/normalization": "L1",
            "input/feature_columns": ["median_twi", "median_normalized_spectrum", "median_thi"],
        })
        batch = DatasetMedianPixel([path0, path1], train=False, config=config)[:]
        assert batch["image_name"] == ["S001#2022_10_24_13_49_45", "S005#2022_10_24_14_44_17"]
        assert torch.all(batch["labels"] == torch.tensor([0, 0], dtype=torch.int64))
        assert batch["features"].shape == (2, 102)

        twi1 = path1.compute_twi().data
        mapping = LabelMapping.from_path(path1)
        finger_mask = path1.read_segmentation() == mapping.name_to_index("finger")
        assert path1.annotated_labels() == ["finger"]
        assert np.allclose(batch["features"][1, 0].numpy(), np.median(twi1[finger_mask]))
