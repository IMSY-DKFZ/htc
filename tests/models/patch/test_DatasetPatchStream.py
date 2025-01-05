# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import io

import numpy as np
import pytest
from lightning import seed_everything

from htc.models.data.DataSpecification import DataSpecification
from htc.models.patch.DatasetPatchStream import DatasetPatchStream
from htc.models.patch.LightningPatch import LightningPatch
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping
from htc.utils.test_helpers import DataPathArray
from htc_projects.camera.settings_camera import settings_camera


class TestDatasetPatchStream:
    @pytest.mark.parametrize(
        "label_pos, center_pos",
        [
            [(10, 10), (16, 16)],
            [(10, 120), (16, 128 - 16)],
            [(120, 10), (128 - 16, 16)],
            [(120, 120), (128 - 16, 128 - 16)],
            [(64, 64), (64, 64)],
        ],
    )
    def test_border_sampling(self, label_pos: tuple[int, int], center_pos: tuple[int, int]) -> None:
        cube = np.zeros((128, 128, 100), dtype=np.float32)
        seg = np.ones((128, 128), dtype=np.int64)
        seg[label_pos] = 0
        mapping = LabelMapping({"a": 0, "b": 1}, last_valid_label_index=0)
        config = Config({
            "input/patch_size": [32, 32],
            "input/n_channels": 100,
            "input/patch_sampling": "proportional",
            "dataloader_kwargs/num_workers": 1,
            "dataloader_kwargs/batch_size": 1,
            "label_mapping": mapping,
        })
        paths = [DataPathArray(cube, seg, mapping)]
        dataset = DatasetPatchStream(paths, train=False, config=config, single_pass=True)

        sample = list(dataset.iter_samples(include_position=True))
        assert len(sample) == 1
        sample = sample[0]

        assert sample["valid_pixels"].sum() == 1
        assert sample["features"].shape == (32, 32, 100)
        assert sample["center_row"] == center_pos[0]
        assert sample["center_col"] == center_pos[1]

    @pytest.mark.parametrize("sampling_strategy, n_patches", [("proportional", 8), ("uniform", 16)])
    def test_n_patches(self, sampling_strategy: str, n_patches: int) -> None:
        cube = np.zeros((128, 128, 100), dtype=np.float32)
        seg = np.ones((128, 128), dtype=np.int64)
        seg[:, :64] = 0  # Half of the image is not used in proportional sampling
        mapping = LabelMapping({"a": 0, "b": 1}, last_valid_label_index=0)
        config = Config({
            "input/patch_size": [32, 32],
            "input/n_channels": 100,
            "input/patch_sampling": sampling_strategy,
            "dataloader_kwargs/num_workers": 1,
            "dataloader_kwargs/batch_size": 1,
            "label_mapping": mapping,
        })
        paths = [DataPathArray(cube, seg, mapping)]
        dataset = DatasetPatchStream(paths, train=False, config=config, single_pass=True)

        assert len(list(dataset.iter_samples(include_position=True))) == n_patches

    def test_sampling_all(self) -> None:
        seed_everything(0, workers=True)

        cube = np.zeros((128, 128, 100), dtype=np.float32)
        cube[0:32, 0:32] = 1
        cube[50 - 16 : 50 + 16, 50 - 16 : 50 + 16] = 2
        cube[0:32, -32:] = 3
        seg = np.ones((128, 128), dtype=np.int64)
        seg[0, 0] = 0
        seg[50, 50] = 0
        seg[0, -1] = 0

        mapping = LabelMapping({"a": 0, "b": 1}, last_valid_label_index=0)
        config = Config({
            "input/patch_size": [32, 32],
            "input/n_channels": 100,
            "input/patch_sampling": "all_valid",
            "dataloader_kwargs/num_workers": 1,
            "dataloader_kwargs/batch_size": 1,
            "label_mapping": mapping,
        })
        paths = [DataPathArray(cube, seg, mapping)]
        dataset = DatasetPatchStream(paths, train=False, config=config, single_pass=True)

        samples = list(dataset.iter_samples(include_position=True))
        assert len(samples) == 3

        possible_values = {1, 2, 3}  # We do not know in which order the samples arrive
        for i in range(len(samples)):
            assert samples[i]["valid_pixels"].sum() == 1

            values = samples[i]["features"].rename(None).unique()
            assert len(values) == 1
            values = values.item()
            assert values in possible_values
            possible_values.remove(values)

    def test_hierarchical_sampling(self) -> None:
        seed_everything(0, workers=True)

        # ('P083#2021_03_05_11_23_20', '0202-00118_wrong-1')
        # ('P072#2020_08_08_13_20_26', '0102-00085_correct-1')
        # ('P094#2021_04_30_12_41_58', '0202-00118_correct-1')
        # ('P077#2021_01_31_14_06_26', '0102-00098_wrong-1')
        specs_json = """
        [
            {
                "fold_name": "fold_1",
                "train": {
                    "image_names": ["P083#2021_03_05_11_23_20", "P072#2020_08_08_13_20_26", "P094#2021_04_30_12_41_58", "P077#2021_01_31_14_06_26"]
                }
            }
        ]
        """
        specs = DataSpecification(io.StringIO(specs_json))
        config = Config({
            "input/patch_size": [32, 32],
            "input/n_channels": 3,
            "input/patch_sampling": "uniform",
            "input/epoch_size": "12 images",
            "input/hierarchical_sampling": True,
            "input/target_domain": ["camera_index"],
            "input/data_spec": specs,
            "dataloader_kwargs/num_workers": 4,
            "dataloader_kwargs/batch_size": 300,
            "label_mapping": settings_camera.label_mapping,
        })
        dataset = LightningPatch.dataset(train=True, paths=specs.paths(), config=config)
        model = LightningPatch(dataset_train=dataset, datasets_val=[], config=config)
        dataloader = model.train_dataloader()
        for b, batch in enumerate(dataloader):
            images = batch["image_index"].unique()
            assert len({dataset.paths[i].meta("camera_name") for i in images}) == len(images) == 4
        assert b == 11, "12 images requested"
