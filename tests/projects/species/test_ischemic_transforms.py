# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pytest
import torch
from lightning import seed_everything
from torch.utils.data import DataLoader

from htc.models.common.torch_helpers import copy_sample, move_batch_gpu
from htc.models.common.transforms import HTCTransformation
from htc.models.image.DatasetImage import DatasetImage
from htc.settings_seg import settings_seg
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config


@pytest.mark.serial
class TestProjectionTransform:
    @pytest.mark.parametrize(
        "p,label_mode", [(0, "random"), (1, "random"), (1, "label_match"), (1, "label_match_extended")]
    )
    def test_basics(self, p: float, label_mode: str) -> None:
        seed_everything(0, workers=True)

        path_kidney = DataPath.from_image_name("P058#2020_05_13_19_05_34")
        path_kidney2 = DataPath.from_image_name("P062#2020_05_15_20_00_19")
        path_other = DataPath.from_image_name("P059#2020_05_14_13_00_11")
        config = Config({
            "label_mapping": settings_seg.label_mapping,
            "input/preprocessing": "L1",
            "input/features_dtype": "float32",
            "input/n_channels": 100,
            "input/transforms_gpu": [
                {
                    "class": "htc_projects.species.species_transforms>ProjectionTransform",
                    "base_name": "weights+bias_malperfusion_pig_kidney=P091,P095,P097,P098+aortic",
                    "label_mode": label_mode,
                    "target_labels": ["kidney"] if label_mode == "label_match" else None,
                    "p": p,
                }
            ],
        })

        dataset = DatasetImage([path_kidney, path_kidney2, path_other], train=True, config=config)
        dataloader = DataLoader(dataset, batch_size=3)
        batch = next(iter(dataloader))
        batch = move_batch_gpu(batch)
        batch_copy = copy_sample(batch)

        seed_everything(0, workers=True)  # Reproducible conditions for the label selection in the transform
        aug = HTCTransformation.parse_transforms(
            config["input/transforms_gpu"], config=config, device=batch["features"].device
        )
        batch = HTCTransformation.apply_valid_transforms(batch, aug)

        assert torch.all(batch["labels"] == batch_copy["labels"])
        assert batch["features"].dtype == torch.float32
        if label_mode == "label_match" or p == 0:
            assert torch.allclose(batch["features"][2], batch_copy["features"][2])
        else:
            assert not torch.allclose(batch["features"][2], batch_copy["features"][2])

        if p == 0:
            assert torch.allclose(batch["features"][0], batch_copy["features"][0])
            assert torch.allclose(batch["features"][1], batch_copy["features"][1])
        else:
            if label_mode == "label_match":
                kidney_features = lambda batch, b: batch["features"][b][
                    batch["labels"][b] == settings_seg.label_mapping.name_to_index("kidney")
                ]
                non_kidney_features = lambda batch, b: batch["features"][b][
                    batch["labels"][b] != settings_seg.label_mapping.name_to_index("kidney")
                ]
                assert not torch.allclose(kidney_features(batch, 0), kidney_features(batch_copy, 0))
                assert not torch.allclose(kidney_features(batch, 1), kidney_features(batch_copy, 1))
                assert torch.allclose(non_kidney_features(batch, 0), non_kidney_features(batch_copy, 0))
                assert torch.allclose(non_kidney_features(batch, 1), non_kidney_features(batch_copy, 1))
            else:
                # Every spectra should change
                assert not torch.isclose(batch["features"][0], batch_copy["features"][0]).all(dim=-1).any()
                assert not torch.isclose(batch["features"][1], batch_copy["features"][1]).all(dim=-1).any()

        if p == 1:
            seed_everything(0, workers=True)
            config["input/transforms_gpu"][0]["interpolate"] = True
            aug = HTCTransformation.parse_transforms(
                config["input/transforms_gpu"], config=config, device=batch["features"].device
            )
            batch_interp = copy_sample(batch_copy)
            batch_interp = HTCTransformation.apply_valid_transforms(batch_interp, aug)

            assert not torch.allclose(batch_interp["features"][0], batch_copy["features"][0])

            diff_full = torch.abs(batch_copy["features"][0] - batch["features"][0]).sum()
            diff_interp = torch.abs(batch_copy["features"][0] - batch_interp["features"][0]).sum()
            assert diff_interp < diff_full
