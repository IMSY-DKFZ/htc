# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pytest
import torch
from lightning import seed_everything

from htc.utils.ColorcheckerReader import ColorcheckerReader
from htc.utils.test_helpers import DataPathArray


class TestColorcheckerReader:
    @pytest.mark.parametrize("safety_margin", [0, 12])
    def test_classic(self, safety_margin: int) -> None:
        seed_everything(42)
        img = torch.rand(480, 640, 3, dtype=torch.float32)
        path = DataPathArray(img.numpy())
        reader = ColorcheckerReader(path, cc_board="cc_classic", rot_angle=0)

        if safety_margin == 0:
            mask_params = {
                "offset_left": 32,
                "offset_top": 64,
                "square_dist_horizontal": 40,
                "square_dist_vertical": 36,
                "square_size": 64,
            }
        else:
            mask_params = {
                "offset_left": 20,
                "offset_top": 20,
                "square_dist_horizontal": 30,
                "square_dist_vertical": 36,
                "square_size": 64,
            }

        mask_params_margin = {"mask_0": mask_params.copy()}
        mask_params_margin["mask_0"]["square_size"] += safety_margin
        mask = reader.create_mask(mask_params_margin)

        # Load the altered image again
        img[mask >= 1] = 0
        reader = ColorcheckerReader(path, cc_board="cc_classic", rot_angle=0)

        reader.safety_margin = safety_margin
        reader.square_dist_horizontal = mask_params["square_dist_horizontal"]
        reader.square_dist_vertical = mask_params["square_dist_vertical"]
        automask = reader.create_automask()
        assert (automask.unique() == torch.arange(25)).all()

        assert len(reader.mask_params) == 1
        assert reader.mask_params["mask_0"].keys() == mask_params.keys()
        if safety_margin == 0:
            assert torch.all(automask == mask)
            for key in mask_params.keys():
                assert reader.mask_params["mask_0"][key] == mask_params[key]
        else:
            assert torch.all(img[automask >= 1] == 0)
            assert reader.mask_params["mask_0"]["square_size"] == mask_params["square_size"]
            for key in ["square_dist_horizontal", "square_dist_vertical"]:
                assert (
                    mask_params[key] + safety_margin - 1
                    <= reader.mask_params["mask_0"][key]
                    <= mask_params[key] + safety_margin + 1
                )
            for key in ["offset_left", "offset_top"]:
                assert (
                    mask_params[key] + safety_margin // 2 - 1
                    <= reader.mask_params["mask_0"][key]
                    <= mask_params[key] + safety_margin // 2 + 1
                )

    def test_passport(self) -> None:
        seed_everything(42)
        img = torch.rand(480, 640, 3, dtype=torch.float32)
        path = DataPathArray(img.numpy())
        reader = ColorcheckerReader(path, cc_board="cc_passport", rot_angle=0)

        mask_params = {
            "mask_0": {
                "offset_left": 40,
                "offset_top": 98,
                "square_size": 26,
                "square_dist_horizontal": 35,
                "square_dist_vertical": 35,
            },
            "mask_1": {
                "offset_left": 397,
                "offset_top": 94,
                "square_size": 26,
                "square_dist_horizontal": 35,
                "square_dist_vertical": 35,
            },
        }
        mask = reader.create_mask(mask_params)

        # Load the altered image again
        img[mask >= 1] = 0
        reader = ColorcheckerReader(path, cc_board="cc_passport", rot_angle=0)

        reader.square_dist_horizontal = 35
        reader.square_dist_vertical = 35
        reader.safety_margin = 0
        automask = reader.create_automask()
        assert (automask.unique() == torch.arange(49)).all()

        assert len(reader.mask_params) == 2
        assert reader.mask_params["mask_0"].keys() == mask_params["mask_0"].keys()
        assert reader.mask_params["mask_1"].keys() == mask_params["mask_1"].keys()
        assert torch.all(automask == mask)
        for key in mask_params["mask_0"].keys():
            assert reader.mask_params["mask_0"][key] == mask_params["mask_0"][key]
            assert reader.mask_params["mask_1"][key] == mask_params["mask_1"][key]
