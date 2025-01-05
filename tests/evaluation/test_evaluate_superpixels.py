# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pytest
import torch

from htc.evaluation.evaluate_superpixels import EvaluateSuperpixelImage
from htc.models.image.DatasetImage import DatasetImage
from htc.settings_seg import settings_seg
from htc.utils.Config import Config


@pytest.mark.filterwarnings(
    r"ignore:(ground truth|prediction) is all 0, this may result in nan/inf distance\.:UserWarning"
)
class TestEvaluateSuperpixelImage:
    def test_implementations_identical(self) -> None:
        config = Config({
            "label_mapping": settings_seg.label_mapping,
            "input/normalization": "L1",
            "input/superpixels/n_segments": 1000,
            "input/superpixels/compactness": 10,
        })

        dataset = DatasetImage.example_dataset(config)
        sample = dataset[0]
        assert sample["spxs"].max() < config["input/superpixels/n_segments"]

        spx_eval = EvaluateSuperpixelImage()
        result_py = spx_eval.evaluate_py(sample)
        result_cpp = spx_eval.evaluate_cpp(sample)

        assert result_py["evaluation"]["dice_metric_image"] == result_cpp["evaluation"]["dice_metric_image"]
        assert result_py["evaluation"].keys() == result_cpp["evaluation"].keys()
        for key in result_py["evaluation"]:
            eq = result_py["evaluation"][key] == result_cpp["evaluation"][key]
            if type(eq) == bool:
                assert eq
            else:
                assert torch.all(eq)

        assert torch.all(result_py["predictions"] == result_cpp["predictions"])
        assert result_py["predictions"].shape == sample["labels"].shape
        assert torch.all(result_py["label_counts"] == result_cpp["label_counts"])
