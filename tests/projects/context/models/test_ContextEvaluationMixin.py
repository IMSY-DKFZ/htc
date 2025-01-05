# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import torch
from lightning import seed_everything

from htc.utils.Config import Config
from htc_projects.context.models.ContextEvaluationMixin import ContextEvaluationMixin


class ContextTestLightning(ContextEvaluationMixin):
    def __init__(self, config: Config, mode: str, labels: torch.Tensor):
        self.config = config
        self.mode = mode
        self.labels = labels
        self.current_epoch = 0
        super().__init__()

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int = None) -> dict[str, torch.Tensor]:
        n_classes = len(np.unique(self.labels))
        if self.mode == "part":
            # Always predict class 1
            class1 = torch.rand(self.labels.shape) + 10
            class2 = torch.rand(self.labels.shape)
            class3 = torch.rand(self.labels.shape)
        else:
            # 100 % correct predictor (via logits)
            class1 = torch.rand(self.labels.shape)
            class1[self.labels == 0] += 10
            class2 = torch.rand(self.labels.shape)
            class2[self.labels == 1] += 10
            class3 = torch.rand(self.labels.shape)
            class3[self.labels == 2] += 10

        if n_classes == 2:
            logits = torch.stack([class1, class2]).permute(1, 0, 2, 3)
        else:
            logits = torch.stack([class1, class2, class3]).permute(1, 0, 2, 3)

        if self.mode == "full":
            assert torch.all(logits.argmax(dim=1) == self.labels)

        return {"class": logits}


class TestContextEvaluationMixin:
    @pytest.mark.parametrize(
        "mode, dice_baseline, dice_context",
        [
            ["full", (1, 1), (1, 1)],  # full image correctly classified
            ["part", (1 / 1.5, 0), (1, 0)],  # Only part (here half) of the image correctly classified
        ],
    )
    def test_isolation(self, mode: str, dice_baseline: tuple[float, float], dice_context: tuple[float, float]) -> None:
        seed_everything(42)

        labels = torch.zeros(1, 100, 100, dtype=torch.int64)
        labels[0, :50, :] = 1
        batch = {
            "features": torch.rand(1, 100, 100, 3),
            "labels": labels,
            "valid_pixels": torch.ones(1, 100, 100, dtype=torch.bool),
            "image_name_annotations": ["P043#2019_12_20_12_38_35@semantic#primary"],
        }

        config = Config({
            "label_mapping": {"class1": 0, "class2": 1},
            "validation/context_transforms_gpu": {
                "isolation_0": [
                    {
                        "class": "htc_projects.context.context_transforms>OrganIsolation",
                        "fill_value": "0",
                    }
                ]
            },
        })
        lightning = ContextTestLightning(config, mode, labels)

        lightning.validation_step(batch=batch, batch_idx=0)
        res_baseline = lightning.validation_results_epoch["baseline"][0]
        res_isolation_0 = lightning.validation_results_epoch["isolation_0"][0]
        del res_baseline[0]["dice_metric_image"]

        assert len(res_baseline) == len(res_isolation_0) == 1
        assert res_baseline[0].keys() == res_isolation_0[0].keys()
        for key in res_baseline[0].keys():
            value_baseline = res_baseline[0][key]
            value_context = res_isolation_0[0][key]
            assert type(value_baseline) == type(value_context)
            if key == "dice_metric":
                assert value_baseline.tolist() == pytest.approx(dice_baseline)
                assert value_context.tolist() == pytest.approx(dice_context)
            elif type(value_baseline) == np.ndarray:
                assert (value_baseline == value_context).all()
            else:
                assert value_baseline == value_context

    @pytest.mark.parametrize(
        "mode, dice_baseline, dice_context",
        [
            [
                "full",
                (1, 1, 1),
                {
                    0: np.array((1, 1), dtype=np.float32),
                    1: np.array((1, 1), dtype=np.float32),
                    2: np.array((1, 1), dtype=np.float32),
                },
            ],  # full image correctly classified
            [
                "part",
                (0.5, 0, 0),
                {
                    0: np.array((0, 0), dtype=np.float32),
                    1: np.array((2 / 3, 0), dtype=np.float32),
                    2: np.array((2 / 3, 0), dtype=np.float32),
                },
            ],  # Only part (here 1/3) of the image correctly classified
        ],
    )
    def test_removal(
        self, mode: str, dice_baseline: tuple[float, float, float], dice_context: tuple[float, float, float]
    ) -> None:
        seed_everything(42)

        labels = torch.zeros(1, 99, 100, dtype=torch.int64)
        labels[0, 33:66, :] = 1
        labels[0, 66:99, :] = 2
        batch = {
            "features": torch.rand(1, 99, 100, 3),
            "labels": labels,
            "valid_pixels": torch.ones(1, 99, 100, dtype=torch.bool),
            "image_name_annotations": ["P043#2019_12_20_12_38_35@semantic#primary"],
        }

        config = Config({
            "label_mapping": {"class1": 0, "class2": 1, "class3": 2},
            "validation/context_transforms_gpu": {
                "removal_0": [
                    {
                        "class": "htc_projects.context.context_transforms>OrganRemoval",
                        "fill_value": "0",
                    }
                ]
            },
        })
        lightning = ContextTestLightning(config, mode, labels)

        lightning.validation_step(batch=batch, batch_idx=0)
        res_baseline = lightning.validation_results_epoch["baseline"][0]
        res_removal_0 = lightning.validation_results_epoch["removal_0"][0]

        assert len(res_baseline) == 1
        assert all(res_baseline[0]["dice_metric"] == dice_baseline)
        assert len(res_removal_0) == 3

        complete_keys = list(res_baseline[0].keys())
        complete_keys += ["target_label"]
        assert sorted(complete_keys) == sorted(res_removal_0[0].keys())
        removal_dices = {}
        for removal_res in res_removal_0:
            target_label = removal_res["target_label"]
            assert target_label not in removal_res["used_labels"]
            removal_dices[target_label] = removal_res["dice_metric"]

        assert removal_dices.keys() == dice_context.keys()
        for key, val in removal_dices.items():
            assert all(val == dice_context[key])
