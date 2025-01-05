# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pytest
import torch
from pytest import LogCaptureFixture

from htc.models.common.utils import get_n_classes, infer_swa_lr, multi_label_condensation
from htc.settings_seg import settings_seg
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping


def test_get_n_classes() -> None:
    config = Config({"label_mapping": LabelMapping({"a": 0, "b": 1, "c": 2}, last_valid_label_index=1)})
    LabelMapping.from_config(config)
    assert get_n_classes(config) == 2

    config = Config({"label_mapping": LabelMapping({"a": 0, "b": 1, "c": 2}, last_valid_label_index=1)})
    assert get_n_classes(config) == 2

    config = Config.from_model_name(model_name="image")
    assert get_n_classes(config) == len(settings_seg.labels)


def test_swa_lr() -> None:
    config = Config.from_model_name("default", "image")
    assert infer_swa_lr(config) == pytest.approx(0.001 * 0.99**79)

    config["swa_kwargs/swa_lrs"] = 0.02
    assert infer_swa_lr(config) == 0.02

    config = Config.from_model_name("default", "median_pixel")
    assert infer_swa_lr(config) == pytest.approx(0.0001 * 0.9**7)


def test_multi_label_condensation(caplog: LogCaptureFixture) -> None:
    config = Config({"label_mapping": {"network_unsure": 2}})
    confidences = (
        torch.tensor([
            [0.8, 0.2, 0.3],
            [0.6, 0.7, 0.1],
            [0.2, 0.1, 0.3],
        ])
        .unsqueeze(dim=-1)
        .unsqueeze(dim=-1)
    )

    res = multi_label_condensation(confidences, config)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING" and "provide the raw logits" in caplog.records[0].msg

    # Invert the sigmoid
    logits = torch.log(confidences / (1 - confidences))

    res = multi_label_condensation(logits, config)
    assert torch.all(res["predictions"] == torch.tensor([0, 2, 2]).unsqueeze(dim=-1).unsqueeze(dim=-1))
    assert torch.all(res["confidences"] == torch.tensor([0.8, 0.1, 0.3]).unsqueeze(dim=-1).unsqueeze(dim=-1))

    res = multi_label_condensation(logits.squeeze(), config)
    assert torch.all(res["predictions"] == torch.tensor([0, 2, 2]))
    assert torch.all(res["confidences"] == torch.tensor([0.8, 0.1, 0.3]))
