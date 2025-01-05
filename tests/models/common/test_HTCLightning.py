# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import logging

import pytest
import torch
from pytest import LogCaptureFixture
from torch.utils.data import DataLoader

from htc.models.common.HTCLightning import HTCLightning
from htc.models.common.torch_helpers import move_batch_gpu
from htc.models.common.transforms import ToType
from htc.models.image.DatasetImage import DatasetImage
from htc.models.image.LightningImage import LightningImage
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.helper_functions import checkpoint_path


class TestHTCLightning:
    def test_class_from_config(self) -> None:
        config = Config({"lightning_class": "htc.models.image.LightningImage>LightningImage"})
        LightningCLass = HTCLightning.class_from_config(config)
        assert type(LightningCLass) == type

        module = LightningCLass(dataset_train=None, datasets_val=None, config=config)
        assert isinstance(module, LightningImage)

    @pytest.mark.serial
    def test_predict_step(self, caplog: LogCaptureFixture) -> None:
        run_dir = settings.training_dir / "image" / "2022-02-03_22-58-44_generated_default_model_comparison"
        config = Config(run_dir / "config.json")
        config["input/preprocessing"] = None  # We test normalization explicitly below
        path = DataPath.from_image_name("P041#2019_12_14_12_00_16")

        dataset = DatasetImage([path], train=False, config=config)
        assert len(dataset.transforms) == 1
        assert type(dataset.transforms[0]) == ToType and dataset.transforms[0].dtype == torch.float16

        dataloader = DataLoader(dataset)
        batch = next(iter(dataloader))
        batch_gpu = move_batch_gpu(batch)

        fold_dir = run_dir / "fold_P041,P060,P069"
        ckpt_file, _ = checkpoint_path(fold_dir)

        LightningClass = HTCLightning.class_from_config(config)
        model = LightningClass.load_from_checkpoint(
            ckpt_file, dataset_train=None, datasets_val=[dataset], config=config
        )
        model.cuda()

        with pytest.raises(AssertionError, match="autocast"):
            model.predict_step(batch_gpu)

        with torch.autocast(device_type="cuda"):
            with pytest.raises(AssertionError, match="gradients"):
                model.predict_step(batch_gpu)

        with torch.autocast(device_type="cuda"), torch.no_grad():
            with pytest.raises(AssertionError, match="eval"):
                model.predict_step(batch_gpu)

        with torch.autocast(device_type="cuda"), torch.no_grad():
            model.eval()
            predictions = model.predict_step(batch_gpu)
            assert batch_gpu["features"].dtype == torch.float32
            assert "class" in predictions
            assert predictions["class"].shape == (1, 19, 480, 640)

        # We get nan values if we forget normalization
        model.transforms["input/test_time_transforms_gpu"] = model.transforms["input/test_time_transforms_gpu"][:1]
        with torch.autocast(device_type="cuda"), torch.no_grad():
            batch_gpu = move_batch_gpu(batch)
            model.predict_step(batch_gpu)
            assert len(caplog.records) > 0
            assert caplog.records[-1].levelno == logging.WARNING
            assert "nan values" in caplog.records[-1].msg

        del config["input/test_time_transforms_gpu"]
        del config["input/transforms_gpu"]

        model = LightningClass.load_from_checkpoint(
            ckpt_file, dataset_train=None, datasets_val=[dataset], config=config
        )
        model.cuda()
        model.eval()

        assert len(model.transforms) == 0
        with torch.autocast(device_type="cuda"), torch.no_grad():
            batch_gpu = move_batch_gpu(batch)
            model.predict_step(batch_gpu)
        assert len(model.transforms["input/test_time_transforms_gpu"]) == 1
        t = model.transforms["input/test_time_transforms_gpu"][0]
        assert type(t) == ToType and t.dtype == torch.float32, "We always need a type transformation"
