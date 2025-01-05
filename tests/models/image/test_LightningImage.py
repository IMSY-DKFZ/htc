# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy
from pathlib import Path

import pytest
import torch
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from htc.models.common.StreamDataLoader import StreamDataLoader
from htc.models.common.torch_helpers import move_batch_gpu
from htc.models.common.transforms import KorniaTransform, Normalization, ToType
from htc.models.common.utils import samples_equal
from htc.models.data.DataSpecification import DataSpecification
from htc.models.image.DatasetImage import DatasetImage
from htc.models.image.LightningImage import LightningImage
from htc.settings_seg import settings_seg
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config


class TestLightningImage:
    def test_num_workers(self) -> None:
        config = Config({
            "input/data_spec": "pigs_masks_loocv_4cam.json",
            "input/n_channels": 3,
            "dataloader_kwargs/batch_size": 10,
            "dataloader_kwargs/num_workers": 2,
        })
        specs = DataSpecification.from_config(config)
        paths = specs.fold_paths(specs.fold_names()[0])
        dataset_train = LightningImage.dataset(paths=paths, train=True, config=config)
        dataset_val = LightningImage.dataset(paths=paths, train=False, config=config)

        module = LightningImage(dataset_train, [dataset_val], config)
        dataloader_train = module.train_dataloader()
        dataloader_val = module.val_dataloader()[0]

        assert isinstance(dataloader_train, StreamDataLoader) and isinstance(dataloader_val, StreamDataLoader)
        assert dataloader_train.dataloader.num_workers == 2
        assert dataloader_val.dataloader.num_workers == 2
        assert config["dataloader_kwargs/num_workers"] == 2

    @pytest.mark.serial
    @pytest.mark.filterwarnings("ignore::UserWarning:lightning")
    def test_transforms_gpu(self, tmp_path: Path) -> None:
        path = DataPath.from_image_name("P058#2020_05_13_18_09_26")
        model = {
            "model_name": "ModelImage",
            "architecture_name": "Unet",
            "architecture_kwargs": {
                "encoder_name": "efficientnet-b5",
                "encoder_weights": "imagenet",
            },
        }

        config_cpu = Config({
            "model": model,
            "trainer_kwargs/precision": 32,
            "label_mapping": settings_seg.label_mapping,
            "input/n_channels": 3,
            "input/transforms_cpu": [
                {
                    "class": "KorniaTransform",
                    "transformation_name": "RandomHorizontalFlip",
                    "p": 1,
                }
            ],
        })
        dataset_cpu = DatasetImage([path], train=True, config=config_cpu)
        loader_cpu = DataLoader(dataset_cpu)
        module_cpu = LightningImage(dataset_cpu, [], config_cpu)
        assert len(module_cpu.transforms) == 0
        t = dataset_cpu.transforms
        assert len(t) == 2
        assert type(t[0]) == ToType and t[0].dtype == torch.float32
        assert type(t[1]) == KorniaTransform

        # For CPU augmentations, the training step does not change the batch
        batch_cpu = next(iter(loader_cpu))
        batch_cpu_backup = copy.deepcopy(batch_cpu)
        module_cpu.training_step(batch_cpu, batch_idx=0)
        assert samples_equal(batch_cpu, batch_cpu_backup)

        config_gpu = Config({
            "model": model,
            "trainer_kwargs/precision": 32,
            "trainer_kwargs/limit_train_batches": 1,
            "trainer_kwargs/limit_val_batches": 1,
            "trainer_kwargs/max_epochs": 1,
            "trainer_kwargs/accelerator": "gpu",
            "validation/checkpoint_metric": "dice_metric",
            "label_mapping": settings_seg.label_mapping,
            "input/n_channels": 3,
            "input/test_time_transforms_gpu": [
                {
                    "class": "KorniaTransform",
                    "transformation_name": "RandomHorizontalFlip",
                    "p": 1,
                }
            ],
            "input/transforms_gpu": [{"class": "Normalization"}],
            "optimization": {"optimizer": {"name": "Adam"}},
        })

        dataset_gpu = DatasetImage([path], train=True, config=config_gpu)
        loader_gpu = DataLoader(dataset_gpu)
        module_gpu = LightningImage(dataset_gpu, [], config_gpu)

        batch_gpu = next(iter(loader_gpu))
        batch_gpu = move_batch_gpu(batch_gpu)
        assert not samples_equal(move_batch_gpu(batch_cpu), batch_gpu)

        # Trigger the GPU augmentation step (the batch is changed in-place)
        logger = TensorBoardLogger(save_dir=tmp_path)
        trainer = Trainer(logger=logger, num_sanity_val_steps=0, **config_gpu["trainer_kwargs"])

        with torch.autocast(device_type="cuda"), torch.no_grad():
            trainer.fit(module_gpu, train_dataloaders=loader_gpu, val_dataloaders=loader_gpu)

        assert len(module_gpu.transforms) == 2
        transforms_train = module_gpu.transforms["input/transforms_gpu"]
        assert len(transforms_train) == 2
        assert type(transforms_train[0]) == ToType and transforms_train[0].dtype == torch.float32
        assert type(transforms_train[1]) == Normalization

        transforms_test = module_gpu.transforms["input/test_time_transforms_gpu"]
        assert len(transforms_test) == 2
        assert type(transforms_test[0]) == ToType and transforms_test[0].dtype == torch.float32
        assert type(transforms_test[1]) == KorniaTransform

        # Trigger the test-time transformation
        with torch.autocast(device_type="cuda"), torch.no_grad():
            module_gpu.eval()
            module_gpu.cuda()
            module_gpu.predict_step(batch_gpu)

        assert batch_gpu["transforms_applied"]
        del batch_gpu["transforms_applied"]
        assert samples_equal(batch_gpu, move_batch_gpu(batch_cpu)), (
            "CPU and GPU transformations should yield the same results"
        )
