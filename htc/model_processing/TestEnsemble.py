# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing_extensions import Self

from htc.models.common.HTCLightning import HTCLightning
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.helper_functions import checkpoint_path


class TestEnsemble(nn.Module):
    def __init__(self, model_paths: list[Path], paths: list[DataPath] | None, config: Config):
        super().__init__()
        self.config = config

        self.models: dict[Path, HTCLightning] = {}
        for fold_dir in model_paths:
            ckpt_file, _ = checkpoint_path(fold_dir)

            # Load dataset and lightning class based on model name
            LightningClass = HTCLightning.class_from_config(self.config)
            if paths is None:
                dataset = []
            else:
                dataset = LightningClass.dataset(paths=paths, train=False, config=self.config, fold_name=fold_dir.stem)
            model = LightningClass.load_from_checkpoint(
                ckpt_file, dataset_train=None, datasets_val=[dataset], config=self.config, fold_name=fold_dir.stem
            )

            self.models[fold_dir] = model

    def eval(self) -> Self:
        for model in self.models.values():
            model.eval()
        return self

    def cuda(self, *args, **kwargs) -> Self:
        for model in self.models.values():
            model.cuda(*args, **kwargs)
        return self

    def to(self, *args, **kwargs) -> Self:
        for model in self.models.values():
            model.to(*args, **kwargs)
        return self

    def val_dataloader(self) -> list[DataLoader]:
        # All models have the same dataset, so we just take the first dataloader and use the data for all models
        return next(iter(self.models.values())).val_dataloader()

    def paths_dataloader(self, paths: list[DataPath], **kwargs) -> DataLoader:
        return next(iter(self.models.values())).paths_dataloader(paths, **kwargs)

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int = None) -> dict[str, torch.Tensor]:
        fold_predictions = []
        for model in self.models.values():
            out = model.predict_step(batch)["class"]
            if self.config["model/activations"] != "sigmoid":
                out = out.softmax(dim=1)
            fold_predictions.append(out)

        # Ensembling over the softmax values (or logits in case of sigmoid)
        return {"class": torch.stack(fold_predictions).mean(dim=0)}
