# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import copy
from functools import partial
from pathlib import Path

import torch
from rich.progress import track

from htc.models.common.HTCLightning import HTCLightning
from htc.models.common.HTCModel import HTCModel
from htc.models.common.torch_helpers import move_batch_gpu
from htc.models.data.DataSpecification import DataSpecification
from htc.utils.Config import Config
from htc.utils.import_extra import requires_extra

try:
    from captum.attr import IntegratedGradients

    _missing_library = ""
except ImportError:
    _missing_library = "captum"


class FeatureImportance:
    @requires_extra(_missing_library)
    def __init__(
        self,
        model: str = None,
        run_folder: str = None,
        path: str | Path = None,
        device: str = "cuda",
    ) -> None:
        self.run_dir = HTCModel.find_pretrained_run(model, run_folder, path)
        self.device = device
        self.config = Config(self.run_dir / "config.json")
        self.spec = DataSpecification.from_config(self.config)
        self.input_names = ["features"]
        if "meta" in self.run_dir.name:
            self.input_names.append("meta")

    def compute_attributions(self, target_index: int, batch_size: int = None) -> dict[str, dict[str, torch.Tensor]]:
        config = copy.copy(self.config)
        if batch_size is not None:
            config["dataloader_kwargs/batch_size"] = batch_size

        fold_maps = {n: [] for n in self.input_names}
        for fold_name in track(self.spec.fold_names(), description="Folds..."):
            model_path = HTCModel.best_checkpoint(self.run_dir / fold_name)
            LightningClass = HTCLightning.class_from_config(config)

            paths = self.spec.fold_paths(fold_name=fold_name, split_name="^val$")
            dataset = LightningClass.dataset(paths=paths, train=False, config=config, fold_name=fold_name)
            model = LightningClass.load_from_checkpoint(
                model_path, dataset_train=None, datasets_val=[dataset], config=config
            )
            model.eval()
            model.to(self.device)

            dataloader = model.val_dataloader()[0]

            def model_helper(*args: tuple[torch.Tensor], names: list[str]) -> torch.Tensor:
                model_input = {name: arg.requires_grad_() for name, arg in zip(names, args, strict=True)}

                return model(model_input)

            maps = {n: [] for n in self.input_names}
            with torch.autocast(device_type=self.device):
                ig = IntegratedGradients(partial(model_helper, names=self.input_names))
                for batch in dataloader:
                    if not batch[self.input_names[0]].is_cuda:
                        batch = move_batch_gpu(batch)

                    attributions = ig.attribute(tuple([batch[n] for n in self.input_names]), target=target_index)
                    attributions = dict(zip(self.input_names, attributions, strict=True))

                    for n in self.input_names:
                        if n == "meta":
                            # attribution_meta.shape = [B, M]
                            attributions[n] = attributions[n].mean(dim=0)
                        elif n == "features":
                            # attribution_features.shape = [B, H, W, C]
                            attributions[n][batch[n] == 0] = torch.nan
                            attributions[n] = attributions[n].nanmean(dim=(0, 1, 2))
                        else:
                            raise ValueError(f"Unknown input name: {n}")

                        maps[n].append(attributions[n].cpu().detach())

            for n in self.input_names:
                # Average over samples
                fold_maps[n].append(torch.stack(maps[n]).mean(dim=0))

        fold_maps = {key: torch.stack(value) for key, value in fold_maps.items()}
        return fold_maps
