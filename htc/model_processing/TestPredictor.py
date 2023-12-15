# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import multiprocessing
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from rich.progress import Progress, TimeElapsedColumn
from torch.utils.data import DataLoader
from typing_extensions import Self

from htc.model_processing.Predictor import Predictor
from htc.models.common.HTCLightning import HTCLightning
from htc.models.common.torch_helpers import move_batch_gpu
from htc.models.data.DataSpecification import DataSpecification
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.helper_functions import checkpoint_path


class TestEnsemble(nn.Module):
    def __init__(self, model_paths: list[Path], paths: Union[list[DataPath], None], config: Config):
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
                ckpt_file, dataset_train=None, datasets_val=[dataset], config=self.config
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

    def dataloader(self) -> DataLoader:
        # All models have the same dataset, so we just take the first dataloader and use the data for all models
        return next(iter(self.models.values())).val_dataloader()[0]

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int = None) -> dict[str, torch.Tensor]:
        fold_predictions = []
        for model in self.models.values():
            out = model.predict_step(batch)["class"]
            if self.config["model/activations"] != "sigmoid":
                out = out.softmax(dim=1)
            fold_predictions.append(out)

        # Ensembling over the softmax values (or logits in case of sigmoid)
        return {"class": torch.stack(fold_predictions).mean(dim=0)}


class TestPredictor(Predictor):
    def __init__(self, *args, paths: list[DataPath] = None, fold_names: list[str] = None, **kwargs):
        super().__init__(*args, **kwargs)

        if fold_names is None:
            model_paths = sorted(self.run_dir.glob("fold*"))  # All folds per default
        else:
            model_paths = [self.run_dir / f for f in fold_names]
        assert len(model_paths) > 0, "At least one fold required"

        # We do not need pretrained model during testing
        self.config["model/pretrained_model"] = None

        if paths is None:
            specs = DataSpecification(self.run_dir / "data.json")
            specs.activate_test_set()
            paths = specs.paths("^test")

        self.name_path_mapping = {p.image_name(): p for p in paths}
        self.model = TestEnsemble(model_paths, paths, self.config)
        self.model.eval()
        self.model.cuda()

    @torch.no_grad()
    def start(self, task_queue: multiprocessing.JoinableQueue, hide_progressbar: bool) -> None:
        dataloader = self.model.dataloader()

        with Progress(*Progress.get_default_columns(), TimeElapsedColumn(), disable=hide_progressbar) as progress:
            task_loader = progress.add_task("Dataloader", total=len(dataloader))

            for batch in dataloader:
                remaining_image_names = []
                for b, image_name in enumerate(batch["image_name"]):
                    predictions = self.load_predictions(image_name)
                    if predictions is not None:
                        task_queue.put({
                            "path": self.name_path_mapping[image_name],
                            "predictions": predictions,
                        })
                    else:
                        remaining_image_names.append(image_name)

                if len(remaining_image_names) > 0:
                    if not batch["features"].is_cuda:
                        batch = move_batch_gpu(batch)

                    self.produce_predictions(
                        task_queue=task_queue,
                        model=self.model,
                        batch=batch,
                        remaining_image_names=remaining_image_names,
                    )

                progress.advance(task_loader)

    def produce_predictions(
        self,
        task_queue: multiprocessing.JoinableQueue,
        model: HTCLightning,
        batch: dict[str, torch.Tensor],
        remaining_image_names: list[str],
    ) -> None:
        batch_predictions = model.predict_step(batch)["class"].cpu().numpy()

        for b in range(batch_predictions.shape[0]):
            image_name = batch["image_name"][b]
            if image_name in remaining_image_names:
                predictions = batch_predictions[b, ...]

                task_queue.put({
                    "path": self.name_path_mapping[image_name],
                    "predictions": predictions,
                })
