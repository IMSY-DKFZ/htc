# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch
import torch.multiprocessing as multiprocessing
from rich.progress import Progress, TimeElapsedColumn

from htc.model_processing.Predictor import Predictor
from htc.model_processing.TestEnsemble import TestEnsemble
from htc.models.common.HTCLightning import HTCLightning
from htc.models.common.torch_helpers import move_batch_gpu
from htc.tivita.DataPath import DataPath


class TestPredictor(Predictor):
    def __init__(self, *args, paths: list[DataPath], fold_names: list[str] = None, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(self.run_dir, list):
            model_paths = []
            for r in self.run_dir:
                if fold_names is None:
                    model_paths += sorted(r.glob("fold*"))  # All folds per default
                else:
                    model_paths += [r / f for f in fold_names]
        else:
            if fold_names is None:
                model_paths = sorted(self.run_dir.glob("fold*"))  # All folds per default
            else:
                model_paths = [self.run_dir / f for f in fold_names]
        assert len(model_paths) > 0, "At least one fold required"

        # We do not need pretrained model during testing
        self.config["model/pretrained_model"] = None

        self.name_path_mapping = {p.image_name(): p for p in paths}
        self.model = TestEnsemble(model_paths, paths, self.config)
        self.model.eval()
        self.model.cuda()

    @torch.no_grad()
    def start(self, task_queue: multiprocessing.JoinableQueue, hide_progressbar: bool) -> None:
        dataloader = self.model.val_dataloader()[0]

        with Progress(*Progress.get_default_columns(), TimeElapsedColumn(), disable=hide_progressbar) as progress:
            task_loader = progress.add_task("Dataloader", total=len(dataloader))

            for batch in dataloader:
                remaining_image_names = []
                for image_name in batch["image_name"]:
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
        batch_predictions_gpu = model.predict_step(batch)["class"]

        # The following approach of creating an empty tensor first, making it shared (which includes copying the data) and then transferring the data from the GPU to the CPU is slightly advantageous compared to .cpu().share_memory_() because the copying can happen while the GPU is still doing the inference (since CUDA is asynchronous)
        # .cpu() has an implicit synchronization point so the copy process introduced by share_memory_() would always happens after the inference
        batch_predictions = torch.empty(dtype=batch_predictions_gpu.dtype, size=batch_predictions_gpu.size())
        # Unfortunately, this involves an unnecessary copy but there is no direct way to create an empty, shared tensor in Pytorch
        batch_predictions.share_memory_()
        batch_predictions.copy_(batch_predictions_gpu)

        for b in range(batch_predictions.shape[0]):
            image_name = batch["image_name"][b]
            if image_name in remaining_image_names:
                predictions = batch_predictions[b, ...]

                task_queue.put({
                    "path": self.name_path_mapping[image_name],
                    "predictions": predictions,
                })
