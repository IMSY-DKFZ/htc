# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import multiprocessing

import torch
from rich.progress import Progress, TimeElapsedColumn

from htc.model_processing.Predictor import Predictor
from htc.models.common.HTCLightning import HTCLightning
from htc.models.common.torch_helpers import move_batch_gpu
from htc.models.data.DataSpecification import DataSpecification
from htc.tivita.DataPath import DataPath
from htc.utils.helper_functions import checkpoint_path


class TestPredictor(Predictor):
    def __init__(self, *args, paths: list[DataPath] = None, fold_names: list[str] = None, **kwargs):
        super().__init__(*args, **kwargs)

        if fold_names is None:
            model_folds = sorted(self.run_dir.glob("fold*"))  # All folds per default
        else:
            model_folds = [self.run_dir / f for f in fold_names]
        assert len(model_folds) > 0, "At least one fold required"

        # Do not need pretrained model during testing
        self.config["model/pretrained_model"] = None

        if paths is None:
            specs = DataSpecification(self.run_dir / "data.json")
            specs.activate_test_set()
            paths = specs.paths("^test")

        # Load models from all folds (we'll do ensembling later)
        self.models = {}
        for fold_dir in model_folds:
            ckpt_file, _ = checkpoint_path(fold_dir)

            # Load dataset and lightning class based on model name
            LightningClass = HTCLightning.class_from_config(self.config)
            dataset = LightningClass.dataset(paths=paths, train=False, config=self.config, fold_name=fold_dir.stem)
            model = LightningClass.load_from_checkpoint(
                ckpt_file, dataset_train=None, datasets_val=[dataset], config=self.config
            )
            model.eval()
            model.cuda()

            self.models[fold_dir] = model

    @torch.no_grad()
    def start(self, task_queue: multiprocessing.JoinableQueue, hide_progressbar: bool) -> None:
        dataloader = next(iter(self.models.values())).val_dataloader()[
            0
        ]  # All models have the same dataset, so we just take the first dataloader and use the data for all models
        with Progress(*Progress.get_default_columns(), TimeElapsedColumn(), disable=hide_progressbar) as progress:
            task_loader = progress.add_task("Dataloader", total=len(dataloader))

            for batch in dataloader:
                remaining_image_names = []
                for b, image_name in enumerate(batch["image_name"]):
                    predictions = self.load_predictions(image_name)
                    if predictions is not None:
                        task_queue.put(
                            {
                                "image_name": image_name,
                                "predictions": predictions,
                            }
                        )
                    else:
                        remaining_image_names.append(image_name)

                if len(remaining_image_names) > 0:
                    if not batch["features"].is_cuda:
                        batch = move_batch_gpu(batch)

                    fold_predictions = []
                    for model in self.models.values():
                        fold_predictions.append(model.predict_step(batch)["class"].softmax(dim=1))

                    batch_predictions = (
                        torch.stack(fold_predictions).mean(dim=0).cpu().numpy()
                    )  # Ensembling over the softmax values
                    for b in range(batch_predictions.shape[0]):
                        image_name = batch["image_name"][b]
                        if image_name in remaining_image_names:
                            predictions = batch_predictions[b, ...]

                            task_queue.put(
                                {
                                    "image_name": image_name,
                                    "predictions": predictions,
                                }
                            )

                progress.advance(task_loader)
