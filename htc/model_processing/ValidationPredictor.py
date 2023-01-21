# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import multiprocessing

import torch
from rich.progress import Progress, TimeElapsedColumn

from htc.model_processing.Predictor import Predictor
from htc.models.common.HTCLightning import HTCLightning
from htc.models.common.torch_helpers import move_batch_gpu
from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings
from htc.utils.helper_functions import checkpoint_path


class ValidationPredictor(Predictor):
    @torch.no_grad()
    def start(self, task_queue: multiprocessing.JoinableQueue, hide_progressbar: bool) -> None:
        settings.log.info(f"Working on run dir: {self.run_dir}")

        specs = DataSpecification(self.run_dir / "data.json")
        LightningClass = HTCLightning.class_from_config(self.config)

        with Progress(*Progress.get_default_columns(), TimeElapsedColumn(), disable=hide_progressbar) as progress:
            fold_dirs = sorted(self.run_dir.glob("fold*"))
            task_folds = progress.add_task("Folds", total=len(fold_dirs))

            for fold_dir in fold_dirs:
                ckpt_file, _ = checkpoint_path(fold_dir)

                # All paths for the respective fold
                fold_data = specs.folds[fold_dir.name]
                split_name = [name for name in fold_data.keys() if name.startswith("val")][0]  # Only the first dataset
                assert not split_name.endswith("_known"), "Predictions should not be done on the known dataset"

                if (
                    specs.table()
                    .query("split_name == @split_name")
                    .duplicated(subset=["split_name", "image_name"])
                    .any()
                ):
                    assert not self.use_predictions and not self.store_predictions, (
                        "Found duplicate image_names across folds. Neither the --use-predictions nor the"
                        " --store-predictions switch is allowed to be set. Otherwise, subsequent runs of the same"
                        " script would only ever use the predictions of the first fold"
                    )

                paths = fold_data[split_name]
                dataset = LightningClass.dataset(paths=paths, train=False, config=self.config, fold_name=fold_dir.stem)

                # Load dataset and lightning class based on model name
                model = LightningClass.load_from_checkpoint(
                    ckpt_file, dataset_train=None, datasets_val=[dataset], config=self.config
                )
                model.eval()
                model.cuda()
                dataloader = model.val_dataloader()[0]
                task_loader = progress.add_task(f"Dataloader (fold={fold_dir.name})", total=len(dataloader))

                for batch in dataloader:
                    remaining_image_names = []
                    for b, image_name in enumerate(batch["image_name"]):
                        predictions = self.load_predictions(image_name)
                        if predictions is not None:
                            task_queue.put(
                                {
                                    "image_name": image_name,
                                    "fold_name": fold_dir.name,
                                    "predictions": predictions,
                                }
                            )
                        else:
                            remaining_image_names.append(image_name)

                    # It is quite unusual that in one sample some images are already predicted and some not, so we just predict them all and yield only new samples
                    if len(remaining_image_names) > 0:
                        if not batch["labels"].is_cuda:
                            batch = move_batch_gpu(batch)

                        batch_predictions = model.predict_step(batch)["class"].softmax(dim=1).cpu().numpy()

                        for b in range(batch_predictions.shape[0]):
                            image_name = batch["image_name"][b]
                            if image_name in remaining_image_names:
                                predictions = batch_predictions[b, ...]

                                task_queue.put(
                                    {
                                        "image_name": image_name,
                                        "fold_name": fold_dir.name,
                                        "predictions": predictions,
                                    }
                                )

                    progress.advance(task_loader)
                progress.advance(task_folds)
