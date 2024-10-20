# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import gc

import torch
import torch.multiprocessing as multiprocessing
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

        spec = DataSpecification.from_config(self.config)
        LightningClass = HTCLightning.class_from_config(self.config)

        with Progress(*Progress.get_default_columns(), TimeElapsedColumn(), disable=hide_progressbar) as progress:
            fold_dirs = sorted(self.run_dir.glob("fold*"))
            task_folds = progress.add_task("Folds", total=len(fold_dirs))

            for fold_dir in fold_dirs:
                ckpt_file, best_epoch_index = checkpoint_path(fold_dir)

                # All paths for the respective fold
                fold_data = spec.folds[fold_dir.name]
                split_name = next(name for name in fold_data.keys() if name.startswith("val"))  # Only the first dataset
                assert not split_name.endswith("_known"), "Predictions should not be done on the known dataset"

                if (
                    spec.table()
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
                self.name_path_mapping = {p.image_name(): p for p in paths}
                dataset = LightningClass.dataset(paths=paths, train=False, config=self.config, fold_name=fold_dir.stem)

                # Load dataset and lightning class based on model name
                model = LightningClass.load_from_checkpoint(
                    ckpt_file, dataset_train=None, datasets_val=[dataset], config=self.config, fold_name=fold_dir.stem
                )
                model.eval()
                model.cuda()
                dataloader = model.val_dataloader()[0]
                task_loader = progress.add_task(f"Dataloader (fold={fold_dir.name})", total=len(dataloader))

                for batch in dataloader:
                    remaining_image_names = []
                    for image_name in batch["image_name"]:
                        predictions = self.load_predictions(image_name)
                        if predictions is not None:
                            task_queue.put({
                                "path": self.name_path_mapping[image_name],
                                "fold_name": fold_dir.name,
                                "best_epoch_index": best_epoch_index,
                                "predictions": predictions,
                            })
                        else:
                            remaining_image_names.append(image_name)

                    # It is quite unusual that in one sample some images are already predicted and some not, so we just predict them all and yield only new samples
                    if len(remaining_image_names) > 0:
                        if not batch["features"].is_cuda:
                            batch = move_batch_gpu(batch)

                        self.produce_predictions(
                            task_queue=task_queue,
                            model=model,
                            batch=batch,
                            remaining_image_names=remaining_image_names,
                            fold_name=fold_dir.name,
                            best_epoch_index=best_epoch_index,
                        )

                    progress.advance(task_loader)

                # Make sure the shared memory buffer is cleared before the next iteration
                del dataloader
                gc.collect()

                progress.advance(task_folds)

    def produce_predictions(
        self,
        task_queue: multiprocessing.JoinableQueue,
        model: HTCLightning,
        batch: dict[str, torch.Tensor],
        remaining_image_names: list[str],
        fold_name: str,
        best_epoch_index: int,
    ) -> None:
        batch_predictions_gpu = model.predict_step(batch)["class"]
        if self.config["model/activations"] != "sigmoid":
            # For the sigmoid activation, we need to pass the logits
            batch_predictions_gpu = batch_predictions_gpu.softmax(dim=1)

        batch_predictions = torch.empty(dtype=batch_predictions_gpu.dtype, size=batch_predictions_gpu.size())
        batch_predictions.share_memory_()
        batch_predictions.copy_(batch_predictions_gpu)

        for b in range(batch_predictions.shape[0]):
            image_name = batch["image_name"][b]
            if image_name in remaining_image_names:
                predictions = batch_predictions[b, ...]

                task_queue.put({
                    "path": self.name_path_mapping[image_name],
                    "fold_name": fold_name,
                    "best_epoch_index": best_epoch_index,
                    "predictions": predictions,
                })
