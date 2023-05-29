# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import multiprocessing
from contextlib import ExitStack
from typing import Union

import torch
from rich.progress import Progress, TimeElapsedColumn

from htc.model_processing.Predictor import Predictor
from htc.models.common.HTCLightning import HTCLightning
from htc.models.common.torch_helpers import move_batch_gpu
from htc.models.data.DataSpecification import DataSpecification
from htc.tivita.DataPath import DataPath
from htc.utils.helper_functions import checkpoint_path


class TestLeaveOneOutPredictor(Predictor):
    def __init__(
        self,
        *args,
        paths: Union[list[DataPath], dict[str, list[DataPath]]] = None,
        fold_names: list[str] = None,
        outputs: list[str] = None,
        **kwargs,
    ):
        """Compared to the TestPredictor, this class expects a leave-one-out structure of the test set and hence does not use ensembling."""
        super().__init__(*args, **kwargs)
        if outputs is None:
            outputs = ["predictions"]
        self.outputs = outputs
        assert len(self.outputs), "At least one output should be provided"
        self.feature_names = [name for name in self.outputs if name != "predictions"]

        if fold_names is None:
            folds = sorted(self.run_dir.glob("fold*"))  # All folds per default
        else:
            folds = [self.run_dir / f for f in fold_names]
        assert len(folds) > 0, "At least one fold required"

        specs = DataSpecification(self.run_dir / "data.json")
        specs.activate_test_set()

        folds_paths = []
        for fold_dir in folds:
            if paths is None:
                fold_data = specs.folds[fold_dir.name]
                dataset_name = [name for name in fold_data.keys() if name.startswith("test")]
                assert len(dataset_name) == 1, "Not exactly one test dataset found for this fold!"
                dataset_name = dataset_name[0]

                folds_paths.append(fold_data[dataset_name])
            else:
                if type(paths) == list:
                    folds_paths.append(paths)
                elif type(paths) == dict:
                    folds_paths.append(paths[fold_dir.name])
                else:
                    raise ValueError(f"Invalid type {type(paths)} for the paths argument")

        self.models = {}
        for fold_dir, fold_paths in zip(folds, folds_paths):
            ckpt_file, _ = checkpoint_path(fold_dir)

            # Load dataset and lightning class based on model name
            LightningClass = HTCLightning.class_from_config(self.config)
            dataset = LightningClass.dataset(paths=fold_paths, train=False, config=self.config, fold_name=fold_dir.stem)
            model = LightningClass.load_from_checkpoint(
                ckpt_file, dataset_train=None, datasets_val=[dataset], config=self.config
            )
            model.eval()
            model.cuda()

            self.models[fold_dir] = model
            self.name_path_mapping |= {p.image_name(): p for p in fold_paths}

    @torch.no_grad()
    def start(self, task_queue: multiprocessing.JoinableQueue, hide_progressbar: bool) -> None:
        with Progress(*Progress.get_default_columns(), TimeElapsedColumn(), disable=hide_progressbar) as progress:
            task_models = progress.add_task("Models", total=len(self.models))

            for fold_dir, model in self.models.items():
                dataloader = model.val_dataloader()[0]
                task_loader = progress.add_task(f"Dataloader (fold={fold_dir.name})", total=len(dataloader))

                for batch in dataloader:
                    remaining_image_names = []
                    for b, image_name in enumerate(batch["image_name"]):
                        predictions = self.load_predictions(image_name)
                        if predictions is not None:
                            task_queue.put(
                                {
                                    "path": self.name_path_mapping[image_name],
                                    "fold_name": fold_dir.name,
                                    "predictions": predictions,
                                }
                            )
                        else:
                            remaining_image_names.append(image_name)

                    if len(remaining_image_names) > 0:
                        if not batch["labels"].is_cuda:
                            batch = move_batch_gpu(batch)

                        with ExitStack() as stack:
                            promises = {
                                name: stack.enter_context(model.model.features(name)) for name in self.feature_names
                            }
                            batch_predictions = model.predict_step(batch)["class"]
                            features = {name: promises[name].data() for name in self.feature_names}

                        batch_outputs = {}
                        if "predictions" in self.outputs:
                            batch_outputs["predictions"] = batch_predictions.softmax(dim=1).cpu().numpy()
                        else:
                            # Ensure that we only read the features after the inference finished
                            torch.cuda.synchronize()

                        for name in self.feature_names:
                            batch_outputs[name] = features[name].cpu().numpy()

                        for b in range(batch["features"].size(0)):
                            image_name = batch["image_name"][b]
                            if image_name in remaining_image_names:
                                data = {
                                    "path": self.name_path_mapping[image_name],
                                    "fold_name": fold_dir.name,
                                }
                                for name in self.outputs:
                                    data[name] = batch_outputs[name][b, ...]

                                task_queue.put(data)

                    progress.advance(task_loader)
                progress.advance(task_models)
