# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Union

import torch

from htc.model_processing.TestPredictor import TestEnsemble
from htc.models.common.HTCLightning import HTCLightning
from htc.models.common.HTCModel import HTCModel
from htc.models.common.utils import dtype_from_config
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping


class SinglePredictor:
    def __init__(
        self,
        model: str = None,
        run_folder: str = None,
        path: Union[str, Path] = None,
        fold_name: str = None,
        device: str = "cuda",
        test: bool = False,
    ) -> None:
        """
        Class which can be used to create predictions for individual samples or batches for a model.

        In contrast to the `TestPredictor` and `ValidationPredictor` classes, this class does not spawn producer-consumer processes but operates only on the main process. It is useful if predictions are only required for individual samples and not for entire datasets, if the post-processing of the prediction is very simple or if everything is done on the GPU anyway.

        Example prediction using a single model:
        >>> from htc import DataPath
        >>> print("some_log_messages"); predictor_val = SinglePredictor(model="image", run_folder="2023-02-08_14-48-02_organ_transplantation_0.8")  # doctest: +ELLIPSIS
        some_log_messages...
        >>> path = DataPath.from_image_name("P041#2019_12_14_12_29_18")
        >>> sample = torch.from_numpy(path.read_cube(normalization=1))
        >>> prediction_val = predictor_val.predict_sample(sample)["class"].argmax(dim=0).cpu()
        >>> prediction_val.shape
        torch.Size([480, 640])

        We get different results when using the test ensemble:
        >>> print("some_log_messages"); predictor_test = SinglePredictor(model="image", run_folder="2023-02-08_14-48-02_organ_transplantation_0.8", test=True)  # doctest: +ELLIPSIS
        some_log_messages...
        >>> prediction_test = predictor_test.predict_sample(sample)["class"].argmax(dim=0).cpu()
        >>> torch.any(prediction_val != prediction_test)
        tensor(True)

        It is also possible to make predictions for batches. This works easily in conjunction with a dataloader class:
        >>> from htc import DatasetImageBatch, DataSpecification
        >>> config = Config("htc/context/models/configs/organ_transplantation_0.8.json")
        >>> spec = DataSpecification("pigs_semantic-only_5foldsV2.json")
        >>> paths = spec.paths("val")
        >>> dataloader = DatasetImageBatch.batched_iteration(paths, config)
        >>> batch = next(iter(dataloader))
        >>> batch["features"].shape
        torch.Size([5, 480, 640, 100])
        >>> batch["features"].device.type  # Already on the GPU
        'cuda'
        >>> predictions_batch = predictor_val.predict_batch(batch)
        >>> predictions_batch["class"].shape
        torch.Size([5, 19, 480, 640])

        Args:
            model: Basic model type like image or pixel. Passed directly to `HTCModel.find_pretrained_run()`.
            run_folder: Name of the training run directory. Passed directly to `HTCModel.find_pretrained_run()`.
            path: Direct path to the run directory or to a fold. Passed directly to `HTCModel.find_pretrained_run()`. If the path to a fold is given (and fold_name is None), the model for this fold will be used.
            fold_name: Name of the validation fold which defines the trained network of the run. If None, the model with the highest metric score will be used.
            device: Device which is used to compute the predictions.
            test: If True, use a test ensemble for the predictions instead of individual models. Similar to the `TestPredictor` and `ValidationPredictor` classes.
        """
        self.run_dir = HTCModel.find_pretrained_run(model, run_folder, path)
        self.device = device
        self.config = Config(self.run_dir / "config.json")
        self.label_mapping = LabelMapping.from_config(self.config)
        self.features_dtype = dtype_from_config(self.config)

        if test:
            model_paths = sorted(self.run_dir.glob("fold*"))
            assert len(model_paths) > 0, "At least one fold required"
            self.model = TestEnsemble(model_paths, paths=None, config=self.config)
            self.model.eval()
            self.model.to(self.device)
        else:
            if fold_name is not None:
                # Explicit fold
                model_path = HTCModel.best_checkpoint(self.run_dir / fold_name)
            elif path is not None:
                # Whatever the user specified (explicit fold if it points to the fold directory, otherwise the best fold will be used)
                model_path = HTCModel.best_checkpoint(path)
            else:
                # Best fold
                model_path = HTCModel.best_checkpoint(self.run_dir)

            LightningClass = HTCLightning.class_from_config(self.config)
            self.model = LightningClass.load_from_checkpoint(
                model_path, dataset_train=None, datasets_val=[], config=self.config
            )
            self.model.eval()
            self.model.to(self.device)

    def predict_sample(self, sample: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute the predictions for a single sample.

        Args:
            sample: Features of the sample without the batch dimension. The remaining dimensions are defined by the lightning class (not the model class!), e.g. [H, W, C] for LightningImage.

        Returns: Dictionary with the predictions from the model for the sample (usually with a "class" key containing the class logits for the input). Does also not contain a batch dimension.
        """
        with torch.no_grad(), torch.autocast(device_type=self.device):
            batch = {"features": sample.to(dtype=self.features_dtype, device=self.device).unsqueeze(0)}
            logits = self.model.predict_step(batch)
            for key, tensor in logits.items():
                logits[key] = tensor.squeeze(dim=0)

        return logits

    def predict_batch(self, batch: Union[dict[str, torch.Tensor], torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Compute the predictions for a batch of samples.

        Args:
            batch: Dictionary with batch of features (must contain a "features" key) as returned by the dataloaders. Alternatively, supply a batch tensor directly. Either way, the first dimension must be the batch dimension and the remaining dimensions according to the lightning class (not the model class!), e.g. [B, H, W, C] for LightningImage.

        Returns: Dictionary with the predictions from the model for the batch (usually with a "class" key containing the class logits for the input). First dimension is the batch dimension.
        """
        with torch.no_grad(), torch.autocast(device_type=self.device):
            if type(batch) == dict:
                batch_predict = {"features": batch["features"].to(dtype=self.features_dtype, device=self.device)}
            else:
                batch_predict = {"features": batch.to(dtype=self.features_dtype, device=self.device)}

            logits = self.model.predict_step(batch_predict)

        return logits
