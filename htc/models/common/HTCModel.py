# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import hashlib
import inspect
import itertools
import re
from pathlib import Path
from types import MappingProxyType
from typing import Self
from zipfile import ZipFile

import pandas as pd
import torch
import torch.nn as nn

from htc.models.common.MetricAggregation import MetricAggregation
from htc.settings import settings
from htc.utils.Config import Config
from htc.utils.general import sha256_file
from htc.utils.helper_functions import run_info


class PostInitCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post__init__()
        return obj


class HTCModel(nn.Module, metaclass=PostInitCaller):
    # The models will appear in reverse order in the public README

    # Models from our MIA2022 paper
    known_models = MappingProxyType({
        "pixel@2022-02-03_22-58-44_generated_default_rgb_model_comparison": {
            "sha256": "93f359368e06e238d76351e0f9ad8818f15b7e75253051d0b48feeb153ead02f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/pixel@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip",
        },
        "pixel@2022-02-03_22-58-44_generated_default_parameters_model_comparison": {
            "sha256": "3d0360f28b4e3cb1c9120521cb88742ac8452ca3184bce5d95ff1b4b9dc8e489",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/pixel@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip",
        },
        "pixel@2022-02-03_22-58-44_generated_default_model_comparison": {
            "sha256": "f8d996d352a70edf6758903ecf678566154833b0e6c5bebbdfd6653a4abe17c4",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/pixel@2022-02-03_22-58-44_generated_default_model_comparison.zip",
        },
        "superpixel_classification@2022-02-03_22-58-44_generated_default_rgb_model_comparison": {
            "sha256": "56396eddd460524702b940c4aabfe470fbacbb6885a89b71441b8002bcb7a696",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/superpixel_classification@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip",
        },
        "superpixel_classification@2022-02-03_22-58-44_generated_default_parameters_model_comparison": {
            "sha256": "87a292b909a6f27a55965f2be42cc183332d91a634e23398930d3930ee3152f6",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/superpixel_classification@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip",
        },
        "superpixel_classification@2022-02-03_22-58-44_generated_default_model_comparison": {
            "sha256": "a27a0ed8a37b20663fc1fbf2ffe9e83a84fca0c5afb8c813974377b318f6e6ad",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/superpixel_classification@2022-02-03_22-58-44_generated_default_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_rgb_model_comparison": {
            "sha256": "d67353936110f4ace2ad73f704267d7f6cec9d1fff8df64029f9517fb28cca11",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_parameters_model_comparison": {
            "sha256": "e66328c9d3166a6fdded6bdc832cd882cacbc7faf98058b2b777831d420eb2b7",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_model_comparison": {
            "sha256": "b0a8f23a52f80b1277ec086b9938463f95c6651948eb8913dd1e7633a4bdc46e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_64_rgb_model_comparison": {
            "sha256": "e1d5c78a9d584b71a6353b618d435e1009eea2144590b2ba7045e55517ce16dd",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_64_rgb_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_64_parameters_model_comparison": {
            "sha256": "b6b086f9ef7014b6529d64be29b817c39e9a4db211de712db10f97220c034175",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_64_parameters_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_64_model_comparison": {
            "sha256": "ea4e0708e7b053e29797b0e607485de74499b7bcb144a68de3937fc8ca95b5c3",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_64_model_comparison.zip",
        },
        "image@2022-02-03_22-58-44_generated_default_rgb_model_comparison": {
            "sha256": "ee589c35cd9aa9fb201cc12e74a9133f1eb822b220a72b3252624f1c4402b636",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip",
        },
        "image@2022-02-03_22-58-44_generated_default_parameters_model_comparison": {
            "sha256": "d21c983dadd13aafab78e3c7dbad5ecfcaa362d5aa2a61833847c12c434eec65",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip",
        },
        "image@2022-02-03_22-58-44_generated_default_model_comparison": {
            "sha256": "c913639483efbf5536e46aea914099f15ae6a80ad21e5dd101c59ee55ee5c658",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2022-02-03_22-58-44_generated_default_model_comparison.zip",
        },
        # Models from our MICCAI2023 paper
        "image@2023-01-29_11-31-04_organ_transplantation_0.8_rgb": {
            "sha256": "9e322a8f13e3331a4d86b96eff63b2879b7d459c35b3894388b201d227a72986",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2023-01-29_11-31-04_organ_transplantation_0.8_rgb.zip",
        },
        "image@2023-02-08_14-48-02_organ_transplantation_0.8": {
            "sha256": "503258ade835f742759bafa63c50fa0925074189b762bfb391b4a2fecd4ff433",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2023-02-08_14-48-02_organ_transplantation_0.8.zip",
        },
        # Models from our XenoLearning2024 paper
        "image@2024-09-11_00-11-38_baseline_human_nested-0-2": {
            "sha256": "da264251dfebc7d4d4d2821a436e57682300fa0c315887907868d8245d33f920",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_baseline_human_nested-0-2.zip",
        },
        "image@2024-09-11_00-11-38_baseline_human_nested-1-2": {
            "sha256": "8a8c14c85fd97582e092e5df9d8e29465fb7a81ad8e2af040838859509f88dd9",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_baseline_human_nested-1-2.zip",
        },
        "image@2024-09-11_00-11-38_baseline_human_nested-2-2": {
            "sha256": "9898d9547e9b41986b4c00787cb4031b1be96d42350465be16ddb18c9ad7c25f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_baseline_human_nested-2-2.zip",
        },
        "image@2024-09-11_00-11-38_baseline_pig_nested-0-2": {
            "sha256": "39fe496587f265c00037d9827893a1ed6ef64d24800ece087476b2e437f6978e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_baseline_pig_nested-0-2.zip",
        },
        "image@2024-09-11_00-11-38_baseline_pig_nested-1-2": {
            "sha256": "6102a76919ce3e8ba0d4eb997a108266bb04e0ecec5d94447e2fab3c0761bb50",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_baseline_pig_nested-1-2.zip",
        },
        "image@2024-09-11_00-11-38_baseline_pig_nested-2-2": {
            "sha256": "11513635788d9307071478c2cf9a460aa07b638168eaa52da0d2d1979db10a3f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_baseline_pig_nested-2-2.zip",
        },
        "image@2024-09-11_00-11-38_baseline_rat_nested-0-2": {
            "sha256": "88cf57df0f3aeef93826d8f253be86dc366b5d83cc2848d4ce851f1e799c0dfa",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_baseline_rat_nested-0-2.zip",
        },
        "image@2024-09-11_00-11-38_baseline_rat_nested-1-2": {
            "sha256": "f150b12447ef8277423615245cf5fdf9eca4247dee0427dcd6643a4c35f09a9f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_baseline_rat_nested-1-2.zip",
        },
        "image@2024-09-11_00-11-38_baseline_rat_nested-2-2": {
            "sha256": "745eee1ebddebaaa132ce751a5762d226c1ce349431800272f96442b265db424",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_baseline_rat_nested-2-2.zip",
        },
        "image@2024-09-11_00-11-38_joint_pig-p+rat-p2human_nested-0-2": {
            "sha256": "8a1f6f31f2dbcc227b488584afb94486a4f3b335949c27e947982256bf20fda4",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_joint_pig-p+rat-p2human_nested-0-2.zip",
        },
        "image@2024-09-11_00-11-38_joint_pig-p+rat-p2human_nested-1-2": {
            "sha256": "2be564e06821e3a7618f0050c4a4711fc16d92be0faffec453a1b5900b120cdd",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_joint_pig-p+rat-p2human_nested-1-2.zip",
        },
        "image@2024-09-11_00-11-38_joint_pig-p+rat-p2human_nested-2-2": {
            "sha256": "5f21bc39468b618369ef8bf8a6c6a7ec036883d9b16eb54c446f733b3a628443",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_joint_pig-p+rat-p2human_nested-2-2.zip",
        },
        "image@2024-09-11_00-11-38_projected_pig2human_nested-0-2": {
            "sha256": "05ef15aee67d28f5bbb95554c90ff91c642871b6e2f3376a70e3a9acd4b274a5",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_pig2human_nested-0-2.zip",
        },
        "image@2024-09-11_00-11-38_projected_pig2human_nested-1-2": {
            "sha256": "480b40ed313f9af8a8c82abc156ae733681ba33b7f74553e4edc8dd05b987ff5",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_pig2human_nested-1-2.zip",
        },
        "image@2024-09-11_00-11-38_projected_pig2human_nested-2-2": {
            "sha256": "ad04b577a8b629279b50ffd0c000ad49218c20bf5dcedf604a4ae99b56b0ef31",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_pig2human_nested-2-2.zip",
        },
        "image@2024-09-11_00-11-38_projected_pig2rat_nested-0-2": {
            "sha256": "b50f7d0c26b15209ee0faa6bc2c6e1a40d04e0edad5aa3c25cc9e9cda70b4a4c",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_pig2rat_nested-0-2.zip",
        },
        "image@2024-09-11_00-11-38_projected_pig2rat_nested-1-2": {
            "sha256": "88b0090d0a7359d3f02ae0d7656e490679d46eb9852c316c076d917a071eef9c",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_pig2rat_nested-1-2.zip",
        },
        "image@2024-09-11_00-11-38_projected_pig2rat_nested-2-2": {
            "sha256": "dbc39220341747a44e810163f213d21f19e9d4974590ee5329f964bfb8bc2616",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_pig2rat_nested-2-2.zip",
        },
        "image@2024-09-11_00-11-38_projected_rat2human_nested-0-2": {
            "sha256": "bc25a6ad74abf647b31f057b23592d0fd387a7207888f9c5145e1e91a3159659",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_rat2human_nested-0-2.zip",
        },
        "image@2024-09-11_00-11-38_projected_rat2human_nested-1-2": {
            "sha256": "117504771dd2d7fe6fdf37bf88a5127c8f28bdb7517adb493fa04caaf1be1a37",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_rat2human_nested-1-2.zip",
        },
        "image@2024-09-11_00-11-38_projected_rat2human_nested-2-2": {
            "sha256": "ccc8210d2ca13e3bdfc0f810420bd936bafe49ebe4f3d9c4dc525703ac1044f2",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_rat2human_nested-2-2.zip",
        },
        "image@2024-09-11_00-11-38_projected_rat2pig_nested-0-2": {
            "sha256": "400e2eca725011e38d271206f5070572926f7bb6bf4f22634dafd4f37535b240",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_rat2pig_nested-0-2.zip",
        },
        "image@2024-09-11_00-11-38_projected_rat2pig_nested-1-2": {
            "sha256": "50987cb1d8f20cdb504cb10cf35331c04ccca289a6d12cb95c656424d63c4fa9",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_rat2pig_nested-1-2.zip",
        },
        "image@2024-09-11_00-11-38_projected_rat2pig_nested-2-2": {
            "sha256": "73123ed6703071d1b9d9a9d1da97f6b9a571365bbe8342569ad9642e5af69a96",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_rat2pig_nested-2-2.zip",
        },
    })

    def __init__(self, config: Config, fold_name: str = None):
        """
        Base class for all model classes. It implements some logic to automatically replace the weights of a network with the weights from another pretrained network.

        If a model class inherits from this class it becomes very easy to use the same model class again with pretrained weights. You only need to specify the pretrained network in the config and then the weights get replaced automatically. In the config, you have two options to reference the pretrained network (see also the config.schema file):
        - With its path (absolute or relative to the training directory): `config["model/pretrained_model/path"] = "image/my_training_run"`
        - With a dictionary of the relevant properties: `config["model/pretrained_model"] = {"model": "image", "run_folder": "my_training_run"}`

        In case more than one fold is found in the run folder, the one with the highest score will be used per default. Alternatively, you can also specify the fold explicitly (either by appending it to the path or via the `model/pretrained_model/fold_name` config attribute).

        If the pretrained network is set in either way, the `_load_pretrained_model()` method of this class gets called at the end of your `__init__` (via the `__post__init__` method). Usually, we don't want all the weights of the pretrained network, for example, the segmentation head is normally not useful if the number of classes in the new task is different from the number of classes of the pretrained task. This is why those weights are skipped by default. You can control what should be skipped in your model by altering the `skip_keys_pattern` attribute of this class:
        >>> class MyModel(HTCModel):
        ...     def __init__(self, config: Config):
        ...         super().__init__(config)
        ...         self.skip_keys_pattern.add("decoder")
        >>> my_model = MyModel(Config({}))
        >>> sorted(my_model.skip_keys_pattern)
        ['classification_head', 'decoder', 'heads.heads', 'segmentation_head']

        Args:
            config: Configuration object of the training run.
            fold_name: the name of the fold being trained. This is used to find the correct parameter for temperature scaling.
        """
        super().__init__()
        self.config = config
        self.fold_name = fold_name

        # Default keys to load/skip for pretraining
        # Subclasses can modify these sets by adding elements to it or replacing them
        # If the key starts with "model.", then this usually corresponds to the name of the attribute in the lightning class
        # load_keys_pattern defines a dictionary with search and replace rules
        self.load_keys_pattern = {"model.": ""}
        # If any of the following names occurs in the state dict, they will be skipped
        self.skip_keys_pattern = {"segmentation_head", "classification_head", "heads.heads"}

        # Check for every model once whether the input is properly L1 normalized
        self._normalization_handle = self.register_forward_pre_hook(self._normalization_check)

    def __post__init__(self):
        if self.config["model/pretrained_model"]:
            self._load_pretrained_model()

        # We initialize temperature scaling only after the pretrained model may be loaded because that could change the fold name
        if self.config.get("post_processing/calibration") is not None:
            factors = self.config["post_processing/calibration/scaling"]
            biases = self.config["post_processing/calibration/bias"]
            if self.fold_name not in factors or self.fold_name not in biases:
                settings.log.warning(
                    "Found temperature scaling parameters in the config but not for the requested fold"
                    f" {self.fold_name} (temperature scaling will not be applied)"
                )
            else:
                self.register_buffer("_temp_factor", torch.tensor(factors[self.fold_name]), persistent=False)
                self.register_buffer("_temp_bias", torch.tensor(biases[self.fold_name]), persistent=False)
                if (nll_prior := self.config.get("post_processing/calibration/nll_prior")) is not None:
                    self.register_buffer("_nll_prior", torch.tensor(nll_prior), persistent=False)
                else:
                    self._nll_prior = None
                self.register_forward_hook(self._temperature_scaling)

    def _temperature_scaling(
        self, module: nn.Module, module_in: tuple, output: torch.Tensor | dict[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if type(output) == dict:
            logits = output["class"]
        else:
            logits = output

        # Actual scaling (based on: https://github.com/luferrer/psr-calibration/blob/master/psrcal/calibration.py)
        scores = logits * self._temp_factor + self._temp_bias
        if self._nll_prior is not None:
            scores = scores - self._nll_prior  # - because nll = -log(prior)

        scores = scores - torch.logsumexp(scores, axis=-1, keepdim=True)

        if type(output) == dict:
            output["class"] = scores
        else:
            output = scores

        return output

    def _normalization_check(self, module: nn.Module, module_in: tuple) -> None:
        if self.config["input/n_channels"] == 100 and (
            self.config["input/preprocessing"] == "L1" or self.config["input/normalization"] == "L1"
        ):
            features = module_in[0]

            # Find the channel dimension
            channel_dim = None
            for dim, length in enumerate(features.shape):
                if length == self.config["input/n_channels"]:
                    channel_dim = dim
                    break

            # It is possible that we cannot find the channel dimensions, e.g. for the pixel model if an input != 100 is passed to the model
            # It is also ok if a spectrum only contains zeros since this is done by some augmentations
            if channel_dim is not None:
                channel_sum = features.abs().sum(dim=channel_dim)
                # Either all values must be close to 1 (or 0)
                all_valid = torch.all(
                    torch.isclose(channel_sum, torch.tensor(1.0, device=features.device), atol=0.1)
                    | torch.isclose(channel_sum, torch.tensor(0.0, device=features.device), atol=0.1)
                )

                # Or the mean/std must fit on average for the non-zero elements (because single pixels may be off)
                nonzeros = channel_sum[channel_sum.nonzero(as_tuple=True)]
                if nonzeros.nelement() == 0:
                    mean = torch.tensor(torch.nan, device=features.device)
                    std = torch.tensor(torch.nan, device=features.device)
                    average_valid = False
                else:
                    mean = nonzeros.mean()
                    std = nonzeros.std(correction=0)  # We may encounter single-element tensors here
                    average_valid = torch.isclose(
                        mean, torch.tensor(1.0, device=features.device), atol=0.01
                    ) and torch.isclose(std, torch.tensor(0.0, device=features.device), atol=0.01)

                if not (all_valid or average_valid):
                    settings.log.warning(
                        f"The model {module.__class__.__name__} expects L1 normalized input but the features"
                        f" ({features.shape = }) do not seem to be L1 normalized:\naverage per pixel ="
                        f" {mean}\nstandard deviation per pixel ="
                        f" {std}\nThis check is only performed for the first batch."
                    )

        # We only perform this check for the first batch
        self._normalization_handle.remove()

    def _load_pretrained_model(self) -> None:
        model_path = None
        pretrained_dir = None
        map_location = None if torch.cuda.is_available() else "cpu"

        if self.config["model/pretrained_model/path"]:
            possible_locations = HTCModel._get_possible_locations(Path(self.config["model/pretrained_model/path"]))
            for location in possible_locations:
                if location.is_dir():
                    pretrained_dir = location
                elif location.is_file():
                    model_path = location
                    break
        else:
            pretrained_dir = HTCModel.find_pretrained_run(
                self.config["model/pretrained_model/model"],
                self.config["model/pretrained_model/run_folder"],
            )
        assert pretrained_dir is not None or model_path is not None, (
            f"Could not find the pretrained model as specified in the config: {self.config['model/pretrained_model']}"
        )

        if model_path is None:
            if self.config["model/pretrained_model/fold_name"]:
                if pretrained_dir.name.startswith("fold"):
                    assert pretrained_dir.name == self.config["model/pretrained_model/fold_name"], (
                        f"The found pretrained directory {pretrained_dir} does not match the fold name"
                        f" {self.config['model/pretrained_model/fold_name']}"
                    )
                else:
                    pretrained_dir = pretrained_dir / self.config["model/pretrained_model/fold_name"]

            model_path = HTCModel.best_checkpoint(pretrained_dir)

        assert model_path is not None, "Could not find the best model"
        pretrained_model = torch.load(model_path, map_location=map_location)

        # Change state dict keys
        model_dict = self.state_dict()
        num_keys_loaded = 0
        skipped_keys = []

        for k in pretrained_model["state_dict"].keys():
            if any(skip_key_pattern in k for skip_key_pattern in self.skip_keys_pattern):
                skipped_keys.append(k)
                continue

            for pattern, replace in self.load_keys_pattern.items():
                if pattern in k:
                    # If the input channels are different then use the same trick as used in segmentation_models library
                    # e.g. in case of 3 channels new_weight[:, i] = pretrained_weight[:, i % 3]

                    new_in_channel = model_dict[k.replace(pattern, replace)].shape
                    pretrained_in_channel = pretrained_model["state_dict"][k].shape

                    if new_in_channel != pretrained_in_channel:
                        new_in_channel, pretrained_in_channel = new_in_channel[1], pretrained_in_channel[1]
                        for c in range(new_in_channel):
                            model_dict[k.replace(pattern, replace)][:, c] = pretrained_model["state_dict"][k][
                                :, c % pretrained_in_channel
                            ]

                        model_dict[k.replace(pattern, replace)] = (
                            model_dict[k.replace(pattern, replace)] * pretrained_in_channel
                        ) / new_in_channel
                    else:
                        model_dict[k.replace(pattern, replace)] = pretrained_model["state_dict"][k]
                    num_keys_loaded += 1

        if self.fold_name is None:
            self.fold_name = model_path.parent.name

        # Load the new weights
        self.load_state_dict(model_dict)
        if num_keys_loaded == 0:
            settings.log.warning(f"No key has been loaded from the pretrained dir: {pretrained_dir}")
        elif num_keys_loaded + len(skipped_keys) != len(model_dict):
            settings.log.warning(
                f"{num_keys_loaded} keys were changed in the model ({len(skipped_keys)} keys were skipped:"
                f" {skipped_keys}) but the model contains {len(model_dict)} keys. This means that some parameters of"
                " the model remain unchanged"
            )
        else:
            settings.log.info(
                f"Successfully loaded the pretrained model ({len(skipped_keys)} keys were skipped: {skipped_keys})."
            )

    @classmethod
    def pretrained_model(
        cls,
        model: str = None,
        run_folder: str = None,
        path: str | Path = None,
        fold_name: str = None,
        n_classes: int = None,
        n_channels: int = None,
        pretrained_weights: bool = True,
        **model_kwargs,
    ) -> Self | list[Self]:
        """
        Load a pretrained segmentation model.

        You can directly use this model to train a network on your data. The weights will be initialized with the weights from the pretrained network, except for the segmentation head which is initialized randomly (and may also be different in terms of number of classes, depending on your data). The returned instance corresponds to the calling class (e.g. `ModelImage`) and you can also find it in the third column of the pretrained models table (cf. readme).

        For example, load the pretrained model for the image-based segmentation network:
        >>> from htc import ModelImage, Normalization
        >>> run_folder = "2022-02-03_22-58-44_generated_default_model_comparison"  # HSI model
        >>> model = ModelImage.pretrained_model(model="image", run_folder=run_folder)  # doctest: +ELLIPSIS
        [...]
        >>> input_data = torch.randn(1, 100, 480, 640)  # NCHW
        >>> input_data = Normalization(channel_dim=1)(input_data)  # Model expects L1 normalized input
        >>> model(input_data).shape
        torch.Size([1, 19, 480, 640])

        It is also possible to have a different number of classes as output or a different number of channels as input:
        >>> model = ModelImage.pretrained_model(
        ...     model="image", run_folder=run_folder, n_classes=3, n_channels=10
        ... )  # doctest: +ELLIPSIS
        [...]
        >>> input_data = torch.randn(1, 10, 480, 640)  # NCHW
        >>> model(input_data).shape
        torch.Size([1, 3, 480, 640])

        The patch-based models also use the `ModelImage` class but with a different input (here using the patch_64 model):
        >>> run_folder = "2022-02-03_22-58-44_generated_default_64_model_comparison"  # HSI model
        >>> model = ModelImage.pretrained_model(model="patch", run_folder=run_folder)  # doctest: +ELLIPSIS
        [...]
        >>> input_data = torch.randn(1, 100, 64, 64)  # NCHW
        >>> input_data = Normalization(channel_dim=1)(input_data)  # Model expects L1 normalized input
        >>> model(input_data).shape
        torch.Size([1, 19, 64, 64])

        The procedure is the same for the superpixel-based segmentation network but this time also using a different calling class (`ModelSuperpixelClassification`):
        >>> from htc import ModelSuperpixelClassification
        >>> run_folder = "2022-02-03_22-58-44_generated_default_model_comparison"  # HSI model
        >>> model = ModelSuperpixelClassification.pretrained_model(
        ...     model="superpixel_classification", run_folder=run_folder
        ... )  # doctest: +ELLIPSIS
        [...]
        >>> input_data = torch.randn(2, 100, 32, 32)  # NCHW
        >>> input_data = Normalization(channel_dim=1)(input_data)  # Model expects L1 normalized input
        >>> model(input_data).shape
        torch.Size([2, 19])

        And also the pixel network:
        >>> from htc import ModelPixel
        >>> run_folder = "2022-02-03_22-58-44_generated_default_model_comparison"  # HSI model
        >>> model = ModelPixel.pretrained_model(model="pixel", run_folder=run_folder)  # doctest: +ELLIPSIS
        [...]
        >>> input_data = torch.randn(2, 100)  # NC
        >>> input_data = Normalization(channel_dim=1)(input_data)  # Model expects L1 normalized input
        >>> model(input_data)["class"].shape
        torch.Size([2, 19])

        For the pixel model, you can specify a different number of classes but you do not need to set the number of input channels because the underlying convolutional operations directly operate along the channel dimension. Hence, you can just supply input data with a different number of channels and it will work as well.
        >>> model = ModelPixel.pretrained_model(model="pixel", run_folder=run_folder, n_classes=3)  # doctest: +ELLIPSIS
        [...]
        >>> input_data = torch.randn(2, 90)  # NC
        >>> model(input_data)["class"].shape
        torch.Size([2, 3])

        Retrieve a collection of models as list:
        >>> models = ModelImage.pretrained_model(
        ...     model="image", run_folder="2024-09-11_00-11-38_baseline_rat_nested-*-2"
        ... )  # doctest: +ELLIPSIS
        [...]
        >>> len(models)
        3

        Args:
            model: Basic model type like image or pixel (first column in the pretrained models table). This corresponds to the folder name in the first hierarchy level of the training directory.
            run_folder: Name of the training run from which the weights should be loaded, e.g. to select HSI or RGB models (fourth column in the pretrained models table). This corresponds to the folder name in the second hierarchy level of the training directory. If the run folder contains a wildcard `*` to indicate a collection of runs (e.g. "2024-09-11_00-11-38_baseline_rat_nested-*-2"), this function will return a list of models from this collection.
            path: Alternatively of specifying the model and run folder, you can also specify the path to the run directory, the fold directory or the path to the checkpoint file (*.ckpt) directly.
            fold_name: Name of the validation fold which defines the trained network of the run. If None, the model with the highest metric score will be used.
            n_classes: Number of classes for the network output. If None, uses the same setting as in the trained network (e.g. 18 organ classes + background for the organ segmentation networks).
            n_channels: Number of channels of the input. If None, uses the same settings as in the trained network (e.g. 100 channels). This is inspired by the timm library (https://timm.fast.ai/models#How-is-timm-able-to-use-pretrained-weights-and-handle-images-that-are-not-3-channel-RGB-images?), i.e. it repeats the weights according to the desired number of channels. Please not that this does not take any semantic of the input into account, e.g. the wavelength range or the filter functions of the camera.
            pretrained_weights: If True, overwrite the weights of the model with the weights from the pretrained model, i.e. make use of the pretrained model. If False, will still load (and download) the model but keep the weights randomly initialized. This mainly ensures that the same config is used for the pretrained model.
            model_kwargs: Additional keyword arguments passed to the model instance.

        Returns: Instance of the calling model class initialized with the pretrained weights. The model object will be an instance of `torch.nn.Module`.
        """

        def _construct_model(run_dir: Path) -> Self:
            config = Config(run_dir / "config.json")

            if pretrained_weights:
                if path is not None:
                    config["model/pretrained_model/path"] = path
                else:
                    config["model/pretrained_model/model"] = model
                    config["model/pretrained_model/run_folder"] = run_dir.name

            if fold_name is not None:
                config["model/pretrained_model/fold_name"] = fold_name
            if n_classes is not None:
                config["input/n_classes"] = n_classes
            if n_channels is not None:
                assert model != "pixel", (
                    "The parameter n_channels cannot be used with the pixel model. The number of channels are solely"
                    " determined by the input (see examples)"
                )
                config["input/n_channels"] = n_channels

            return cls(config, **model_kwargs)

        run_dir = HTCModel.find_pretrained_run(model, run_folder, path)
        if isinstance(run_dir, list):
            return [_construct_model(r) for r in run_dir]
        else:
            return _construct_model(run_dir)

    @staticmethod
    def find_pretrained_run(
        model_name: str = None, run_folder: str = None, path: str | Path = None
    ) -> Path | list[Path]:
        """
        Searches for a pretrained run either in the local results directory, in the local PyTorch model cache directory or it will attempt to download the model. For the local results directory, the following folders are searched:
        - `results/training/<model_name>/<run_folder>`
        - `results/pretrained_models/<model_name>/<run_folder>`
        - `results/<model_name>/<run_folder>`
        - `<model_name>/<run_folder>` (relative/absolute path)

        Args:
            model_name: Basic model type like image or pixel.
            run_folder: Name of the training run directory (e.g. 2022-02-03_22-58-44_generated_default_model_comparison). If the run folder contains a wildcard `*` to indicate a collection of runs (e.g. "2024-09-11_00-11-38_baseline_rat_nested-*-2"), this function will return all matching run directories.
            path: Alternatively to model_name and run_folder, you can also specify the path to the run directory (may also be relative to the results directory in one of the folders from above). If the path points to the fold directory or the checkpoint file (*.ckpt), the corresponding run directory will be returned.

        Returns: Path to the requested training run (run directory usually starting with a timestamp).
        """
        if path is not None:
            if type(path) is str:
                path = Path(path)

            possible_locations = HTCModel._get_possible_locations(path)
            for location in possible_locations:
                if location.is_dir():
                    if location.name.startswith("fold"):
                        # At this point, we are only interested in the run directory and not the fold directory
                        location = location.parent

                    if model_name is not None:
                        assert location.parent.name == model_name, (
                            f"The found location {location} does not match the given model_name {model_name}"
                        )
                    if run_folder is not None:
                        assert location.name == run_folder, (
                            f"The found location {location} does not match the given run_folder {run_folder}"
                        )

                    return location
                elif location.is_file():
                    # From the checkpoint file to the run directory
                    return location.parent.parent

            raise ValueError(
                f"Could not find the pretrained model. Tried the following locations: {possible_locations}"
            )
        else:
            assert path is None, "The path parameter is not used if model_name and run_folder are specified"
            assert model_name is not None and run_folder is not None, (
                "Please specify model_name and run_folder (e.g. in your config via the keys"
                " `model/pretrained_model/model` and `model/pretrained_model/run_folder`) if no path is given"
            )

            def _load_run_folder(run_folder: str) -> Path:
                # Option 1: local results directory
                if settings.results_dir is not None:
                    possible_locations = HTCModel._get_possible_locations(Path(model_name) / run_folder)
                    for run_dir in possible_locations:
                        if run_dir.is_dir():
                            settings.log_once.info(f"Found pretrained run in the local results dir at {run_dir}")
                            return run_dir

                # Option 2: local hub dir (cache folder)
                hub_dir = Path(torch.hub.get_dir()) / "htc_checkpoints"
                run_dir = hub_dir / model_name / run_folder
                if run_dir.is_dir():
                    settings.log_once.info(f"Found pretrained run in the local hub dir at {run_dir}")
                    return run_dir

                # Option 3: download the model to the local hub dir
                name = f"{model_name}@{run_folder}"
                assert name in HTCModel.known_models, (
                    f"Could not find the training run for {model_name}/{run_folder} (neither locally nor as download option)"
                )
                model_info = HTCModel.known_models[name]

                hub_dir.mkdir(parents=True, exist_ok=True)

                # Download the archive containing all trained models for the run (i.e. a model per fold)
                zip_path = hub_dir / f"{name}.zip"
                settings.log.info(f"Downloading pretrained model {name} since it is not locally available")
                torch.hub.download_url_to_file(model_info["url"], zip_path)

                # Extract the archive in the models dir with the usual structure (e.g. image/run_folder/fold_name)
                with ZipFile(zip_path) as f:
                    f.extractall(hub_dir)
                zip_path.unlink()

                assert run_dir.is_dir(), "run folder not available even after download"

                # Check file contents to catch download errors
                hash_cat = ""
                for f in sorted(run_dir.rglob("*"), key=lambda x: str(x).lower()):
                    if f.is_file():
                        hash_cat += sha256_file(f)

                hash_folder = hashlib.sha256(hash_cat.encode()).hexdigest()
                if model_info["sha256"] != hash_folder:
                    settings.log.error(
                        f"The hash of the folder (hash of the file hashes, {hash_folder}) does not match the expected hash"
                        f" ({model_info['sha256']}). The download of the model was likely not successful. The downloaded"
                        f" files are not deleted and are still available at {hub_dir}. Please check the files manually"
                        " (e.g. for invalid file sizes). If you want to re-trigger the download process, just delete the"
                        f" corresponding run directory {run_dir}"
                    )
                else:
                    settings.log.info(f"Successfully downloaded the pretrained run to the local hub dir at {run_dir}")

                return run_dir

            if "*" in run_folder:
                # Find all * from left to right
                max_indices = []
                for match in re.findall(r"\*-(\d+)", run_folder):
                    max_indices.append(int(match))

                assert len(max_indices) > 0, (
                    f"Could not infer any maximum index from {run_folder} for the run folder collection. The collection must for example be be named nested-0-2, nested-1-2, nested-2-2."
                )

                run_folders = []
                for indices in itertools.product(*[range(i + 1) for i in max_indices]):
                    nested_run_folder = run_folder
                    for i in indices:
                        nested_run_folder = nested_run_folder.replace("*", str(i), 1)

                    assert "*" not in nested_run_folder, f"Could not replace all * in {run_folder} with {indices}"
                    run_folders.append(_load_run_folder(nested_run_folder))

                return run_folders
            else:
                return _load_run_folder(run_folder)

    @staticmethod
    def best_checkpoint(path: Path) -> Path:
        """
        Searches for the best model checkpoint path within the given run directory across all available folds.

        Args:
            path: The path to the training run or to a specific fold.

        Returns: The path to the best model checkpoint.
        """
        # Choose the model with the highest dice score
        checkpoint_paths = sorted(path.rglob("*.ckpt"))
        if len(checkpoint_paths) == 1:
            model_path = checkpoint_paths[0]
        else:
            table_path = path / "validation_table.pkl.xz"
            if table_path.exists():
                # Best model based on the validation table
                df_val = pd.read_pickle(table_path)
                df_val = df_val.query("epoch_index == best_epoch_index and dataset_index == 0")

                # Best model per fold
                config = Config(path / "config.json")
                agg = MetricAggregation(df_val, config=config)
                df_best = agg.grouped_metrics(domains=["fold_name", "epoch_index"])
                df_best = df_best.groupby(["fold_name", "epoch_index"], as_index=False)["dice_metric"].mean()
                df_best = df_best.sort_values(by=agg.metrics, ascending=False, ignore_index=True)

                fold_dir = path / df_best.iloc[0].fold_name
                checkpoint_paths = sorted(fold_dir.rglob("*.ckpt"))
                if len(checkpoint_paths) == 1:
                    model_path = checkpoint_paths[0]
                else:
                    checkpoint_paths = sorted(fold_dir.rglob(f"epoch={df_best.iloc[0].epoch_index}*.ckpt"))
                    assert len(checkpoint_paths) == 1, (
                        f"More than one checkpoint found for the epoch {df_best.iloc[0].epoch_index}: {checkpoint_paths}"
                    )
                    model_path = checkpoint_paths[0]
            else:
                model_path = checkpoint_paths[0]
                settings.log.warning(
                    f"Could not find the validation table at {table_path} but this is required to automatically"
                    f" determine the best model. The first found checkpoint will be used instead: {model_path}"
                )

        return model_path

    @staticmethod
    def markdown_table() -> str:
        """
        Generate a markdown table with all known pretrained models (used in the README).

        If you want to update the table in the README, simple call this function on the command line:
        ```bash
        python -c "from htc import HTCModel; print(HTCModel.markdown_table())"
        ```
        and replace the resulting table with the one in the README.
        """
        from htc.models.image.ModelImage import ModelImage
        from htc.models.pixel.ModelPixel import ModelPixel
        from htc.models.pixel.ModelPixelRGB import ModelPixelRGB
        from htc.models.superpixel_classification.ModelSuperpixelClassification import ModelSuperpixelClassification

        table_lines = [
            "| model type | modality | class | run folder |",
            "| ----------- | ----------- | ----------- | ----------- |",
        ]
        model_lines = []

        for name, download_info in HTCModel.known_models.items():
            model_type, run_folder = name.split("@")
            model_info = run_info(settings.training_dir / model_type / run_folder)

            if model_type == "superpixel_classification":
                ModelClass = ModelSuperpixelClassification
            elif model_type == "pixel":
                if "parameters" in run_folder or "rgb" in run_folder:
                    ModelClass = ModelPixelRGB
                else:
                    ModelClass = ModelPixel
            else:
                ModelClass = ModelImage

            # Check that the model can be loaded
            model = ModelClass.pretrained_model(model_type, run_folder=run_folder)
            assert isinstance(model, nn.Module)

            class_name = ModelClass.__name__
            class_path = Path(inspect.getfile(ModelClass)).relative_to(settings.src_dir)

            run_folder_md = ""
            if "nested" in run_folder:
                if "nested-0" not in run_folder:
                    # We only add one collection item to the table (not all nested runs)
                    continue

                run_folder_collection = re.sub(r"nested-\d+", "nested-*", run_folder)
                run_folders = HTCModel.find_pretrained_run(model_type, run_folder_collection)
                links = ", ".join(
                    f"[{i}]({HTCModel.known_models[f'{model_type}@{f.name}']['url']})"
                    for i, f in enumerate(run_folders)
                )
                run_folder_md = f"`{run_folder_collection}` (outer folds: {links})"
            else:
                run_folder_md = f"[`{run_folder}`]({download_info['url']})"

            model_lines.append(
                f"| {model_type} | {model_info['model_type']} | [`{class_name}`](./{class_path}) | {run_folder_md} |"
            )

        table_lines += reversed(model_lines)
        return "\n".join(table_lines)

    @staticmethod
    def _get_possible_locations(path: Path) -> list[Path]:
        return [
            settings.training_dir / path,
            settings.results_dir / path,
            settings.results_dir / "pretrained_models" / path,
            path,
        ]
