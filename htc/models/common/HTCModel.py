# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import hashlib
import inspect
import re
from pathlib import Path
from zipfile import ZipFile

import torch
import torch.nn as nn

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
    known_models = {
        "pixel@2022-02-03_22-58-44_generated_default_rgb_model_comparison": {
            "sha256": "c19c600958fa36f86d8742752bdbab9d067d0ee3f9c0e37cf281ce3b84b139da",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/pixel@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip",
        },
        "pixel@2022-02-03_22-58-44_generated_default_parameters_model_comparison": {
            "sha256": "b38de2a28464aa422b2b8d44861ad9dd1184ceb3053abb5d3d2e811f4ada662c",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/pixel@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip",
        },
        "pixel@2022-02-03_22-58-44_generated_default_model_comparison": {
            "sha256": "628f2c79ef3ea020bfdda3820670ffb277c0eca6f46d0d72a1692ca53d80a62a",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/pixel@2022-02-03_22-58-44_generated_default_model_comparison.zip",
        },
        "superpixel_classification@2022-02-03_22-58-44_generated_default_rgb_model_comparison": {
            "sha256": "c515ff6ae939408b0b866ce9d630cc5cd6c16182b0ea90e0e7e43e15673daa35",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/superpixel_classification@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip",
        },
        "superpixel_classification@2022-02-03_22-58-44_generated_default_parameters_model_comparison": {
            "sha256": "57c81b3abfcc785ba04c1203192aa2ebb7d4274f295daff9c50d64a62fcadd5f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/superpixel_classification@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip",
        },
        "superpixel_classification@2022-02-03_22-58-44_generated_default_model_comparison": {
            "sha256": "a39b3fad1e422e6e00563d9193ae751fb54aa3193196878b62be3e3dee8241f2",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/superpixel_classification@2022-02-03_22-58-44_generated_default_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_rgb_model_comparison": {
            "sha256": "66aa16eb5c0c3969377f30fd927722c48850dba522582565247ed02dbab8db78",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_parameters_model_comparison": {
            "sha256": "a4fc42c1f49cdb8a63c070a278c790805c72c8b91679eb39b2b7af4aae73827d",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_model_comparison": {
            "sha256": "05c44ad7122260f391d88436df3402c798b6c866c8c26714802f5cbc2dfa4335",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_64_rgb_model_comparison": {
            "sha256": "e158f5138d478f0e186106588b843609d5995dfdf91d82b914e87095af31ba78",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_64_rgb_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_64_parameters_model_comparison": {
            "sha256": "307e1770dc433dc1ad1418b8ecdfac58efe0fee1d5f6c06b555a3465965838d0",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_64_parameters_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_64_model_comparison": {
            "sha256": "09a270aa35923b5053fd558307b6c272815578142dcdbf68cbb695111df51224",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_64_model_comparison.zip",
        },
        "image@2022-02-03_22-58-44_generated_default_rgb_model_comparison": {
            "sha256": "98fd75c3d4729e5d8ed34676e9a6a0e3c1203f56738ffe517198c7807d152611",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip",
        },
        "image@2022-02-03_22-58-44_generated_default_parameters_model_comparison": {
            "sha256": "82617ffbf9ebc31c93e130a1af3ac8690c46f9f84c032745e97faddfb9786fce",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip",
        },
        "image@2022-02-03_22-58-44_generated_default_model_comparison": {
            "sha256": "9df57e16a73c700ff3de1d2b0de2bbf3efdc0b954b50a7f222c268e4079c806f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2022-02-03_22-58-44_generated_default_model_comparison.zip",
        },
    }

    def __init__(self, config: Config):
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
        """
        super().__init__()
        self.config = config

        # Default keys to load/skip for pretraining
        # Subclasses can modify these sets by adding elements to the list or replacing them
        self.load_keys_pattern = {"model."}  # This corresponds to the name of the attribute in the lightning class
        self.skip_keys_pattern = {"segmentation_head", "classification_head", "heads.heads"}

    def __post__init__(self):
        if self.config["model/pretrained_model"]:
            self._load_pretrained_model()

    def _load_pretrained_model(self) -> None:
        config_pretrained = self.config["model/pretrained_model"]

        # Find training run
        if "path" in config_pretrained:
            possible_directories = [settings.training_dir / config_pretrained["path"], config_pretrained["path"]]
            pretrained_dir = None

            for d in possible_directories:
                if d.exists():
                    pretrained_dir = d
                    break

            if pretrained_dir is None:
                raise ValueError(
                    f"Could not find the pretrained model. Tried the following locations: {possible_directories}"
                )
        else:
            assert "model" in config_pretrained and "run_folder" in config_pretrained, (
                "Please specify the model, run_folder and fold_name in your config (as subkeys from"
                f" model/pretrained_model). Given options: {config_pretrained}"
            )

            pretrained_dir = HTCModel.find_pretrained_run(config_pretrained["model"], config_pretrained["run_folder"])
            if "fold_name" in config_pretrained and config_pretrained["fold_name"] is not None:
                pretrained_dir /= config_pretrained["fold_name"]

        # Choose the model with the highest dice checkpoint
        highest_metric = 0
        pretrained_model = None
        model_path = None
        checkpoint_paths = sorted(pretrained_dir.rglob("*.ckpt"))
        map_location = None if torch.cuda.is_available() else "cpu"

        for checkpoint_path in checkpoint_paths:
            match = re.search(r"dice_metric=(\d+\.\d+)", checkpoint_path.name)
            if match is not None:
                current_metric = float(match.group(1))
                if highest_metric < current_metric:
                    highest_metric = current_metric
                    model_path = checkpoint_path
            else:
                current_model = torch.load(checkpoint_path, map_location=map_location)
                current_metric = [
                    v["best_model_score"].item()
                    for k, v in current_model["callbacks"].items()
                    if "ModelCheckpoint" in k
                ][0]
                if highest_metric < current_metric:
                    highest_metric = current_metric
                    pretrained_model = current_model

        if pretrained_model is None:
            assert model_path is not None, "Could not find the best model"
            pretrained_model = torch.load(model_path, map_location=map_location)

        # Change state dict keys
        model_dict = self.state_dict()
        num_keys_loaded = 0
        skipped_keys = []
        for k in pretrained_model["state_dict"].keys():
            if any([skip_key_pattern in k for skip_key_pattern in self.skip_keys_pattern]):
                skipped_keys.append(k)
                continue

            for load_key_pattern in self.load_keys_pattern:
                if load_key_pattern in k:
                    # If the input channels are different then use the same trick as used in segmentation_models library
                    # e.g. in case of 3 channels new_weight[:, i] = pretrained_weight[:, i % 3]

                    new_in_channel = model_dict[k.replace(load_key_pattern, "")].shape
                    pretrained_in_channel = pretrained_model["state_dict"][k].shape

                    if new_in_channel != pretrained_in_channel:
                        new_in_channel, pretrained_in_channel = new_in_channel[1], pretrained_in_channel[1]
                        for c in range(new_in_channel):
                            model_dict[k.replace(load_key_pattern, "")][:, c] = pretrained_model["state_dict"][k][
                                :, c % pretrained_in_channel
                            ]

                        model_dict[k.replace(load_key_pattern, "")] = (
                            model_dict[k.replace(load_key_pattern, "")] * pretrained_in_channel
                        ) / new_in_channel
                    else:
                        model_dict[k.replace(load_key_pattern, "")] = pretrained_model["state_dict"][k]
                    num_keys_loaded += 1

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
        model: str,
        run_folder: str,
        fold_name: str = None,
        n_classes: int = None,
        n_channels: int = None,
        pretrained_weights: bool = True,
    ) -> "HTCModel":
        """
        Load a pretrained segmentation model.

        You can directly use this model to train a network on your data. The weights will be initialized with the weights from the pretrained network, except for the segmentation head which is initialized randomly (and may also be different in terms of number of classes, depending on your data). The returned instance corresponds to the calling class (e.g. `ModelImage`) and you can also find it in the third column of the pretrained models table (cf. readme).

        For example, load the pretrained model for the image-based segmentation network:
        >>> from htc import ModelImage
        >>> run_folder = "2022-02-03_22-58-44_generated_default_model_comparison"  # HSI model
        >>> print("some log messages"); model = ModelImage.pretrained_model(model="image", run_folder=run_folder)  # doctest: +ELLIPSIS
        some log messages...
        >>> input_data = torch.randn(1, 100, 480, 640)  # NCHW
        >>> model(input_data).shape
        torch.Size([1, 19, 480, 640])

        It is also possible to have a different number of classes as output or a different number of channels as input:
        >>> print("some log messages"); model = ModelImage.pretrained_model(model="image", run_folder=run_folder, n_classes=3, n_channels=10)  # doctest: +ELLIPSIS
        some log messages...
        >>> input_data = torch.randn(1, 10, 480, 640)  # NCHW
        >>> model(input_data).shape
        torch.Size([1, 3, 480, 640])

        The patch-based models also use the `ModelImage` class but with a different input (here using the patch_64 model):
        >>> run_folder = "2022-02-03_22-58-44_generated_default_64_model_comparison"  # HSI model
        >>> print("some log messages"); model = ModelImage.pretrained_model(model="patch", run_folder=run_folder)  # doctest: +ELLIPSIS
        some log messages...
        >>> input_data = torch.randn(1, 100, 64, 64)  # NCHW
        >>> model(input_data).shape
        torch.Size([1, 19, 64, 64])

        The procedure is the same for the superpixel-based segmentation network but this time also using a different calling class (`ModelSuperpixelClassification`):
        >>> from htc import ModelSuperpixelClassification
        >>> run_folder = "2022-02-03_22-58-44_generated_default_model_comparison"  # HSI model
        >>> print("some log messages"); model = ModelSuperpixelClassification.pretrained_model(model="superpixel_classification", run_folder=run_folder)  # doctest: +ELLIPSIS
        some log messages...
        >>> input_data = torch.randn(2, 100, 32, 32)  # NCHW
        >>> model(input_data).shape
        torch.Size([2, 19])

        And also the pixel network:
        >>> from htc import ModelPixel
        >>> run_folder = "2022-02-03_22-58-44_generated_default_model_comparison"  # HSI model
        >>> print("some log messages"); model = ModelPixel.pretrained_model(model="pixel", run_folder=run_folder)  # doctest: +ELLIPSIS
        some log messages...
        >>> input_data = torch.randn(2, 100)  # NC
        >>> model(input_data)['class'].shape
        torch.Size([2, 19])

        For the pixel model, you can specify a different number of classes but you do not need to set the number of input channels because the underlying convolutional operations directly operate along the channel dimension. Hence, you can just supply input data with a different number of channels and it will work as well.
        >>> print("some log messages"); model = ModelPixel.pretrained_model(model="pixel", run_folder=run_folder, n_classes=3)  # doctest: +ELLIPSIS
        some log messages...
        >>> input_data = torch.randn(2, 90)  # NC
        >>> model(input_data)['class'].shape
        torch.Size([2, 3])

        Args:
            model: Basic model type like image or pixel (first column in the pretrained models table). This corresponds to the folder name in the first hierarchy level of the training directory.
            run_folder: Name of the training run from which the weights should be loaded, e.g. to select HSI or RGB models (fourth column in the pretrained models table). This corresponds to the folder name in the second hierarchy level of the training directory.
            fold_name: Name of the validation fold which defines the trained network of the run. If None, the model with the highest metric score will be used.
            n_classes: Number of classes for the network output. If None, uses the same setting as in the trained network (e.g. 18 organ classes + background for the organ segmentation networks).
            n_channels: Number of channels of the input. If None, uses the same settings as in the trained network (e.g. 100 channels). This is inspired by the timm library (https://timm.fast.ai/models#How-is-timm-able-to-use-pretrained-weights-and-handle-images-that-are-not-3-channel-RGB-images?), i.e. it repeats the weights according to the desired number of channels. Please not that this does not take any semantic of the input into account, e.g. the wavelength range or the filter functions of the camera.
            pretrained_weights: If True, overwrite the weights of the model with the weights from the pretrained model, i.e. make use of the pretrained model. If False, will still load (and download) the model but keep the weights randomly initialized. This mainly ensures that the same config is used for the pretrained model.

        Returns: Instance of the calling model class initialized with the pretrained weights. The model object will be an instance of `torch.nn.Module`.
        """
        run_dir = HTCModel.find_pretrained_run(model, run_folder)
        config = Config(run_dir / "config.json")
        if pretrained_weights:
            config["model/pretrained_model/model"] = model
            config["model/pretrained_model/run_folder"] = run_folder
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

        return cls(config)

    @staticmethod
    def find_pretrained_run(model_name: str, run_folder: str) -> Path:
        """
        Searches for a pretrained run either in the local training directory, in the local PyTorch model cache directory or it will attempt to download the model.

        Args:
            model_name: Basic model type like image or pixel.
            run_folder: Name of the training run directory.

        Returns: Path to the requested training run.
        """
        # Option 1: local training directory
        if settings.training_dir is not None:
            run_dir = settings.training_dir / model_name / run_folder
            if run_dir.is_dir():
                settings.log_once.info(f"Found pretrained run in the local training dir at {run_dir}")
                return run_dir

        # Option 2: local hub dir (cache folder)
        hub_dir = Path(torch.hub.get_dir()) / "htc_checkpoints"
        run_dir = hub_dir / model_name / run_folder
        if run_dir.is_dir():
            settings.log_once.info(f"Found pretrained run in the local hub dir at {run_dir}")
            return run_dir

        # Option 3: download the model to the local hub dir
        name = f"{model_name}@{run_folder}"
        assert (
            name in HTCModel.known_models
        ), f"Could not find the training run for {model_name}/{run_folder} (neither locally nor as download option)"
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
                f" ({model_info['sha256']}). The download of the model was likely not successful. The downloaded files"
                f" are not deleted and are still available at {hub_dir}. Please check the files manually (e.g. for"
                " invalid file sizes). If you want to re-trigger the download process, just delete the corresponding"
                f" run directory {run_dir}"
            )
        else:
            settings.log.info(f"Successfully downloaded the pretrained run to the local hub dir at {run_dir}")

        return run_dir

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

            model_lines.append(
                f"| {model_type} | {model_info['model_type']} | [`{class_name}`](./{class_path}) |"
                f" [{run_folder}]({download_info['url']}) |"
            )

        table_lines += reversed(model_lines)
        return "\n".join(table_lines)
