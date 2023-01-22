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
            "sha256": "98332bf06ba6da992c191933f0179d6b8bdebf25548613728631df6c64338916",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/pixel@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip",
        },
        "pixel@2022-02-03_22-58-44_generated_default_parameters_model_comparison": {
            "sha256": "95e9ccd79e3c085f57020a11d763e832aadc61b33810b6547647d1f64fc728fc",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/pixel@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip",
        },
        "pixel@2022-02-03_22-58-44_generated_default_model_comparison": {
            "sha256": "647783ca86f4233ebe29aace90a38d1fb800148882a5ef9b6bcf56d67140efe9",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/pixel@2022-02-03_22-58-44_generated_default_model_comparison.zip",
        },
        "superpixel_classification@2022-02-03_22-58-44_generated_default_rgb_model_comparison": {
            "sha256": "78aa094446af744c636322ff544fd4ad0c3185abf21cdb2ed30d150e01e0af8a",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/superpixel_classification@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip",
        },
        "superpixel_classification@2022-02-03_22-58-44_generated_default_parameters_model_comparison": {
            "sha256": "3a0118b93d915e2998a4f32a28e8f07622fb4f0f90f2e1de72d4fe956d659cf5",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/superpixel_classification@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip",
        },
        "superpixel_classification@2022-02-03_22-58-44_generated_default_model_comparison": {
            "sha256": "f88190537240ae15d1125a35c0e358a83020dba07626078ae7537cee9f170469",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/superpixel_classification@2022-02-03_22-58-44_generated_default_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_rgb_model_comparison": {
            "sha256": "c4026909bdae8685b257b0709d495aa10b6d7711013ce21eb49185318483196f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_parameters_model_comparison": {
            "sha256": "66d3acd52a23b2b973aa4ef42ac37c702516a0c7e8e938ef1efd9d2844fc8f1d",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_model_comparison": {
            "sha256": "48209726e8c5bc01a8af8bcae89300c50f8250ce54c162ed329b8eeae960de9f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_64_rgb_model_comparison": {
            "sha256": "1dd4de497540fb8ce0b24d6c108c83dd223f214715940516b77ddaa5564b7bb3",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_64_rgb_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_64_parameters_model_comparison": {
            "sha256": "f9ca5e9ac6e57c1eeebd7f46cabbc1c654f2e0164f7f29b3e8030cc96b804d92",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_64_parameters_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_64_model_comparison": {
            "sha256": "8ef6a82f5395e08ff31c6dd01aa9477f12fd4aa45707eeb5d2e72d0e25900f98",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_64_model_comparison.zip",
        },
        "image@2022-02-03_22-58-44_generated_default_rgb_model_comparison": {
            "sha256": "da7483a4bf494b78518b267d0e8bdaa5d682b3a1913d2a48766ccccdf3564431",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip",
        },
        "image@2022-02-03_22-58-44_generated_default_parameters_model_comparison": {
            "sha256": "603637f5be4cbe5d3fbdbddf5f7561b5437b1e408ef560e924c1ee96eaccf8f7",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip",
        },
        "image@2022-02-03_22-58-44_generated_default_model_comparison": {
            "sha256": "88590ee95bae09b5a914f3cd2ef17799fbe8d95f07d52a24752be2e9e74945d2",
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

        for checkpoint_path in checkpoint_paths:
            match = re.search(r"dice_metric=(\d+\.\d+)", checkpoint_path.name)
            if match is not None:
                current_metric = float(match.group(1))
                if highest_metric < current_metric:
                    highest_metric = current_metric
                    model_path = checkpoint_path
            else:
                current_model = torch.load(checkpoint_path)
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
            pretrained_model = torch.load(model_path)

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
        cls, model_name: str, run_folder: str, fold_name: str = None, n_classes: int = None, n_channels: int = None
    ) -> "HTCModel":
        """
        Creates a model of the calling class and uses weights from the pretrained model.

        Note: This is very similar to the functions in hubconf.py in the repository root but works with all trained models (not just the published ones).

        >>> from htc.models.image.ModelImage import ModelImage
        >>> run_folder = "2022-02-03_22-58-44_generated_default_model_comparison"  # HSI model
        >>> print("some log messages"); model = ModelImage.pretrained_model("image", run_folder)  # doctest: +ELLIPSIS
        some log messages...
        >>> input_data = torch.randn(1, 100, 480, 640)  # NCHW
        >>> model(input_data).shape
        torch.Size([1, 19, 480, 640])

        Args:
            model_name: Basic model type like image or pixel. Folder name in the first hierarchy level of the training directory.
            run_folder: Name of the training run from which the weights should be loaded (e.g. to select HSI or RGB models). Folder name in the second hierarchy level of the training directory.
            fold_name: Name of the validation fold which defines the trained network of the run. If None, the model with the highest metric score will be used.
            n_classes: Number of classes for the network output. If None, uses the same setting as in the trained network (e.g. 18 organ classes + background for the organ segmentation networks).
            n_channels: Number of channels of the input. If None, uses the same settings as in the trained network (e.g. 100 channels).

        Returns: Instance of the calling model class initialized with the pretrained weights.
        """
        run_dir = HTCModel.find_pretrained_run(model_name, run_folder)
        config = Config(run_dir / "config.json")
        config["model/pretrained_model/model"] = model_name
        config["model/pretrained_model/run_folder"] = run_folder
        if fold_name is not None:
            config["model/pretrained_model/fold_name"] = fold_name
        if n_classes is not None:
            config["input/n_classes"] = n_classes
        if n_channels is not None:
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
        for f in sorted(run_dir.rglob("*")):
            if f.is_file():
                hash_cat += sha256_file(f)

        hash_folder = hashlib.sha256(hash_cat.encode()).hexdigest()
        assert model_info["sha256"] == hash_folder, (
            "The hash of the folder (hash of the file hashes) does not match the expected hash. The download of the"
            " model was likely not successful. The downloaded files are not deleted and are still available at"
            f" {hub_dir}. Please check the files manually. If you want to re-trigger the download process, just delete"
            " the corresponding run directory"
        )

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
        table_lines = [
            "| model type | modality | run folder | class |",
            "| ----------- | ----------- | ----------- | ----------- |",
        ]
        model_lines = []

        for name, download_info in HTCModel.known_models.items():
            model_type, run_folder = name.split("@")
            model_info = run_info(settings.training_dir / model_type / run_folder)

            # Load the model to get the class name
            model = torch.hub.load(settings.src_dir, model_type, run_folder=run_folder, source="local")
            class_name = model.__class__.__name__
            class_path = Path(inspect.getfile(model.__class__)).relative_to(settings.src_dir)

            model_lines.append(
                f"| {model_type} | {model_info['model_type']} | [{run_folder}]({download_info['url']}) |"
                f" [`{class_name}`](./{class_path}) |"
            )

        table_lines += reversed(model_lines)
        return "\n".join(table_lines)
