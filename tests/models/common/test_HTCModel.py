# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import logging
import re
import shutil
import warnings
from pathlib import Path

import pytest
import requests
import torch
from lightning import seed_everything
from pytest import LogCaptureFixture, MonkeyPatch

from htc.models.common.HTCModel import HTCModel
from htc.models.common.transforms import Normalization
from htc.models.common.utils import samples_equal
from htc.models.image.ModelImage import ModelImage
from htc.models.pixel.ModelPixel import ModelPixel
from htc.models.run_pretrained_semantic_models import compress_run
from htc.models.superpixel_classification.ModelSuperpixelClassification import ModelSuperpixelClassification
from htc.settings import settings
from htc.utils.Config import Config


# Loading of the pretrained networks also requires GPU memory
@pytest.mark.serial
class TestHTCModel:
    run_folder = "2022-02-03_22-58-44_generated_default_model_comparison"

    def test_find_trained_run(self, tmp_path: Path, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture) -> None:
        monkeypatch.setattr(type(settings), "training_dir", tmp_path / "nonexistent")
        monkeypatch.setenv("TORCH_HOME", str(tmp_path))

        run_dir = HTCModel.find_pretrained_run("pixel", "2022-02-03_22-58-44_generated_default_model_comparison")
        assert run_dir.exists()
        assert len(caplog.records) == 2 and "Successfully downloaded" in caplog.records[1].msg

        run_dir_same = HTCModel.find_pretrained_run(path=run_dir)
        assert run_dir == run_dir_same

    @pytest.mark.parametrize(
        "model_name, ModelClass, example_key, head_key, n_skipped_keys",
        [
            ("pixel", ModelPixel, "fc2.weight", "heads", 2),
            ("image", ModelImage, "architecture.encoder._blocks.3._bn0.weight", "segmentation_head", 2),
            ("patch", ModelImage, "architecture.encoder._blocks.3._bn0.weight", "segmentation_head", 2),
            (
                "superpixel_classification",
                ModelSuperpixelClassification,
                "architecture.encoder._blocks.3._bn0.weight",
                "segmentation_head|classification_head",
                4,
            ),
        ],
    )
    def test_pretrained_model(
        self,
        model_name: str,
        ModelClass: type,
        example_key: str,
        head_key: str,
        n_skipped_keys: int,
        caplog: LogCaptureFixture,
    ) -> None:
        # Default model
        config = Config.from_model_name("default", model_name)

        seed_everything(settings.default_seed, workers=True)
        model_empty = ModelClass(config).state_dict()

        # Pretrained via run parts
        fold_name = "fold_P041,P060,P069"
        config["model/pretrained_model/model"] = model_name
        config["model/pretrained_model/run_folder"] = self.run_folder
        config["model/pretrained_model/fold_name"] = fold_name

        seed_everything(settings.default_seed, workers=True)
        model_pretrained = ModelClass(config).state_dict()
        assert f"{n_skipped_keys} keys were skipped" in caplog.records[-1].msg

        # Weights are different, head is the same
        assert model_empty.keys() == model_pretrained.keys()
        assert not torch.allclose(model_empty[example_key], model_pretrained[example_key])
        n_head_matches = 0
        for k in model_empty.keys():
            if re.search(head_key, k) is not None:
                assert torch.allclose(model_empty[k], model_pretrained[k])
                n_head_matches += 1
        assert n_head_matches in [2, 4]

        # Pretrained via path
        del config["model/pretrained_model/model"]
        del config["model/pretrained_model/run_folder"]
        config["model/pretrained_model/path"] = (
            f"{model_name}/2022-02-03_22-58-44_generated_default_model_comparison/fold_P041,P060,P069"
        )

        seed_everything(settings.default_seed, workers=True)
        model_pretrained2 = ModelClass(config).state_dict()
        assert f"{n_skipped_keys} keys were skipped" in caplog.records[-1].msg

        assert model_pretrained.keys() == model_pretrained2.keys()
        for k in model_empty.keys():
            assert torch.allclose(model_pretrained[k], model_pretrained2[k])

        # Pretrained via model class
        seed_everything(settings.default_seed, workers=True)
        model_pretrained_class = ModelClass.pretrained_model(
            model_name, run_folder=self.run_folder, fold_name=fold_name
        )
        model_pretrained_class = model_pretrained_class.state_dict()
        assert f"{n_skipped_keys} keys were skipped" in caplog.records[-1].msg

        assert model_pretrained.keys() == model_pretrained_class.keys()
        for k in model_empty.keys():
            assert torch.allclose(model_pretrained[k], model_pretrained_class[k])

        # Pretrained but with different number of classes
        config["input/n_classes"] = 2

        seed_everything(settings.default_seed, workers=True)
        model_binary = ModelClass(config).state_dict()
        assert f"{n_skipped_keys} keys were skipped" in caplog.records[-1].msg

        for k in model_empty.keys():
            if re.search(head_key, k) is not None:
                assert model_binary[k].size(0) == 2

        assert model_pretrained.keys() == model_binary.keys()
        for k in model_empty.keys():
            if re.search(head_key, k) is None:
                assert torch.allclose(model_pretrained[k], model_binary[k])

        # Skip this part if pixel model is being used
        if model_name != "pixel":
            # Pretrained with different number of input channels (in this case, the pretrained model has 3 channels)
            config["model/pretrained_model/path"] = (
                f"{model_name}/2022-02-03_22-58-44_generated_default_rgb_model_comparison/fold_P041,P060,P069"
            )

            seed_everything(settings.default_seed, workers=True)
            model_pretrained_rgb = ModelClass(config).state_dict()

            # Check if all the mod pretrained channels have the same weights for an example key
            first_key = next(iter(model_pretrained_rgb.keys()))
            for c in range(config["input/n_channels"]):
                assert torch.allclose(
                    model_pretrained_rgb[first_key][:, c % 3, :, :], model_pretrained_rgb[first_key][:, c, :, :]
                )

        assert "Successfully downloaded" not in caplog.text
        assert not any(r.levelno > logging.INFO for r in caplog.records)

    def test_best_model_fold(self, caplog: LogCaptureFixture) -> None:
        seed_everything(settings.default_seed, workers=True)
        model_pretrained1 = ModelImage.pretrained_model(
            "image", run_folder=self.run_folder, fold_name="fold_P044,P050,P059"
        )
        model_pretrained1 = model_pretrained1.state_dict()
        seed_everything(settings.default_seed, workers=True)
        model_pretrained2 = ModelImage.pretrained_model("image", run_folder=self.run_folder)
        model_pretrained2 = model_pretrained2.state_dict()

        assert "downloaded" not in caplog.text
        assert all(r.levelno < logging.WARNING for r in caplog.records)
        assert model_pretrained1.keys() == model_pretrained2.keys()
        for k in model_pretrained1.keys():
            assert torch.allclose(model_pretrained1[k], model_pretrained2[k])

    def test_reproduce_hash(self, tmp_path: Path) -> None:
        hash_folder = compress_run(
            run_dir=settings.training_dir / "pixel" / "2022-02-03_22-58-44_generated_default_model_comparison",
            output_path=tmp_path / "test.zip",
        )
        assert (
            hash_folder
            == HTCModel.known_models["pixel@2022-02-03_22-58-44_generated_default_model_comparison"]["sha256"]
        )

    def test_markdown_table(self, caplog: LogCaptureFixture) -> None:
        with (settings.src_dir / "README_public.md").open() as f:
            table_readme = f.read()
        match = re.search(r"(\| model type.*\|)\s+>", table_readme, flags=re.DOTALL)
        assert match is not None
        table_readme = match.group(1)

        # This also ensures that every model can be loaded
        table_new = HTCModel.markdown_table()

        assert "Successfully downloaded" not in caplog.text
        assert table_readme == table_new, "The pretrained models table in the README is not up-to-date"

    def test_links_accessible(self) -> None:
        for name, info in HTCModel.known_models.items():
            assert info["url"].endswith(".zip"), name
            req = requests.head(info["url"])
            if req.status_code != 200:
                # No failure because the server may be down
                warnings.warn(
                    f"The link {info['url']} returned a status code of: {req.status_code}\nPlease check the link manually",
                    stacklevel=2,
                )

    @pytest.mark.parametrize(
        "model_name, ModelClass, input_shape",
        [
            ("pixel", ModelPixel, (2, 100)),
            ("image", ModelImage, (1, 100, 480, 640)),
            ("patch", ModelImage, (1, 100, 64, 64)),
            ("superpixel_classification", ModelSuperpixelClassification, (2, 100, 32, 32)),
        ],
    )
    def test_normalization_check(
        self, model_name: str, ModelClass: type, input_shape: tuple[int, ...], caplog: LogCaptureFixture
    ) -> None:
        with torch.no_grad():
            seed_everything(settings.default_seed, workers=True)
            fold_name = "fold_P041,P060,P069"
            model = ModelClass.pretrained_model(model_name, run_folder=self.run_folder, fold_name=fold_name)

            input_data = torch.rand(*input_shape)
            input_data_normalized = Normalization(channel_dim=1)(input_data)

            model(input_data_normalized)
            model(input_data)

            records = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert len(records) == 0, "Only checked for the first batch"

            input_data_normalized[0, ...] = 0
            model = ModelClass.pretrained_model(model_name, run_folder=self.run_folder, fold_name=fold_name)
            model(input_data_normalized)
            records = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert len(records) == 0, "Zero values are allowed"

            if model_name == "image":
                input_data_normalized = Normalization(channel_dim=1)(input_data)
                input_data_normalized[0, 0, 0, 0] = 5
                model = ModelClass.pretrained_model(model_name, run_folder=self.run_folder, fold_name=fold_name)
                model(input_data_normalized)
                records = [r for r in caplog.records if r.levelno >= logging.WARNING]
                assert len(records) == 0, "Single pixel deviations are allowed"

            model = ModelClass.pretrained_model(model_name, run_folder=self.run_folder, fold_name=fold_name)
            model(input_data)
            records = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert len(records) == 1
            assert "not seem to be L1 normalized" in records[0].msg

    def test_path_loading(self, caplog: LogCaptureFixture) -> None:
        fold_name = "fold_P044,P050,P059"
        ckpt_name = "epoch=100-dice_metric=0.74.ckpt"

        # Different ways of loading the same model
        seed_everything(settings.default_seed, workers=True)
        model1 = ModelPixel.pretrained_model(model="pixel", run_folder=self.run_folder, fold_name=fold_name)
        seed_everything(settings.default_seed, workers=True)
        model2 = ModelPixel.pretrained_model(path=f"pixel/{self.run_folder}/{fold_name}")
        seed_everything(settings.default_seed, workers=True)
        model3 = ModelPixel.pretrained_model(path=settings.training_dir / "pixel" / self.run_folder / fold_name)
        seed_everything(settings.default_seed, workers=True)
        model4 = ModelPixel.pretrained_model(
            path=settings.training_dir / "pixel" / self.run_folder / fold_name / ckpt_name
        )
        seed_everything(settings.default_seed, workers=True)
        model5 = ModelPixel.pretrained_model(
            path=settings.training_dir / "pixel" / self.run_folder, fold_name=fold_name
        )

        assert model1.fold_name == fold_name
        assert model2.fold_name == fold_name
        assert model3.fold_name == fold_name
        assert model4.fold_name == fold_name
        assert model5.fold_name == fold_name

        model1 = model1.state_dict()
        model2 = model2.state_dict()
        model3 = model3.state_dict()
        model4 = model4.state_dict()
        model5 = model5.state_dict()

        assert "downloaded" not in caplog.text
        assert model1.keys() == model2.keys() == model3.keys() == model4.keys() == model5.keys()
        for k in model1.keys():
            assert torch.allclose(model1[k], model2[k])
            assert torch.allclose(model1[k], model3[k])
            assert torch.allclose(model1[k], model4[k])
            assert torch.allclose(model1[k], model5[k])

    def test_temperature_scaling(self, tmp_path: Path) -> None:
        seed_everything(settings.default_seed, workers=True)
        input_data = torch.randn(2, 100)

        run_dir = HTCModel.find_pretrained_run("pixel", self.run_folder)
        shutil.copytree(run_dir, tmp_path / "run", ignore=shutil.ignore_patterns("prediction*"))

        seed_everything(settings.default_seed, workers=True)
        model = ModelPixel.pretrained_model(path=tmp_path / "run", fold_name="fold_P044,P050,P059")
        model.eval()
        assert model.fold_name == "fold_P044,P050,P059"

        with torch.no_grad():
            output_default1 = model(input_data)
            output_default2 = model(input_data)

            assert samples_equal(output_default1, output_default2)

        config = Config(tmp_path / "run" / "config.json")
        config["post_processing/calibration/scaling/fold_P044,P050,P059"] = 1.5
        config["post_processing/calibration/bias/fold_P044,P050,P059"] = 0.0
        config["post_processing/calibration/nll_prior"] = 0.0
        config.save_config(tmp_path / "run" / "config.json")

        seed_everything(settings.default_seed, workers=True)
        model = ModelPixel.pretrained_model(path=tmp_path / "run", fold_name="fold_P044,P050,P059")
        model.eval()
        assert model.fold_name == "fold_P044,P050,P059"

        with torch.no_grad():
            output_default3 = model(input_data)
            output_default4 = model(input_data)

            assert samples_equal(output_default3, output_default4)
            assert not samples_equal(output_default1, output_default3)

            output_default1["class"] = output_default1["class"] * 1.5
            output_default1["class"] -= torch.logsumexp(output_default1["class"], axis=-1, keepdim=True)
            assert samples_equal(output_default1, output_default3)
