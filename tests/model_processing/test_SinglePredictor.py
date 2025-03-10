# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import logging

import pytest
import torch
from pytest import LogCaptureFixture

from htc import settings
from htc.model_processing.SinglePredictor import SinglePredictor
from htc.models.common.utils import samples_equal
from htc.models.data.DataSpecification import DataSpecification
from htc.models.image.DatasetImageBatch import DatasetImageBatch
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc_projects.species.settings_species import settings_species


class TestSinglePredictor:
    @pytest.mark.serial
    @pytest.mark.parametrize("test", [True, False])
    def test_predict_batch(self, test: bool) -> None:
        spec = DataSpecification("pigs_semantic-only_5foldsV2.json")
        paths = spec.paths("val")
        config = Config("htc_projects/context/models/configs/organ_transplantation_0.8.json")
        dataloader = DatasetImageBatch.batched_iteration(paths, config)
        batch = next(iter(dataloader))

        predictor = SinglePredictor(
            model="image", run_folder="2023-02-08_14-48-02_organ_transplantation_0.8", test=test
        )
        prediction1 = predictor.predict_batch(batch)
        prediction2 = predictor.predict_batch(batch["features"])
        assert samples_equal(prediction1, prediction2)

        prediction_paths = next(
            iter(predictor.predict_paths(paths[: prediction1["class"].size(0)], return_batch=False))
        )
        assert samples_equal(prediction1, prediction_paths)

    def test_fold_path(self, caplog: LogCaptureFixture) -> None:
        predictor1 = SinglePredictor(
            model="image",
            run_folder="2023-02-08_14-48-02_organ_transplantation_0.8",
            device="cpu",
            test=False,
        )
        predictor2 = SinglePredictor(
            model="image",
            run_folder="2023-02-08_14-48-02_organ_transplantation_0.8",
            fold_name="fold_P044,P050,P059",
            device="cpu",
            test=False,
        )
        predictor3 = SinglePredictor(
            path=settings.training_dir / "image" / "2023-02-08_14-48-02_organ_transplantation_0.8",
            device="cpu",
            test=False,
        )
        predictor4 = SinglePredictor(
            path=settings.training_dir
            / "image"
            / "2023-02-08_14-48-02_organ_transplantation_0.8"
            / "fold_P044,P050,P059",
            device="cpu",
            test=False,
        )

        sample = torch.rand(480, 640, 100)
        prediction1 = predictor1.predict_sample(sample)
        prediction2 = predictor2.predict_sample(sample)
        prediction3 = predictor3.predict_sample(sample)
        prediction4 = predictor4.predict_sample(sample)
        assert samples_equal(prediction1, prediction2)
        assert samples_equal(prediction1, prediction3)
        assert samples_equal(prediction1, prediction4)

        warnings = [r.msg for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 4
        assert all("expects L1 normalized" in w for w in warnings)

    @pytest.mark.serial
    def test_nested(self, caplog: LogCaptureFixture) -> None:
        caplog.set_level(logging.DEBUG, "htc")

        predictor = SinglePredictor(model="image", run_folder="2025-03-09_19-38-10_baseline_rat_nested-*-2", test=True)
        assert len(predictor.run_dir) == 3
        assert predictor.run_dir[0].name == "2025-03-09_19-38-10_baseline_rat_nested-0-2"
        assert predictor.run_dir[1].name == "2025-03-09_19-38-10_baseline_rat_nested-1-2"
        assert predictor.run_dir[2].name == "2025-03-09_19-38-10_baseline_rat_nested-2-2"
        assert len(predictor.model.models) == 15, "3 outer folds and 5 inner folds expected"

        path = DataPath.from_image_name("R002#2023_09_19_10_14_28#0202-00118")
        prediction = next(iter(predictor.predict_paths(paths=[path], return_batch=False)))["class"]
        assert prediction.shape == (1, len(settings_species.label_mapping), *path.dataset_settings["spatial_shape"])

        assert "Reducing the batch size to 1" in caplog.text
        assert "Reducing the number of workers to 1" in caplog.text
        assert predictor.model.paths_dataloader([path]).dataset.shared_dict["features"].shape == (3, 1, 480, 640, 100)
