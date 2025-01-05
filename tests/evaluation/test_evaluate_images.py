# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import contextlib
import gc
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from htc.evaluation.evaluate_images import calc_surface_dice, calc_surface_distance, evaluate_images
from htc.models.common.utils import samples_equal
from htc.models.image.DatasetImage import DatasetImage
from htc.settings import settings
from htc.settings_seg import settings_seg


class TestCalcSurfaceDice:
    def test_tolerance(self) -> None:
        predictions = torch.zeros((2, 480, 640), dtype=torch.int64)
        labels = torch.zeros((2, 480, 640), dtype=torch.int64)
        mask = torch.ones(labels.shape, dtype=torch.bool)
        predictions[0, :, 50:] = 1
        labels[0, :, 60:] = 1  # 10 px shift

        res1 = evaluate_images(predictions, labels, mask, tolerances=[1, 1], metrics=["NSD"])
        res9 = evaluate_images(predictions, labels, mask, tolerances=[9, 9], metrics=["NSD"])
        res10 = evaluate_images(predictions, labels, mask, tolerances=[10, 10], metrics=["NSD"])
        res11 = evaluate_images(predictions, labels, mask, tolerances=[11, 11], metrics=["NSD"])

        assert len(res1) == len(res10) == len(res11) == 2
        assert (
            len(res1[0]["surface_dice_metric"])
            == len(res10[0]["surface_dice_metric"])
            == len(res11[0]["surface_dice_metric"])
            == 2
        )
        assert (
            len(res1[1]["surface_dice_metric"])
            == len(res10[1]["surface_dice_metric"])
            == len(res11[1]["surface_dice_metric"])
            == 1
        )
        assert (
            res1[0]["surface_dice_metric_image"]
            < res9[0]["surface_dice_metric_image"]
            < res10[0]["surface_dice_metric_image"]
        )
        assert res1[1]["surface_dice_metric_image"] == res10[1]["surface_dice_metric_image"]
        assert res10[0]["surface_dice_metric_image"] == res11[0]["surface_dice_metric_image"], (
            "No change anymore after the 10 px shift"
        )

    def test_mask(self) -> None:
        predictions = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.int64)
        predictions = predictions.unsqueeze(dim=0)

        labels = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int64)
        labels = labels.unsqueeze(dim=0)
        mask = torch.ones(1, 2, 4, dtype=torch.bool)

        res = evaluate_images(predictions, labels, mask, tolerances=[0, 0], metrics=["NSD"])
        assert len(res) == 1
        res = res[0]

        assert res["surface_dice_metric"][0] < 1.0
        assert res["surface_dice_metric_image"] < 1.0

        mask[0, :, 2:] = 0
        res_mask = evaluate_images(predictions, labels, mask, tolerances=[0, 0], metrics=["NSD"])
        assert len(res_mask) == 1
        res_mask = res_mask[0]

        assert res_mask["surface_dice_metric"] == torch.tensor([1.0])
        assert res_mask["surface_dice_metric_image"] == 1.0

    def test_multiple_classes(self) -> None:
        predictions = torch.tensor([[1, 1, 2, 2]], dtype=torch.int64)
        predictions = predictions.unsqueeze(dim=0)

        labels = torch.tensor([[0, 1, 2, 2]], dtype=torch.int64)
        labels = labels.unsqueeze(dim=0)

        mask = torch.ones(1, 1, 4, dtype=torch.bool)

        res1 = evaluate_images(predictions, labels, mask, tolerances=[0, 10, 10], metrics=["NSD"])
        assert len(res1) == 1
        res1 = res1[0]

        assert torch.all(res1["surface_dice_metric"] == torch.tensor([0.0, 1.0, 1.0]))
        assert res1["surface_dice_metric_image"] == pytest.approx(2 / 3)

        # The tolerances for the first class are not relevant here sind the class 0 does not occur in the prediction (and hence is always wrong irrespective of the tolerance)
        res2 = evaluate_images(predictions, labels, mask, tolerances=[10, 10, 10], metrics=["NSD"])[0]

        assert torch.all(res1["surface_dice_metric"] == res2["surface_dice_metric"])
        assert res1["surface_dice_metric_image"] == pytest.approx(res1["surface_dice_metric_image"])

    def test_missing_class(self) -> None:
        predictions = torch.zeros(1, 10, 10, dtype=torch.int64)
        predictions[0, :, 5:] = 1
        predictions[0, :, 8:] = 4
        labels = torch.zeros(1, 10, 10, dtype=torch.int64)
        labels[0, :, 6:] = 2
        mask = torch.ones(1, 10, 10, dtype=torch.bool)

        res = evaluate_images(predictions, labels, mask, tolerances=[1, 1, 1, 1, 1], metrics=["NSD"])
        assert len(res) == 1
        res = res[0]

        assert torch.all(res["surface_dice_metric"] == torch.tensor([1.0, 0.0]))

    def test_multiple_tolerances(self) -> None:
        predictions = torch.zeros(10, 10, dtype=torch.int64)
        predictions[:, 5:] = 1
        predictions = predictions.unsqueeze(dim=0)

        labels = torch.zeros(10, 10, dtype=torch.int64)
        labels[:, 6:] = 1
        labels = labels.unsqueeze(dim=0)

        mask = torch.ones(1, 10, 10, dtype=torch.bool)

        # Example for the class 0
        # edges_pred:
        # array([[ True,  True,  True,  True,  True, False],
        #        [ True, False, False, False,  True, False],
        #        [ True, False, False, False,  True, False],
        #        [ True, False, False, False,  True, False],
        #        [ True, False, False, False,  True, False],
        #        [ True, False, False, False,  True, False],
        #        [ True, False, False, False,  True, False],
        #        [ True, False, False, False,  True, False],
        #        [ True, False, False, False,  True, False],
        #        [ True,  True,  True,  True,  True, False]])
        #
        # edges_gt
        # array([[ True,  True,  True,  True,  True,  True],
        #        [ True, False, False, False, False,  True],
        #        [ True, False, False, False, False,  True],
        #        [ True, False, False, False, False,  True],
        #        [ True, False, False, False, False,  True],
        #        [ True, False, False, False, False,  True],
        #        [ True, False, False, False, False,  True],
        #        [ True, False, False, False, False,  True],
        #        [ True, False, False, False, False,  True],
        #        [ True,  True,  True,  True,  True,  True]])

        res1 = evaluate_images(predictions, labels, mask, tolerances=[0, 0], metrics=["NSD"])[0]
        res2 = evaluate_images(predictions, labels, mask, tolerances=[1, 0], metrics=["NSD"])[0]
        assert res1["surface_dice_metric"][0] == pytest.approx(1 - (8 + 10) / (20 + 6 + 20 + 8))
        assert res2["surface_dice_metric"][0] == 1
        assert res1["surface_dice_metric"][0] < res2["surface_dice_metric"][0]
        assert res1["surface_dice_metric"][1] == res2["surface_dice_metric"][1]
        assert res1["surface_dice_metric_image"] < res2["surface_dice_metric_image"]


@pytest.fixture(autouse=True, scope="module")
def clean_up_gpu():
    yield
    gc.collect()
    torch.cuda.empty_cache()


class TestCalcSurfaceDistance:
    def test_simple(self) -> None:
        labels = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.int64)
        labels = labels.unsqueeze(dim=0)

        predictions = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]], dtype=torch.int64)
        predictions = predictions.unsqueeze(dim=0)

        mask = torch.ones(2, 4, dtype=torch.bool)
        mask = mask.unsqueeze(dim=0)

        result = evaluate_images(predictions, labels, mask, metrics=["ASD"])
        assert len(result) == 1
        result = result[0]
        assert result["surface_distance_metric"][0] == 2 / 6

        labels = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [10, 0, 2, 3]], dtype=torch.int64)
        labels = labels.unsqueeze(dim=0)

        predictions = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1], [11, 1, 2, 2]], dtype=torch.int64)
        predictions = predictions.unsqueeze(dim=0)

        mask = torch.ones(3, 4, dtype=torch.bool)
        mask[2, :] = 0
        mask = mask.unsqueeze(dim=0)

        result_mask = evaluate_images(predictions, labels, mask, metrics=["ASD"])
        assert len(result_mask) == 1
        result_mask = result_mask[0]

        for key in result.keys():
            eq = result[key] == result_mask[key]
            if type(eq) == bool:
                assert eq
            else:
                assert torch.all(eq)

    def test_invalid_all(self) -> None:
        # inf value for the class 0 which does not occur in the predictions
        labels = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int64)
        labels = labels.unsqueeze(dim=0)

        predictions = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.int64)
        predictions = predictions.unsqueeze(dim=0)

        mask = torch.ones(2, 4, dtype=torch.bool)
        mask = mask.unsqueeze(dim=0)

        result = evaluate_images(predictions, labels, mask, metrics=["ASD"])
        assert len(result) == 1
        result = result[0]

        diagonal = math.sqrt(4 + 16)
        assert len(result["surface_distance_metric"]) == 1
        assert result["surface_distance_metric"][0] == pytest.approx(diagonal)
        assert result["surface_distance_metric_image"] == pytest.approx(diagonal)

    def test_invalid_some(self) -> None:
        # nan value due to the small structure of class id 2
        labels = torch.tensor([[2, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int64)
        labels = labels.unsqueeze(dim=0)

        predictions = torch.tensor([[3, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int64)
        predictions = predictions.unsqueeze(dim=0)

        mask = torch.ones(2, 4, dtype=torch.bool)
        mask = mask.unsqueeze(dim=0)

        result = evaluate_images(predictions, labels, mask, metrics=["ASD"])
        assert len(result) == 1
        result = result[0]

        assert result["surface_distance_metric"][0] == 0.0
        assert result["surface_distance_metric"][1] == 0.0, (
            "Structure is too small to calculate an edge, should be set to the maximum distance of all classes"
        )
        assert result["surface_distance_metric_image"] == 0.0


class TestEvaluateImagesCPU:
    def test_metric_combination(self) -> None:
        # Similar to test_input_labels() but only on the CPU since ASD and NSD cannot be computed on the GPU
        labels = torch.zeros(100, 100, dtype=torch.int64)
        labels[0:50, 0:20] = 1  # Area with 1000 pixels
        labels[0:50, 30:50] = 2
        labels[0:50, 60:80] = 3
        labels = labels.unsqueeze(dim=0)

        predictions = torch.zeros(100, 100, dtype=torch.int64)
        predictions[0:50, 10:20] = 1  # 500 pixels are not correctly classified
        predictions[0:50, 30:50] = 2
        predictions[0:50, 60:80] = 3
        predictions = predictions.unsqueeze(dim=0)

        mask = torch.ones(100, 100, dtype=torch.bool)
        mask = mask.unsqueeze(dim=0)

        # this test calls the individual functions from evaluate_images as a sanity test
        result = evaluate_images(predictions, labels, mask)
        result_asd = calc_surface_distance(predictions, labels, mask)
        result_nsd = calc_surface_dice(predictions, labels, mask, [1, 1, 1, 1])
        assert len(result) == len(result_asd) == len(result_nsd) == 1, "Batch size of 1"
        assert all(torch.all(result[b]["used_labels"] == result_asd[b]["used_labels"]) for b in range(len(result)))
        assert all(torch.all(result[b]["used_labels"] == result_nsd[b]["used_labels"]) for b in range(len(result)))

        for b in range(len(result)):
            result[b] |= result_asd[b]

        assert len(result) == 1
        result = result[0]
        assert "ce_loss" not in result
        assert "ece" not in result

        assert result["surface_distance_metric"].shape == (4,)
        assert torch.all(result["surface_distance_metric"][2:] == torch.tensor([0, 0]))
        assert "surface_distance_metric_image" in result

        assert torch.all(result["used_labels"] == torch.tensor([0, 1, 2, 3]))


@pytest.mark.serial
@pytest.mark.parametrize("device", ["cpu", "cuda"])
class TestEvaluateImages:
    def test_input_logits(self, device: str) -> None:
        # Setting the reference for classes 0, 1, 2 and 3
        labels = torch.zeros(100, 100, dtype=torch.int64, device=device)
        labels[0:50, 0:20] = 1  # Area with 1000 pixels
        labels[0:50, 30:50] = 2
        labels[0:50, 60:80] = 3
        labels = labels.unsqueeze(dim=0)

        # setting the correct shape for logits i.e. (batch, channels, height, width)
        logits = torch.zeros(1, 4, 100, 100, dtype=torch.float32, device=device)

        # initialize test softmaxes values
        # set class 0 as the default predicted class
        logits[0, 0, 0:100, 0:100] = 5.0
        logits[0, 1, 0:100, 0:100] = 0.0
        logits[0, 2, 0:100, 0:100] = 0.0
        logits[0, 3, 0:100, 0:100] = 0.0

        # set the class 1 in the logits, however 500 pixels are not correctly classified
        logits[0, 0, 0:50, 10:20] = 0.0
        logits[0, 1, 0:50, 10:20] = 5.0
        logits[0, 2, 0:50, 10:20] = 0.0
        logits[0, 3, 0:50, 10:20] = 0.0

        # set the class 2 in the softmaxes
        logits[0, 0, 0:50, 30:50] = 0.0
        logits[0, 1, 0:50, 30:50] = 0.0
        logits[0, 2, 0:50, 30:50] = 5.0
        logits[0, 3, 0:50, 30:50] = 0.0

        # set the class 3 in the softmaxes
        logits[0, 0, 0:50, 60:80] = 0.0
        logits[0, 1, 0:50, 60:80] = 0.0
        logits[0, 2, 0:50, 60:80] = 0.0
        logits[0, 3, 0:50, 60:80] = 5.0

        mask = torch.ones(100, 100, dtype=torch.bool, device=device)
        mask = mask.unsqueeze(dim=0)

        # take a softmax of logits before evaluation, as the function only accept softmax
        softmaxes = F.softmax(logits, dim=1)
        result = evaluate_images(softmaxes, labels, mask, n_classes=4)

        assert len(result) == 1
        result = result[0]
        assert "ece" in result

        assert len(result["dice_metric"]) == len(labels.unique())
        dice_background = 2 * 7000 / (7500 + 7000)  # dice = 2*|A ∩ B| / (|A| + |B|) with A = predictions and B = labels
        dice_class1 = 2 * 500 / (500 + 1000)
        assert torch.all(
            torch.isclose(result["dice_metric"][:4], torch.tensor([dice_background, dice_class1, 1, 1], device=device))
        ) and torch.all(result["dice_metric"][4:] == 0)

        assert torch.all(result["used_labels"] == torch.tensor([0, 1, 2, 3], device=device))

        # Construct the expected conf mat of shape 4x4, as we are considering four classes 0, 1, 2 and 3
        conf_mat = torch.zeros(4, 4, dtype=torch.int32, device=device)
        conf_mat[0, 0] = 100 * 100 - 3 * 1000
        conf_mat[1, 1] = 500
        conf_mat[1, 0] = 500
        conf_mat[2, 2] = 1000
        conf_mat[3, 3] = 1000

        assert torch.all(result["confusion_matrix"] == conf_mat)

    def test_input_labels(self, device: str) -> None:
        labels = torch.zeros(100, 100, dtype=torch.int64, device=device)
        labels[0:50, 0:20] = 1  # Area with 1000 pixels
        labels[0:50, 30:50] = 2
        labels[0:50, 60:80] = 3
        labels = labels.unsqueeze(dim=0)

        predictions = torch.zeros(100, 100, dtype=torch.int64, device=device)
        predictions[0:50, 10:20] = 1  # 500 pixels are not correctly classified
        predictions[0:50, 30:50] = 2
        predictions[0:50, 60:80] = 3
        predictions = predictions.unsqueeze(dim=0)

        mask = torch.ones(100, 100, dtype=torch.bool, device=device)
        mask = mask.unsqueeze(dim=0)

        result = evaluate_images(predictions, labels, mask)

        assert len(result) == 1
        result = result[0]
        assert "ece" not in result

        assert len(result["dice_metric"]) == len(labels.unique())
        dice_background = 2 * 7000 / (7500 + 7000)  # dice = 2*|A ∩ B| / (|A| + |B|) with A = predictions and B = labels
        dice_class1 = 2 * 500 / (500 + 1000)
        assert torch.all(
            torch.isclose(result["dice_metric"][:4], torch.tensor([dice_background, dice_class1, 1, 1], device=device))
        ) and torch.all(result["dice_metric"][4:] == 0)

        assert torch.all(result["used_labels"] == torch.tensor([0, 1, 2, 3], device=device))

        # Construct the expected conf mat
        conf_mat = torch.zeros(len(settings_seg.labels), len(settings_seg.labels), dtype=torch.int32, device=device)
        conf_mat[0, 0] = 100 * 100 - 3 * 1000
        conf_mat[1, 1] = 500
        conf_mat[1, 0] = 500
        conf_mat[2, 2] = 1000
        conf_mat[3, 3] = 1000

        assert torch.all(result["confusion_matrix"] == conf_mat)

    def test_input_softmax(self, device: str) -> None:
        # Example from https://www.jeremyjordan.me/semantic-segmentation/
        labels = torch.tensor(
            [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int64, device=device
        )
        labels = labels.unsqueeze(dim=0)

        predictions1 = torch.tensor(
            [[0.01, 0.03, 0.02, 0.02], [0.05, 0.12, 0.09, 0.07], [0.89, 0.85, 0.88, 0.91], [0.99, 0.97, 0.95, 0.97]],
            dtype=torch.float32,
            device=device,
        )
        predictions2 = 1 - predictions1
        missing_softmax = torch.zeros(len(settings_seg.labels) - 2, 4, 4, dtype=torch.float32, device=device)
        predictions = torch.cat([predictions1.unsqueeze(dim=0), predictions2.unsqueeze(dim=0), missing_softmax])
        predictions = predictions.unsqueeze(dim=0)

        mask = torch.ones(4, 4, dtype=torch.bool, device=device)
        mask = mask.unsqueeze(dim=0)

        result = evaluate_images(predictions, labels, mask)
        assert len(result) == 1
        result = result[0]

        assert "ece" in result

        assert len(result["dice_metric"]) == len(labels.unique())
        assert torch.all(result["dice_metric"][:2] == torch.tensor([1, 1], device=device)) and torch.all(
            result["dice_metric"][2:] == 0
        )

        conf_mat = torch.zeros(len(settings_seg.labels), len(settings_seg.labels), dtype=torch.int32, device=device)
        conf_mat[0, 0] = 8
        conf_mat[1, 1] = 8

        assert torch.all(result["confusion_matrix"] == conf_mat)

    def test_identical(self, device: str) -> None:
        dataset = DatasetImage.example_dataset()
        batch_size = 2
        dataloader = DataLoader(dataset, batch_size=batch_size)
        sample = next(iter(dataloader))
        for key, value in sample.items():
            if type(value) == torch.Tensor:
                sample[key] = value.to(device)

        assert sample["labels"].shape == sample["valid_pixels"].shape

        # create sample logits of shape (batch, classes, height, width) from labels
        b, h, w = sample["labels"].shape
        logits = torch.zeros(batch_size, len(settings_seg.labels), h, w, dtype=torch.float32, device=device)

        logits = logits.permute(1, 0, 2, 3)
        for c in range(len(settings_seg.labels)):
            # set the correct class according to labels
            logits[c, sample["labels"] == c] = 5.0

        logits = logits.permute(1, 0, 2, 3)
        softmaxes = F.softmax(logits, dim=1)

        result_softmaxes = evaluate_images(
            predictions=softmaxes,
            labels=sample["labels"],
            mask=sample["valid_pixels"],
            n_classes=len(settings_seg.labels),
        )
        result_labels = evaluate_images(sample["labels"], sample["labels"], sample["valid_pixels"])
        assert len(result_softmaxes) == batch_size

        for result_image in result_softmaxes:
            assert "ece" in result_image

        labels0, counts0 = sample["labels"][0, sample["valid_pixels"][0]].unique(return_counts=True)
        counts0 = counts0[labels0 < settings.label_index_thresh]
        labels0 = labels0[labels0 < settings.label_index_thresh]

        labels1, counts1 = sample["labels"][1, sample["valid_pixels"][1]].unique(return_counts=True)
        counts1 = counts1[labels1 < settings.label_index_thresh]
        labels1 = labels1[labels1 < settings.label_index_thresh]

        assert len(result_softmaxes[0]["dice_metric"]) == len(labels0)
        assert len(result_softmaxes[1]["dice_metric"]) == len(labels1)

        assert torch.all(result_softmaxes[0]["used_labels"] == labels0)
        assert torch.all(result_softmaxes[1]["used_labels"] == labels1)

        assert len(result_labels[0]["dice_metric"]) == len(labels0)
        assert len(result_labels[1]["dice_metric"]) == len(labels1)

        assert torch.all(result_labels[0]["used_labels"] == labels0)
        assert torch.all(result_labels[1]["used_labels"] == labels1)

        conf_mat0 = torch.zeros(len(settings_seg.labels), len(settings_seg.labels), dtype=torch.int32, device=device)
        conf_mat0[labels0, labels0] = counts0.type(torch.int32)
        conf_mat1 = torch.zeros(len(settings_seg.labels), len(settings_seg.labels), dtype=torch.int32, device=device)
        conf_mat1[labels1, labels1] = counts1.type(torch.int32)

        assert torch.all(result_softmaxes[0]["confusion_matrix"] == conf_mat0)
        assert torch.all(result_softmaxes[1]["confusion_matrix"] == conf_mat1)

        assert torch.all(result_labels[0]["confusion_matrix"] == conf_mat0)
        assert torch.all(result_labels[1]["confusion_matrix"] == conf_mat1)

    def test_repeating(self, device: str) -> None:
        @contextlib.contextmanager
        def deterministic_algorithms_enabled():
            previous_value = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(True)
            yield
            torch.use_deterministic_algorithms(previous_value)

        torch.manual_seed(0)
        example_shape = (480, 640)
        logits = torch.rand(2, len(settings_seg.labels), *example_shape, device=device)
        softmaxes = F.softmax(logits, dim=1)
        labels = torch.stack([
            torch.randint(0, 10, size=example_shape, dtype=torch.int64, device=device),
            torch.randint(0, 15, size=example_shape, dtype=torch.int64, device=device),
        ])
        masks = torch.ones(2, *example_shape, dtype=torch.bool, device=device)

        assert (
            labels[0, :, :].unique().shape != labels[1, :, :].unique().shape
            or labels[0, :, :].unique() != labels[1, :, :].unique()
        ), "The two example images should use different classes"

        with deterministic_algorithms_enabled():
            assert torch.are_deterministic_algorithms_enabled()
            # Check whether single vs. batch evaluation makes a difference (it should not)
            result0 = evaluate_images(softmaxes[:1, :, :, :], labels[:1, :, :], masks[:1, :, :])
            result_combined = evaluate_images(softmaxes, labels, masks)

        assert len(result0) == 1
        result0 = result0[0]
        assert len(result_combined) == 2
        assert result0.keys() == result_combined[0].keys()

        for key in result0.keys():
            compare = result0[key] == result_combined[0][key]
            if type(compare) == bool:
                assert compare
            else:
                torch.all(compare)

    def test_missing_class(self, device: str) -> None:
        labels = torch.tensor([[0, 1]], device=device)
        labels = labels.unsqueeze(dim=0)

        logits = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]], device=device)
        logits = logits.unsqueeze(dim=0)  # adding batch dimension
        logits = logits.unsqueeze(dim=0)  # adding height dimension

        # bringing logits to the correct shape of (batch, classes, height, width)
        logits = logits.permute(0, 3, 1, 2)
        softmaxes = F.softmax(logits, dim=1)

        mask = torch.ones(1, 2, dtype=torch.bool, device=device)
        mask = mask.unsqueeze(dim=0)

        result = evaluate_images(softmaxes, labels, mask, n_classes=3)

        assert len(result) == 1
        result = result[0]

        assert result["dice_metric_image"] == 0

    def test_invalid_input(self, device: str) -> None:
        labels = torch.tensor([[0, 1]], device=device)
        labels = labels.unsqueeze(dim=0)

        logits = torch.zeros(1, 2, 1, 2, dtype=torch.float32, device=device)
        logits[0, :, 0, 0] = torch.tensor([0.1, np.inf], device=device)
        logits[0, :, 0, 1] = torch.tensor([np.nan, 1.9], device=device)

        softmaxes = F.softmax(logits, dim=1)

        mask = torch.ones(1, 2, dtype=torch.bool, device=device)
        mask = mask.unsqueeze(dim=0)

        evaluate_images(softmaxes, labels, mask, n_classes=2)
        assert not torch.any(torch.isnan(softmaxes) | torch.isinf(softmaxes)), (
            "Invalid prediction values should be replaced"
        )

    def test_exclude_background(self, device: str) -> None:
        labels = torch.tensor([[0, 1, 1]], dtype=torch.int64, device=device)
        labels = labels.unsqueeze(dim=0)
        n_labels = len(labels.unique())

        logits = torch.tensor([[5.0, 0.0], [0.0, 5.0], [0.0, 5.0]], device=device)
        logits = logits.unsqueeze(dim=0)
        logits = logits.unsqueeze(dim=0)

        # bringing logits to the correct shape of (batch, classes, height, width)
        logits = logits.permute(0, 3, 1, 2)

        softmaxes = F.softmax(logits, dim=1)

        mask = torch.ones(1, 3, dtype=torch.bool, device=device)
        mask = mask.unsqueeze(dim=0)

        result_all_classes = evaluate_images(softmaxes, labels, mask, n_classes=2)
        assert len(result_all_classes) == 1
        result_all_classes = result_all_classes[0]

        # Again without the background
        mask[labels == 0] = False
        labels[labels == 0] = 10  # This label should not have any effect

        result_no_background = evaluate_images(softmaxes, labels, mask, n_classes=2)
        assert len(result_no_background) == 1
        result_no_background = result_no_background[0]

        # Background only in prediction
        softmaxes[0, 0, :, :] = 1.0
        softmaxes[0, 1, :, :] = 0.0
        results_pred_back = evaluate_images(softmaxes, labels, mask, n_classes=2)
        assert len(results_pred_back) == 1
        results_pred_back = results_pred_back[0]

        assert torch.all(result_all_classes["used_labels"] == torch.tensor([0, 1], device=device))
        assert torch.all(result_no_background["used_labels"] == torch.tensor([1], device=device))
        assert torch.all(results_pred_back["used_labels"] == torch.tensor([1], device=device))
        assert (
            len(result_all_classes["dice_metric"]) == n_labels
            and len(result_no_background["dice_metric"]) == n_labels - 1
            and len(results_pred_back["dice_metric"]) == n_labels - 1
        )
        assert result_all_classes["dice_metric_image"] == 1.0 and result_no_background["dice_metric_image"] == 1.0
        assert results_pred_back["dice_metric_image"] < result_all_classes["dice_metric_image"]

    def test_confidences(self, device: str) -> None:
        labels = torch.tensor([[0, 1, 1, 1]], dtype=torch.int64, device=device)
        labels = torch.stack([labels, labels, labels])

        softmaxes = torch.tensor(
            [
                [[0.7, 0.3], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8]],
            ],
            device=device,
        )
        softmaxes = torch.stack([softmaxes, softmaxes, softmaxes])
        softmaxes = softmaxes.permute(0, 3, 1, 2)

        mask = torch.ones(1, 4, dtype=torch.bool, device=device)
        mask = torch.stack([mask, mask, mask])

        r = evaluate_images(
            softmaxes,
            labels,
            mask,
            n_classes=2,
            metrics=["DSC_confidences"],
            confidence_thresholds=np.arange(0, 1, 0.1),
        )
        assert len(r) == 3
        assert samples_equal(r[0], r[1], equal_nan=True) and samples_equal(r[0], r[2], equal_nan=True)

        res0 = r[0]["DSC_confidences"]
        for t, r in res0.items():
            assert list(r.keys()) == ["areas", "dice_metric"]
            assert r["areas"].shape == (2,)
            assert r["dice_metric"].shape == (2,)

            if t < 0.7:
                assert torch.allclose(r["areas"], torch.tensor([1.0, 1.0], device=device))
                assert torch.allclose(r["dice_metric"], torch.tensor([1.0, 1.0], device=device))
            elif t < 0.8:
                assert torch.allclose(r["areas"], torch.tensor([1.0, 2 / 3], device=device))
                assert torch.allclose(r["dice_metric"], torch.tensor([1.0, 1.0], device=device))
            elif t < 0.9:
                assert torch.allclose(r["areas"], torch.tensor([0.0, 1 / 3], device=device))
                assert torch.allclose(r["dice_metric"], torch.tensor([torch.nan, 1.0], device=device), equal_nan=True)
            else:
                assert torch.allclose(r["areas"], torch.tensor([0.0, 0.0], device=device))
                assert torch.allclose(
                    r["dice_metric"], torch.tensor([torch.nan, torch.nan], device=device), equal_nan=True
                )
