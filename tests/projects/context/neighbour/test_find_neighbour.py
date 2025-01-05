# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch

from htc.models.data.DataSpecification import DataSpecification
from htc.models.image.DatasetImage import DatasetImage
from htc.settings_seg import settings_seg
from htc.utils.Config import Config
from htc_projects.context.neighbour.find_neighbour_valid_pixels import neighbour_class_percentage_for_valid_pixels
from htc_projects.context.neighbour.find_normalized_neighbour_matrix import find_normalized_neighbour_matrix


class TestFindNeighbour:
    def test_neighbour_class_percentage_sum_must_be_one(self) -> None:
        array = torch.tensor([[1, 3, 0], [1, 2, 2], [4, 4, 4]], dtype=torch.int64)
        valid_pixels = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.bool)
        number_of_classes = 5

        percentage_matrix = neighbour_class_percentage_for_valid_pixels(array, valid_pixels, number_of_classes)
        row_sum = torch.sum(percentage_matrix, axis=1)

        neighbours_of_0 = percentage_matrix[0, :]
        neighbours_of_1 = percentage_matrix[1, :]
        neighbours_of_2 = percentage_matrix[2, :]
        neighbours_of_3 = percentage_matrix[3, :]
        neighbours_of_4 = percentage_matrix[4, :]
        print(percentage_matrix)

        assert torch.allclose(torch.tensor([0, 0, 0.666, 0.333, 0]), neighbours_of_0, atol=0.001, rtol=0.01)
        assert torch.allclose(torch.tensor([0, 0, 0.25, 0.25, 0.5]), neighbours_of_1, atol=0.001, rtol=0.01)
        assert torch.allclose(torch.tensor([0.142, 0.286, 0, 0.142, 0.428]), neighbours_of_2, atol=0.001, rtol=0.01)

        assert torch.allclose(torch.tensor([0.2, 0.4, 0.4, 0, 0]), neighbours_of_3, atol=0.001, rtol=0.01)
        assert torch.allclose(torch.tensor([0, 0.333, 0.666, 0, 0]), neighbours_of_4, atol=0.001, rtol=0.01)
        assert torch.unique(row_sum)[0] == 1

    def test_neighbour_class_percentage_valid_pixels(self) -> None:
        array = torch.tensor([[1, 3, 0], [1, 2, 2], [4, 4, 4]])
        valid_pixels = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.bool)
        number_of_classes = 5

        percentage_matrix = neighbour_class_percentage_for_valid_pixels(array, valid_pixels, number_of_classes)
        row_sum = torch.sum(percentage_matrix, axis=1)
        assert torch.unique(row_sum)[0] == 0

    def test_find_normalized_neighbour_matrix(self) -> None:
        specs = DataSpecification("pigs_semantic-only_5foldsV2.json")

        config = Config({"input/no_features": True, "label_mapping": settings_seg.label_mapping})
        # load test set
        specs.activate_test_set()
        dataset = DatasetImage(specs.paths("test"), train=False, config=config)

        number_of_classes = len(settings_seg.label_mapping)
        normalized_neighbour_matrix = find_normalized_neighbour_matrix(dataset, number_of_classes)
        items = torch.unique(torch.sum(normalized_neighbour_matrix, dim=1, dtype=torch.float32))

        for item in items:
            assert int(item) == 0 or int(item) == 1
