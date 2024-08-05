# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch

from htc.context.neighbour.find_neighbour_valid_pixels import (
    count_rows_sum_eq_1,
    neighbour_class_percentage_for_valid_pixels,
)
from htc.models.image.DatasetImage import DatasetImage


def find_normalized_neighbour_matrix(dataset: DatasetImage, n_classes: int) -> torch.FloatTensor:
    """
    Calculate the normalized neighbourhood confusion matrix for all images.

    Arg:
        dataset: A DatasetImage class which needs to contain a matrix for the 'labels' key and one for the 'valid_pixels' key.
        n_classes: int number of different classes that appear in the dataset. (The neighbouir class percentage will only be calculated for these classes)
    """
    result = {}
    rows_diff_from_0 = {}

    # Group all images per label (what pig they are from) and add them
    for sample in dataset:
        # get neighbour matrix
        subject_name = sample["image_name"].split("#")[0]
        neighbour_matrix = neighbour_class_percentage_for_valid_pixels(
            sample["labels"], sample["valid_pixels"], n_classes
        )

        # Add the matrices that have the same subject name & keep track
        if subject_name in result:
            result[subject_name] = torch.add(result[subject_name], neighbour_matrix)
        else:
            result[subject_name] = neighbour_matrix

        if subject_name in rows_diff_from_0:
            rows_diff_from_0[subject_name] = torch.add(
                rows_diff_from_0[subject_name],
                count_rows_sum_eq_1(neighbour_matrix),
            )
        else:
            rows_diff_from_0[subject_name] = count_rows_sum_eq_1(neighbour_matrix)

    # For each label, divide to get the average
    normalized_result = {}
    for key in result.keys():
        normalized_result[key] = torch.div(result[key], rows_diff_from_0[key][:, None])
        # Make sure that NaN turn into 0s
        normalized_result[key] = torch.nan_to_num(normalized_result[key])

    # Add all labels and divide
    result_matrix = torch.zeros(n_classes, n_classes)
    result_rows_diff_from_0 = torch.zeros(n_classes)

    for key in normalized_result.keys():
        result_matrix = torch.add(result_matrix, normalized_result[key])
        result_rows_diff_from_0 = torch.add(
            result_rows_diff_from_0,
            count_rows_sum_eq_1(normalized_result[key]),
        )
        result_matrix = torch.nan_to_num(result_matrix)

    normalized_result_matrix = torch.div(result_matrix, result_rows_diff_from_0[:, None])
    return normalized_result_matrix
