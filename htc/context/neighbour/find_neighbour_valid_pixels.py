# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch


def find_neighbour_classes_valid_pixels(
    labels: torch.IntTensor, label_index: int, valid_pixels: torch.BoolTensor
) -> torch.FloatTensor:
    """
    Create a matrix which has True for label_index and False for all other classes.

    Arg:
        labels: torch.IntTensor containing a label of a class (int) in each entry
        label_index: int which describes for what class the neighbour pixels have to be found
        valid_pixels: torch.BoolTensor containing which pixels of the image are valid (bool)
    """

    class_matrix = labels == label_index

    # The kernel defines our neighbour concept, does not work with torch
    kernel = torch.tensor([[True, True, True], [True, True, True], [True, True, True]], dtype=torch.float32)

    unsqueezed_matrix = class_matrix.unsqueeze(dim=0).unsqueeze(dim=0).type("torch.FloatTensor")
    unsqueezed_kernel = kernel.unsqueeze(dim=0).unsqueeze(dim=0)
    unsqueezed_dilation_matrix = torch.nn.functional.conv2d(unsqueezed_matrix, unsqueezed_kernel, padding=(1, 1))
    dilation_matrix = unsqueezed_dilation_matrix.squeeze(dim=0).squeeze(dim=0).type("torch.BoolTensor")

    # Superpose the class matrix to the dilation matrix and set their class values into the neighbour vector.
    superposion_matrix = ~class_matrix & dilation_matrix & valid_pixels

    neighbour_vector = labels[superposion_matrix]
    neighbour_classes, counts = torch.unique(neighbour_vector, return_counts=True)

    return neighbour_classes, counts


def neighbour_class_percentage_for_valid_pixels(
    labels: torch.IntTensor, valid_pixels: torch.BoolTensor, n_classes: int
) -> torch.FloatTensor:
    """
    Find the "percentage matrix", which indicates the neighbour class pixels percentage
    to every class. EX: class 0 has a neighbour the class 1 to 0.75 and the class 2 to 0.25.
    In the matrix ixj the i represents each class and j the neighbour to the given class.
    EX: (0.00, 0.75, 0.25)
        (0.50, 0.00, 0.50)
        (0.50, 0.50, 0.00)

    Arg:
        labels: torch.IntTensor containing a label of a class in each entry
        valid_pixels: torch.BoolTensor containing which pixels of the image are valid (
        n_classes: int number of different classes that appear in the image
    """

    class_vector = labels[valid_pixels].unique()

    percentage_matrix = torch.zeros((n_classes, n_classes))

    for label_index in class_vector:
        neighbour_classes, counts = find_neighbour_classes_valid_pixels(labels, label_index, valid_pixels)
        length = sum(counts)
        # Set the percentages in the spot matrix[class, neighbour_class]
        percentage_matrix[label_index, neighbour_classes] = counts / length

    return percentage_matrix


def count_rows_sum_eq_1(neighbour_matrix: torch.FloatTensor) -> torch.FloatTensor:
    """
    Count which classes appear in the image.
    Arg:
        neighbour_matrix: torch.FloatTensor
    """
    length = neighbour_matrix.shape[0]
    ROW_IS_0 = torch.zeros(length, dtype=torch.float)
    rows_diff_from_0 = torch.zeros(length)

    for i in range(length):
        if not (torch.equal(neighbour_matrix[i, :], ROW_IS_0)):
            rows_diff_from_0[i] += 1

    return rows_diff_from_0
