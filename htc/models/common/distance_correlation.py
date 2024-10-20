# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import torch

from htc.cpp import hierarchical_bootstrapping
from htc.settings import settings


def distance_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculates the biased, non-squared distance correlation as this function: https://dcor.readthedocs.io/en/latest/functions/dcor.distance_correlation.html#dcor.distance_correlation A good description is available in this answer: https://stats.stackexchange.com/a/183930.

    >>> x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]])
    >>> y = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
    >>> distance_correlation(x, x).item()
    1.0
    >>> distance_correlation(x, y).item()  # doctest: +ELLIPSIS
    0.526640...
    >>> distance_correlation(y, y).item()
    1.0

    Args:
        x: First data matrix with rows denoting instances and columns variables.
        y: second data matrix with rows denoting instances and columns variables.

    Returns: Distance correlation value € [0;1].
    """
    assert 1 <= x.dim() <= 2 and 1 <= y.dim() <= 2, "x and y must be one- or two-dimensional"
    assert x.device == y.device, "Both tensors must be on the same device"

    if x.dim() == 1:
        x = x.unsqueeze(dim=1)  # Matrix with one variable
    if y.dim() == 1:
        y = y.unsqueeze(dim=1)
    if not x.is_floating_point():
        x = x.float()
    if not y.is_floating_point():
        y = y.float()

    # Calculate distance matrices for x and y
    dx = torch.cdist(x, x)
    dy = torch.cdist(y, y)

    # Double center the matrices. This is usually done via
    #   A = dx - dx.mean(dim=0) - dx.mean(dim=1) + dx.mean()
    # but optimized here making use of the fact that dx and dy are symmetric
    dx_mean_vec = dx.mean(dim=0)
    dy_mean_vec = dy.mean(dim=0)
    dx_mean = dx_mean_vec.mean()
    dy_mean = dy_mean_vec.mean()
    dx_center = dx - dx_mean_vec.reshape(1, -1) - dx_mean_vec.reshape(-1, 1) + dx_mean
    dy_center = dy - dy_mean_vec.reshape(1, -1) - dy_mean_vec.reshape(-1, 1) + dy_mean

    # Calculate covariance matrices
    n_square = x.size(0) ** 2
    dcov2_xy = (dx_center * dy_center).sum() / n_square
    dcov2_xx = (dx_center * dx_center).sum() / n_square
    dcov2_yy = (dy_center * dy_center).sum() / n_square

    # Distance correlation
    dcor = dcov2_xy.sqrt() / torch.sqrt(dcov2_xx.sqrt() * dcov2_yy.sqrt())

    return torch.nan_to_num(dcor)


def distance_correlation_features(df: pd.DataFrame, device: str = "cuda") -> pd.DataFrame:
    """
    Calculates the distance correlation between some features and the camera domain. The calculation is conducted via hierarchical bootstrapping stratified by label ensuring equal distribution of cameras, subjects and images in each bootstrap.

    Note: Make sure you wrap your code in an autocasting region if you want to work with float16 features.

    Args:
        df: Table with all the necessary data.
        device: PyTorch device string where the computation should take place.

    Returns: Table with distance_correlation results across bootstraps for each label.
    """
    assert all(
        c in df for c in ["camera_index", "subject_index", "label_name", "features"]
    ), "Not all required columns available"
    assert "image_index" not in df, "The image index will be re-created by this function (it must be consecutive)"

    # Image index is just the row index of the table
    df = df.reset_index(drop=True)
    df["image_index"] = df.index

    # We need matrices for the features and the camera labels
    n_cams = df["camera_index"].max() + 1
    x_all = np.stack(df["features"].values)
    y_all = np.eye(n_cams, dtype=x_all.dtype)[df["camera_index"].values]

    x_all = torch.from_numpy(x_all).to(device)
    y_all = torch.from_numpy(y_all).to(device)

    if x_all.dtype == torch.float16 and not torch.is_autocast_enabled():
        settings.log.warning(
            "Features are float16 but autocast is not enabled. Results may be incorrect or even contain only zeros"
        )

    rows = []
    for label_name in df["label_name"].unique():
        df_labels = df.query("label_name == @label_name")[["camera_index", "subject_index", "image_index"]]

        # Create a mapping (camera_index -> subject_index -> image_index) for the bootstrapping
        mapping = {}
        for row in df_labels.values:
            camera_index = row[0]
            subject_index = row[1]
            image_index = row[2]

            if camera_index not in mapping:
                mapping[camera_index] = {}

            if subject_index not in mapping[camera_index]:
                mapping[camera_index][subject_index] = []

            mapping[camera_index][subject_index].append(image_index)

        # Create the bootstrap matrix (every row denotes a bootstrap case)
        n_subjects = 500
        n_images = 1
        n_bootstraps = 100
        bootstraps = hierarchical_bootstrapping(mapping, n_subjects, n_images, n_bootstraps).to(device)

        # Distance correlation for each bootstrap (chunked to reduce memory consumption)
        dcors = []
        for chunk in bootstraps.tensor_split(10):
            dcors.append(torch.vmap(lambda indices: distance_correlation(x_all[indices], y_all[indices]))(chunk))
        dcors = torch.cat(dcors)

        rows.append({
            "label_name": label_name,
            "dcor": dcors.tolist(),
            "dcor_mean": dcors.mean().item(),
            "dcor_std": dcors.std().item(),
        })

    return pd.DataFrame(rows)
