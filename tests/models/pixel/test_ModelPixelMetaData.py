# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np
import torch
from lightning import seed_everything

from htc.models.pixel.ModelPixelMetaData import MetadataNorm


class TestMetadataNorm:
    def test_linear_regression(self) -> None:
        seed_everything(42)
        # create dummy dataset
        dataset_size = 100000
        num_features = 100
        features = torch.rand(dataset_size, num_features)
        labels = torch.randint(2, (dataset_size,))
        metadata = torch.randint(2, (dataset_size,))

        regression_matrix = torch.empty((metadata.size(0), 3), dtype=torch.float32, device=metadata.device)
        regression_matrix[:, 0] = labels
        regression_matrix[:, 1] = metadata
        regression_matrix[:, 2] = 1
        kernel = torch.inverse(regression_matrix.T @ regression_matrix)

        # compute MetadataNorm residuals
        residuals = MetadataNorm(
            batch_size=dataset_size, cf_kernel=kernel, dataset_size=dataset_size, num_features=num_features
        ).forward(features, regression_matrix)

        # compute linear regression residuals
        from sklearn.linear_model import LinearRegression

        regressor = LinearRegression()
        x = torch.zeros(dataset_size, 2)
        x[:, 0] = metadata
        x[:, 1] = labels
        regressor.fit(x, features)  # metadata.reshape(-1, 1)
        residuals_lr = (
            features
            - np.matmul(metadata.reshape(-1, 1).numpy(), regressor.coef_[:, 0].reshape(1, -1))
            - regressor.intercept_
        )

        assert torch.allclose(residuals, residuals_lr.float(), atol=1e-05), (
            "Linear regression and metadata normalization yield different residual values!"
        )
        print()
