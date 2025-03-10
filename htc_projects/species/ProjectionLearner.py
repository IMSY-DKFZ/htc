# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

from htc.models.image.DatasetImage import DatasetImage
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.specular_highlights import specular_highlights_mask_lab


class ProjectionLearner(nn.Module):
    def __init__(self, config: Config, mode: str = "weights+bias", highlights_threshold: int = None):
        """
        This class can be used to learn a projection matrix that maps the spectra from one image to the spectra of another image. This is useful if the images show the same object but in different stats, for example physiological and ischemic states.

        The number of pixels do not have to be the same in both images since this optimization is carried out indirectly by enforcing that the mean and standard deviation of the spectra are similar in both images. See the ProjectionExample.ipynb notebook for an example of the usage of this class.

        Args:
            config: The configuration object which is used to load the spectral data and the valid pixels.
            mode: The general mode of the projection: set to "weights" to only use a projection matrix, "bias" to only use a bias vector, "weights+bias" to use both.
            highlights_threshold: An optional threshold to filter out specular highlight pixels in case you want to exclude them from the optimizations.
        """
        super().__init__()
        self.config = config
        self.mode = mode
        self.highlights_threshold = highlights_threshold

        self.projection_matrix = nn.Parameter(torch.eye(100, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(100, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "weights":
            x = x @ self.projection_matrix
        elif self.mode == "bias":
            x = x + self.bias
        elif self.mode == "weights+bias":
            x = x @ self.projection_matrix + self.bias
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return x

    def fit_pair(self, path_from: DataPath, path_to: DataPath, n_steps: int = 100) -> float:
        spectra_from = self._load_spectra(path_from)
        spectra_to = self._load_spectra(path_to)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        mse_loss = torch.nn.MSELoss()

        spectra_to_mean = spectra_to.mean(dim=0)
        spectra_to_std = spectra_to.std(dim=0)

        for _ in range(n_steps):
            spectra_transformed = self(spectra_from)

            loss = mse_loss(spectra_transformed.mean(dim=0), spectra_to_mean)
            loss += mse_loss(spectra_transformed.std(dim=0), spectra_to_std)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return loss.item()

    def _load_spectra(self, path: DataPath) -> torch.Tensor:
        sample = DatasetImage([path], train=False, config=self.config)[0]
        valid_pixels = sample["valid_pixels"]

        if self.highlights_threshold is not None:
            highlights = specular_highlights_mask_lab(path, threshold=self.highlights_threshold)
            valid_pixels.masked_fill_(highlights, False)

        spectra = sample["features"][valid_pixels].to(
            dtype=self.projection_matrix.dtype, device=self.projection_matrix.device
        )

        return spectra
