# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch

from htc.models.common.transforms import HTCTransformation
from htc.settings import settings
from htc.utils.blosc_compression import decompress_file
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping


class ProjectionTransform(HTCTransformation):
    def __init__(
        self,
        base_name: str,
        device: str | torch.device,
        config: Config,
        label_mode: str = "label_match",
        interpolate: bool = False,
        target_labels: list[str] = None,
        p: float = 1,
        **kwargs,
    ):
        """
        This transformation can be used to randomly apply projections to the spectra of the input images. The projections can be learned using the ProjectionLearner class.

        This can for example be used to transform physiological images to ischemic images.

        Args:
            base_name: The stem (name without the suffix) of the blosc file containing the projection matrices relative to `results_dir / projection_matrices`. The first part of the filename defines the projection mode ("weights", "bias" or "weights+bias")
            device: The device where the transformations should be applied (usually cuda).
            config: The training configuration object.
            label_mode: The label operation mode of this transform:
                - label_match: Apply the transformation only to spectra with matching labels between the segmentation mask and the label of the projection matrix. A projection matrix is randomly selected for each label and image.
                - label_match_extended: Same as label_match, but also apply the transformation to spectra with labels where no projection matrix exists. In this case, a random projection matrix is selected.
                - random: Apply the transformation to all spectra with a random projection matrix. In this mode, the label information is ignored completely.
            interpolate: If True, interpolate the original spectra with the transformed spectra with a random interpolation weight.
            target_labels: Apply the transformation only to spectra from these labels (label_match mode). If None, apply to all spectra where a corresponding projection matrix with the same label exists.
            p: Probability of applying the augmentation to an image.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.base_name = base_name
        self.label_mode = label_mode
        assert self.label_mode in [
            "label_match",
            "label_match_extended",
            "random",
        ], f"Invalid label mode: {self.label_mode}"
        self.interpolate = interpolate
        self.target_labels = target_labels
        self.p = p
        assert 0 <= self.p <= 1, f"Invalid p value: {self.p}"

        mapping = LabelMapping.from_config(config)
        variables = decompress_file(settings.results_dir / "projection_matrices" / f"{self.base_name}.blosc")

        self.projection_mode = self.base_name.split("_")[0]
        if self.projection_mode == "weights":
            self.matrices = {mapping.name_to_index(l): torch.from_numpy(t).to(device) for l, t in variables.items()}
        elif self.projection_mode == "bias":
            self.biases = {mapping.name_to_index(l): torch.from_numpy(t).to(device) for l, t in variables.items()}
        elif self.projection_mode == "weights+bias":
            self.matrices = {
                mapping.name_to_index(l): torch.from_numpy(t[:, :, :-1]).to(device) for l, t in variables.items()
            }
            self.biases = {
                mapping.name_to_index(l): torch.from_numpy(t[:, :, -1]).to(device) for l, t in variables.items()
            }
        else:
            raise ValueError(f"Invalid projection mode: {self.projection_mode}")

        n_channels = config["input/n_channels"]
        if "weights" in self.projection_mode:
            assert all(m.ndim == 3 for m in self.matrices.values()), "Incorrect number of dimensions for the matrices"
            assert all(m.shape[1:] == (n_channels, n_channels) for m in self.matrices.values()), "Invalid matrix size"
        if "bias" in self.projection_mode:
            assert all(b.ndim == 2 for b in self.biases.values()), "Incorrect number of dimensions for the bias"
            assert all(b.shape[1] == n_channels for b in self.biases.values()), "Invalid bias size"
        if self.projection_mode == "weights+bias":
            assert self.matrices.keys() == self.biases.keys(), "Matrices and biases do not use the same labels"
            assert all(
                m.shape[0] == b.shape[0] for m, b in zip(self.matrices.values(), self.biases.values(), strict=True)
            ), "Shape mismatch between matrices and biases"

        if self.label_mode == "random":
            if "weights" in self.projection_mode:
                self.matrices = torch.cat(list(self.matrices.values()), dim=0)
            if "bias" in self.projection_mode:
                self.biases = torch.cat(list(self.biases.values()), dim=0)

        self._main_variable = self.matrices if "weights" in self.projection_mode else self.biases

        if self.target_labels is not None:
            assert self.label_mode.startswith("label_match"), "target_labels can only be used with label_match modes"
            self.target_label_indices = {mapping.name_to_index(label) for label in self.target_labels}

    def __repr__(self) -> str:
        return (
            f"ProjectionTransform(base_name={self.base_name}, label_mode={self.label_mode},"
            f" projection_mode={self.projection_mode}, interpolate={self.interpolate},"
            f" target_labels={self.target_labels}, p={self.p})"
        )

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        selection = torch.rand(batch["features"].size(0)) <= self.p

        if selection.any():
            if self.label_mode == "label_match" or self.label_mode == "label_match_extended":
                self._apply_label_match(batch, selection)
            elif self.label_mode == "random":
                self._apply_random(batch, selection)
            else:
                raise ValueError(f"Invalid mode: {self.label_mode}")

        return batch

    def _apply_label_match(self, batch: dict[str, torch.Tensor], selection: torch.Tensor) -> None:
        for b in range(batch["features"].size(0)):
            if not selection[b]:
                continue

            labels = batch["labels"][b].unique().tolist()
            if self.target_labels is not None:
                labels = [l for l in labels if l in self.target_label_indices]

            if self.label_mode != "label_match_extended":
                labels = [l for l in labels if l in self._main_variable]

            if len(labels) == 0:
                continue

            for label_index in labels:
                if label_index in self._main_variable:
                    matrix_label_index = label_index
                else:
                    assert self.label_mode == "label_match_extended", "Invalid mode"

                    # Use matrix from any other label
                    matrix_label_index = torch.randint(0, len(self._main_variable), (1,)).item()
                    matrix_label_index = list(self._main_variable.keys())[matrix_label_index]

                if self.label_mode == "label_match":
                    assert label_index == matrix_label_index, "labels must match in label_match mode"

                matrix_index = torch.randint(0, self._main_variable[matrix_label_index].size(0), (1,)).item()

                # Only spectra from the current target label
                label_selection = batch["labels"][b] == label_index
                selected_features = batch["features"][b, label_selection]

                if self.projection_mode == "weights":
                    transformed = selected_features @ self.matrices[matrix_label_index][matrix_index]
                elif self.projection_mode == "bias":
                    transformed = selected_features + self.biases[matrix_label_index][matrix_index]
                elif self.projection_mode == "weights+bias":
                    transformed = (
                        selected_features @ self.matrices[matrix_label_index][matrix_index]
                        + self.biases[matrix_label_index][matrix_index]
                    )
                else:
                    raise ValueError(f"Invalid projection mode: {self.projection_mode}")
                transformed = transformed.to(dtype=batch["features"].dtype)

                if self.interpolate:
                    # Weights must be broadcastable
                    weights = torch.rand((1,), device=transformed.device)  # [B]
                    weights = weights.reshape(weights.shape + (1,) * len(transformed.shape[1:]))  # [B, 1]

                    batch["features"][b, label_selection] = torch.lerp(
                        batch["features"][b, label_selection], transformed, weights
                    )
                else:
                    batch["features"][b, label_selection] = transformed

    def _apply_random(self, batch: dict[str, torch.Tensor], selection: torch.Tensor) -> None:
        # Select one matrix per image
        matrix_indices = torch.randint(
            0, self._main_variable.size(0), (selection.sum(),), device=batch["features"].device
        )

        selected_features = batch["features"][selection]
        original_shape = selected_features.shape
        selected_features = selected_features.reshape(
            selected_features.size(0),
            selected_features.size(1) * selected_features.size(2),
            selected_features.size(3),
        )

        if self.projection_mode == "weights":
            transformed = torch.bmm(selected_features, self.matrices[matrix_indices]).to(dtype=batch["features"].dtype)
        elif self.projection_mode == "bias":
            transformed = selected_features + self.biases[matrix_indices].unsqueeze(1)
        elif self.projection_mode == "weights+bias":
            transformed = torch.bmm(selected_features, self.matrices[matrix_indices]) + self.biases[
                matrix_indices
            ].unsqueeze(1)
        else:
            raise ValueError(f"Invalid projection mode: {self.projection_mode}")
        transformed = transformed.reshape(original_shape).to(dtype=batch["features"].dtype)

        if self.interpolate:
            # Weights must be broadcastable
            weights = torch.rand(transformed.size(0), device=transformed.device)  # [B]
            weights = weights.reshape(weights.shape + (1,) * len(transformed.shape[1:]))  # [B, 1, 1, 1]

            batch["features"][selection] = torch.lerp(batch["features"][selection], transformed, weights)
        else:
            batch["features"][selection] = transformed
