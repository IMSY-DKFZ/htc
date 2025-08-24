# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from htc.models.common.transforms import HTCTransformation
from htc.settings import settings
from htc.utils.blosc_compression import decompress_file
from htc.utils.Config import Config
from htc.utils.general import sha256_file
from htc.utils.LabelMapping import LabelMapping


class ProjectionTransform(HTCTransformation):
    known_projection_matrices = MappingProxyType({
        "weights+bias_ICG_pig_subjects=P062,P072,P076,P113": {
            "sha256": "e39d2ce939e9fa1d277fc7dcc5ff080b48d79182d274f5edae6c6ee2e6491783",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/projection_matrices/weights+bias_ICG_pig_subjects=P062,P072,P076,P113.blosc",
        },
        "weights+bias_ICG_rat_subjects=R043,R048": {
            "sha256": "7a76aa499e5f97a2e4dc56c15075dfbafd24a17e8c31b28861727b70b3a5d2f4",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/projection_matrices/weights+bias_ICG_rat_subjects=R043,R048.blosc",
        },
        "weights+bias_malperfusion_pig_kidney=P091,P095,P097,P098+aortic": {
            "sha256": "f14ec6b2b39bbe98005246a8e31c708938ed7cab9829723199da23754c6ec98e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/projection_matrices/weights+bias_malperfusion_pig_kidney=P091,P095,P097,P098+aortic.blosc",
        },
        "weights+bias_malperfusion_rat_subjects=R017,R019,R025,R029": {
            "sha256": "8bf548a0eb074ff7d2fcd1abc70f44f3cfcd289da0c215bef855c4bc5959448e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/projection_matrices/weights+bias_malperfusion_rat_subjects=R017,R019,R025,R029.blosc",
        },
    })

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

        An example use case is to transform physiological organs to ischemic organs.

        >>> _ = torch.manual_seed(0)
        >>> batch = {
        ...     "features": torch.ones(1, 3, 3, 100),
        ...     "labels": torch.tensor([
        ...         [0, 1, 1],
        ...         [0, 1, 1],
        ...         [1, 1, 1],
        ...     ]).unsqueeze(0),
        ... }
        >>> mapping = LabelMapping({"colon": 0, "liver": 1})
        >>> transform = ProjectionTransform(
        ...     "weights+bias_ICG_rat_subjects=R043,R048",
        ...     "cpu",
        ...     Config({"label_mapping": mapping, "input/n_channels": 100}),
        ...     target_labels=["liver"],
        ... )
        >>> transform(batch)["features"][0, :, :, 0].round(decimals=1)
        tensor([[1.0000, 1.2000, 1.2000],
                [1.0000, 1.2000, 1.2000],
                [1.2000, 1.2000, 1.2000]])

        Note: Since this transformation is applied separately for each pixel and only consists of simple matrix multiplications and vector additions, it is best suited to be used as a GPU augmentation.

        The following table lists which projection matrices are publicly available for this transformation:
        | base name | projection mode | experiment type | source species |
        | ----------- | ----------- | ----------- | ----------- |
        | weights+bias_ICG_pig_subjects=P062,P072,P076,P113 | weights+bias | ICG | pig |
        | weights+bias_ICG_rat_subjects=R043,R048 | weights+bias | ICG | rat |
        | weights+bias_malperfusion_pig_kidney=P091,P095,P097,P098+aortic | weights+bias | malperfusion | pig |
        | weights+bias_malperfusion_rat_subjects=R017,R019,R025,R029 | weights+bias | malperfusion | rat |

        All projection matrices expect spectral data with 100 channels from 500 nm to 1000 nm.

        Args:
            base_name: The stem (name without the suffix) of the blosc file containing the projection matrices. The first part of the filename defines the projection mode ("weights", "bias" or "weights+bias"). The file must either be relative to `results_dir / projection_matrices` directory or the base name must match with a name from the table above (in that case the projection matrix will be automatically downloaded).
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
        file_path_local = settings.results_dir / "projection_matrices" / f"{self.base_name}.blosc"
        if file_path_local.exists():
            variables = decompress_file(file_path_local)
        else:
            assert self.base_name in self.known_projection_matrices, (
                f"The projection matrix {self.base_name} is not available locally and is not a known projection matrix"
            )

            hub_dir = Path(torch.hub.get_dir()) / "htc_projection_matrices"
            hub_dir.mkdir(exist_ok=True, parents=True)

            hub_path = hub_dir / f"{self.base_name}.blosc"
            if hub_path.exists():
                variables = decompress_file(hub_path)
            else:
                try:
                    # Try to download from remote
                    file_path_remote = f"https://e130-hyperspectal-tissue-classification.s3.dkfz.de/projection_matrices/{self.base_name}.blosc"
                    torch.hub.download_url_to_file(file_path_remote, hub_path)

                    hash_file = sha256_file(hub_path)
                    if self.known_projection_matrices[self.base_name]["sha256"] != hash_file:
                        settings.log.error(
                            f"The hash of the local file {hub_path} does not match the expected hash. The download from {file_path_remote} is likely corrupted. Please check the file manually (e.g., for a broken file size) and delete it to re-trigger the download"
                        )
                    else:
                        settings.log.info(f"Successfully downloaded the projection matrix for {self.base_name}")

                    variables = decompress_file(hub_path)
                except FileNotFoundError as error:
                    raise FileNotFoundError(
                        f"Could not find projection matrices for {self.base_name}, neither locally at {file_path_local} nor remotely at {file_path_remote}"
                    ) from error

        if self.target_labels is not None:
            # Remove any organ data which we don't need (to save memory)
            for l in list(variables.keys()):
                if l not in self.target_labels:
                    del variables[l]

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


class ProjectionTransformMultiple(HTCTransformation):
    def __init__(self, projections: list[dict[str, Any]], p: float = 1, **kwargs):
        """
        This meta transformation can be used to apply multiple projection transformations with different parameters (e.g., malperfusion and ICG) to the same batch.

        For each image where a projection should be applied, one of the transformations is randomly selected and can transform the image. One image is only transformed once (this is usually not necessary, especially if the OrganTransplantation augmentation is applied afterwards).

        Args:
            projections: A list of dictionaries containing the keywords arguments for the ProjectionTransform classes.
            p: Probability of applying any projection transformation to an image. Please note that it is also possible to set a `p` parameter for the individual projections.
            **kwargs: Additional keyword arguments passed to the parent class and to the individual ProjectionTransform classes.
        """
        super().__init__(**kwargs)

        self.transforms = []
        for projection_kwargs in projections:
            projection_kwargs = kwargs | projection_kwargs  # User arguments have precedence
            self.transforms.append(ProjectionTransform(**projection_kwargs))

        self.p = p
        assert 0 <= self.p <= 1, f"Invalid p value: {self.p}"

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        selection = torch.rand(batch["features"].size(0)) <= self.p

        if selection.any():
            for b in range(batch["features"].size(0)):
                if not selection[b]:
                    continue

                transform_index = torch.randint(0, len(self.transforms), (1,)).item()
                batch_image = {k: v[b] for k, v in batch.items()}
                self.transforms[transform_index](batch_image)

        return batch

    def __repr__(self) -> str:
        res = "ProjectionTransformMultiple(projections=[\n"
        res += "\n".join([f"\t{t}," for t in self.transforms])
        res += "\n])"

        return res
