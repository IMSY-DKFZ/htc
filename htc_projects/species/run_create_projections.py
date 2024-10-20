# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.blosc_compression import compress_file
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping
from htc.utils.parallel import p_map
from htc_projects.species.ProjectionLearner import ProjectionLearner
from htc_projects.species.settings_species import settings_species
from htc_projects.species.tables import ischemic_table


class ProjectionPairs:
    def __init__(self, mode: str, highlights_threshold: int | None, kidney_spec: Path):
        self.mode = mode
        self.highlights_threshold = highlights_threshold
        self.kidney_spec = kidney_spec

    def compute_species_projections(self, species: str) -> None:
        settings.log.info(f"Computing projections for the {species} species")
        variables = {}
        meta = {}
        df = ischemic_table()

        if species == "pig":
            spec = DataSpecification(self.kidney_spec)
            paths_kidney = spec.paths("train")

            phase_type_matrices, phase_type_meta = self.compute_phase_type_projections(paths_kidney, ["kidney"])
            variables |= phase_type_matrices
            meta |= phase_type_meta

            df = df[
                (df.species_name == species)
                & (df.clamping_location == "aorta")
                & (df.label_name.isin(settings_species.pig_aortic_labels))
                & (df.reperfused == False)  # noqa: E712
            ]
            assert df.subject_name.nunique() == 4, "Missing some subjects"
            assert set(df.label_name.unique()) == set(
                settings_species.pig_aortic_labels
            ), f"Labels are missing: {df.label_name.unique()}"

            paths = DataPath.from_table(df)

            aortic_matrices, aortic_meta = self.compute_phase_type_projections(
                paths, settings_species.pig_aortic_labels
            )
            variables |= aortic_matrices
            meta |= aortic_meta

            kidney_pigs = sorted({p.subject_name for p in paths_kidney})
            name_datasets = f"kidney={','.join(kidney_pigs)}+aortic"
        elif species == "rat":
            labels = ["stomach", "small_bowel", "colon", "liver", "kidney", "spleen"]
            subjects = ["R017", "R019", "R025", "R029"]

            df = df[
                (df.species_name == species)
                & (df.subject_name.isin(subjects))
                & (df.label_name.isin(labels))
                & (df.reperfused == False)  # noqa: E712
                & (~df.baseline_dataset)
            ]
            assert (
                df.clamping_location.nunique() == 2
            ), f"Aorta and organ clamping locations are needed for rats: {df.clamping_location.unique()}"
            assert set(df.label_name.unique()) == set(labels), f"Labels are missing: {df.label_name.unique()}"

            paths = DataPath.from_table(df)

            phase_type_matrices, phase_type_meta = self.compute_phase_type_projections(paths, labels)
            variables |= phase_type_matrices
            meta |= phase_type_meta

            name_datasets = f"subjects={','.join(subjects)}"
        elif species == "human":
            labels = ["kidney"]

            df = df[(df.species_name == species) & (df.label_name.isin(labels)) & (df.perfusion_state != "unclear")]
            assert not pd.isna(df.perfusion_state).any(), "Perfusion state is missing for some images"
            assert sorted(df.perfusion_state.unique()) == [
                "malperfused",
                "physiological",
            ], "Perfusion states are not complete"

            paths_physiological = DataPath.from_table(df[df.perfusion_state == "physiological"])
            paths_malperfused = DataPath.from_table(df[df.perfusion_state == "malperfused"])

            phase_type_matrices, phase_type_meta = self.create_pairs(paths_physiological, paths_malperfused, labels)
            variables |= phase_type_matrices
            meta |= phase_type_meta

            name_datasets = "subjects=all"
        else:
            raise ValueError(f"Unknown species: {species}")

        name_suffix = f"_highlights={self.highlights_threshold}" if self.highlights_threshold is not None else ""
        name = f"{self.mode}_{species}_{name_datasets}{name_suffix}"

        target_dir = settings_species.results_dir / "projection_matrices"
        target_dir.mkdir(parents=True, exist_ok=True)

        compress_file(target_dir / f"{name}.blosc", variables)
        with (target_dir / f"{name}.json").open("w") as f:
            json.dump(meta, f)

    def compute_phase_type_projections(
        self, paths: list[DataPath], label_names: list[str]
    ) -> tuple[np.ndarray, list[dict[str, str | float]]]:
        paths_physiological = []
        paths_ischemic = []
        for path in paths:
            if path.meta("phase_type") == "physiological":
                paths_physiological.append(path)
            else:
                paths_ischemic.append(path)

        return self.create_pairs(paths_physiological, paths_ischemic, label_names)

    def create_pairs(
        self, paths_physiological: list[DataPath], paths_ischemic: list[DataPath], label_names: list[str]
    ) -> tuple[np.ndarray, list[dict[str, str | float]]]:
        assert len(paths_physiological) > 0, "No physiological images found."
        assert len(paths_ischemic) > 0, "No ischemic images found."

        assert sorted(set(paths_physiological + paths_ischemic)) == sorted(
            paths_physiological + paths_ischemic
        ), "No duplicates allowed."
        settings.log.info(f"Physiological images: {len(paths_physiological)}")
        settings.log.info(f"Ischemic images: {len(paths_ischemic)}")

        variables_labels = {}
        meta_labels = {}
        for label_name in label_names:
            mapping = LabelMapping({label_name: 0}, unknown_invalid=True)
            pairs = []
            for p_physiological in paths_physiological:
                if label_name not in p_physiological.annotated_labels():
                    continue

                for p_ischemic in paths_ischemic:
                    if label_name not in p_ischemic.annotated_labels():
                        continue

                    pairs.append((p_physiological, p_ischemic))

            results = p_map(
                partial(self._compute_pair, mapping=mapping),
                pairs,
                num_cpus=4,
                task_name=f"Label: {label_name}...",
            )
            variables_labels[label_name] = np.stack([r[0] for r in results])
            meta_labels[label_name] = [r[1] for r in results]

        return variables_labels, meta_labels

    def _compute_pair(
        self, pair: tuple[DataPath, DataPath], mapping: LabelMapping
    ) -> tuple[np.ndarray, dict[str, str | float]]:
        config = Config({"label_mapping": mapping, "input/preprocessing": "L1"})

        projection = ProjectionLearner(config=config, mode=self.mode, highlights_threshold=self.highlights_threshold)
        projection.cuda()
        loss = projection.fit_pair(*pair)

        if self.mode == "weights":
            variables = projection.projection_matrix.detach().cpu().numpy()
        elif self.mode == "bias":
            variables = projection.bias.detach().cpu().numpy()
        elif self.mode == "weights+bias":
            variables = np.concatenate(
                [
                    projection.projection_matrix.detach().cpu().numpy(),
                    np.expand_dims(projection.bias.detach().cpu().numpy(), axis=1),
                ],
                axis=1,
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return variables, {
            "name_physiological": pair[0].image_name_annotations(),
            "name_ischemic": pair[1].image_name_annotations(),
            "loss": loss,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learn projections between every pair of physiological and ischemic images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--species",
        type=str,
        nargs="+",
        choices=["pig", "rat", "human"],
        default=["pig", "rat", "human"],
        help=(
            "One or more species names for which the projections should be compted for. If not given, projections for"
            " all species will be computed."
        ),
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("kidney_projection_train=P091,P095,P097,P098.json"),
        help="Name or path to the data specification file which defines the train kidney set.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="weights+bias",
        choices=["weights", "bias", "weights+bias"],
        help="The mode of the projection.",
    )
    parser.add_argument(
        "--highlights-threshold",
        type=int,
        default=None,
        help=(
            "If set, detect specular highlights with the given threshold (via LAB space) and ignore the corresponding"
            " pixels in the projection learning."
        ),
    )
    args = parser.parse_args()

    projection_pairs = ProjectionPairs(
        mode=args.mode, highlights_threshold=args.highlights_threshold, kidney_spec=args.spec
    )
    for species in args.species:
        projection_pairs.compute_species_projections(species)
