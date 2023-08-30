# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import os

from htc.settings import settings
from htc.utils.MultiPath import MultiPath
from htc.utils.unify_path import unify_path


class SettingContext:
    def __init__(self):
        # Color settings for the plots
        self.modality_names = {
            "hsi": "HSI",
            "param": "TPI",
            "rgb": "RGB",
        }
        self.model_colors = {
            "pixel": "#FF8C00",
            "superpixel_classification": "#E6003D",
            "patch_32": "#4169E1",
            "patch_64": "#808000",
            "image": "#800080",
        }
        self.modality_colors = {
            "RGB": "indianred",
            "TPI": "lightseagreen",
            "HSI": "#ffbb78",
        }

        self.isolation_colors = {
            "HSI (organ isolation 0)": "#177F7A",
            "HSI (organ isolation cloth)": "#177F7A",
            "HSI (reference)": "#00A8B8",
        }
        self.model_type_colors = {
            "HSI (organ removal)": "#00A8B8",
            "HSI (reference)": "lightseagreen",
            "RGB (organ removal)": "indianred",
            "RGB (reference)": "#994545",
        }
        self.model_comparison_colors = {
            "HSI (organ removal 0)": "#00A8B8",
            "HSI (organ removal cloth)": "#177F7A",
            "HSI (organ isolation 0)": "#00A8B8",
            "HSI (organ isolation cloth)": "#177F7A",
            "HSI (reference)": "lightseagreen",
            "RGB (organ removal 0)": "#994545",
            "RGB (organ removal cloth)": "#9E2121",
            "RGB (organ isolation 0)": "#994545",
            "RGB (organ isolation cloth)": "#9E2121",
            "RGB (reference)": "indianred",
        }
        self.network_colors = {
            "baseline#HSI": "#508587",
            "baseline#RGB": "#66ABAD",
            "surgical_augmentations#HSI": "#A39F5D",
            "surgical_augmentations#RGB": "#C9C473",
            "baseline": "#508587",
            "organ_transplantation": "#A39F5D",
        }
        self.augmentation_colors = {
            "organ_transplantation": self.network_colors["surgical_augmentations#HSI"],
            "cut_mix": "#CD5C5C",
            "jigsaw": "#277EDB",
            "random_erasing": "#1FA83A",
            "hide_and_seek": "#A222A8",
            "elastic": "#F4A460",
            "baseline": self.network_colors["baseline#HSI"],
        }

        # This also specifies which tasks we include in the paper (e.g. box plots)
        self.task_name_mapping = {
            "semantic": "original",
            "isolation_0": "isolation_zero",
            "isolation_cloth": "isolation_bgr",
            "masks_isolation": "isolation_real",
            "removal_0": "removal_zero",
            "removal_cloth": "removal_bgr",
            "no-glove": "no-occlusion",
            "glove": "occlusion",
        }

        self.transforms = {
            "organ_transplantation": {
                "class": "htc.context.context_transforms>OrganTransplantation",
            },
            "cut_mix": {
                "class": "htc.context.context_transforms>RectangleOrganTransplantation",
            },
            "jigsaw": {
                "class": "htc.context.context_transforms>RandomJigsaw",
                "patch_size": [[96, 128], [60, 80], [48, 64], [30, 40], [24, 32]],
            },
            "elastic": {
                "class": "KorniaTransform",
                "transformation_name": "RandomElasticTransform",
                "padding_mode": "reflection",
                "alpha": [0.7, 0.7],
                "sigma": [16, 16],
            },
            "random_erasing": {
                "class": "htc.context.context_transforms>RandomRectangleErasing",
                "fill_value": "0",
            },
            "hide_and_seek": {
                "class": "htc.context.context_transforms>HideAndSeek",
                "fill_value": "0",
                "proportion": [0.2, 0.8],
                "patch_size": [[96, 128], [60, 80], [48, 64], [30, 40], [24, 32]],
            },
        }

        # Images from the masks dataset showing organs in isolation
        self.masks_isolation_dataset = {
            "stomach": [
                "P065#2020_06_19_21_02_33",  # unknown
                "P060#2020_05_14_20_53_16",  # train
                "P043#2019_12_20_10_40_58",  # test
                "P043#2019_12_20_10_39_09",  # test
                "P042#2019_12_15_11_49_40",  # unknown
                "P042#2019_12_15_11_48_50",  # unknown
            ],
            "small_bowel": [
                "P041#2019_12_14_13_31_35",  # train
                "P043#2019_12_20_11_24_58",  # test
                "P044#2020_02_01_12_57_21",  # train
                "P045#2020_02_05_13_10_15",  # train
                "P054#2020_03_10_18_37_00",  # unknown
            ],
            "colon": [
                "P041#2019_12_14_12_29_18",  # train
                "P042#2019_12_15_11_00_17",  # unknown
                "P043#2019_12_20_11_31_21",  # test
            ],
            "liver": [
                "P041#2019_12_14_13_35_06",  # train
                "P042#2019_12_15_11_08_04",  # unknown
                "P043#2019_12_20_10_16_32",  # test
            ],
            "gallbladder": [
                "P041#2019_12_14_12_23_01",  # train
                "P042#2019_12_15_11_15_55",  # unknown
                "P043#2019_12_20_10_20_39",  # test
            ],
            "pancreas": [
                "P041#2019_12_14_12_26_30",  # train
                "P042#2019_12_15_11_12_23",  # unknown
                "P043#2019_12_20_11_41_06",  # test
                "P045#2020_02_05_16_36_55",  # train
            ],
            "kidney": [
                "P041#2019_12_14_13_42_00",  # train
                "P042#2019_12_15_11_18_48",  # unknown
                "P043#2019_12_20_11_12_37",  # test
                "P044#2020_02_01_10_23_29",  # train
                "P047#2020_02_07_19_00_03",  # train
                "P048#2020_02_08_11_01_21",  # train
                "P049#2020_02_11_19_37_58",  # train
                "P052#2020_03_04_13_30_47",  # unknown
            ],
            "spleen": [
                "P041#2019_12_14_12_10_33",  # train
                "P042#2019_12_15_10_50_14",  # unknown
                "P043#2019_12_20_10_24_22",  # test
            ],
            "bladder": [
                "P042#2019_12_15_11_02_18",  # unknown
                "P043#2019_12_20_11_33_41",  # test
                "P041#2019_12_14_12_20_17",  # train
            ],
            "omentum": [
                "P042#2019_12_15_10_51_29",  # unknown
                "P042#2019_12_15_10_51_49",  # unknown
            ],
            "lung": [
                "P043#2019_12_20_12_50_21",  # test
                "P043#2019_12_20_12_56_00",  # test
                "P044#2020_02_01_17_39_46",  # train
                "P045#2020_02_05_16_53_26",  # train
                "P058#2020_05_13_20_50_50",  # train
            ],
            "heart": [
                "P043#2019_12_20_12_42_12",  # test
                "P043#2019_12_20_12_45_06",  # test
                "P045#2020_02_05_16_55_22",  # train
                "P058#2020_05_13_20_53_54",  # train
            ],
            "skin": [
                "P041#2019_12_14_10_50_54",  # train
                "P041#2019_12_14_12_06_34",  # train
                "P042#2019_12_15_10_14_19",  # unknown
                "P042#2019_12_15_10_16_19",  # unknown
                "P045#2020_02_05_10_16_23",  # train
                "P045#2020_02_05_10_18_37",  # train
                "P047#2020_02_07_17_03_35",  # train
                "P047#2020_02_07_17_09_00",  # train
                "P048#2020_02_08_10_03_45",  # train
                "P048#2020_02_08_10_07_50",  # train
                "P051#2020_03_03_19_02_24",  # unknown
                "P051#2020_03_03_19_04_06",  # unknown
                "P052#2020_03_04_12_22_54",  # unknown
                "P052#2020_03_04_12_31_04",  # unknown
                "P053#2020_03_06_11_09_56",  # unknown
                "P053#2020_03_06_11_13_21",  # unknown
                "P054#2020_03_10_17_50_00",  # unknown
                "P054#2020_03_10_18_06_00",  # unknown
                "P055#2020_03_11_10_35_25",  # unknown
                "P055#2020_03_11_10_35_55",  # unknown
                "P059#2020_05_14_11_19_00",  # train
                "P059#2020_05_14_11_20_31",  # train
                "P063#2020_05_28_15_48_10",  # unknown
                "P063#2020_05_28_15_50_51",  # unknown
                "P064#2020_05_29_10_08_39",  # unknown
                "P064#2020_05_29_10_09_28",  # unknown
                "P065#2020_06_19_18_49_59",  # unknown
                "P065#2020_06_19_18_51_28",  # unknown
                "P066#2020_07_07_08_47_39",  # unknown
                "P066#2020_07_07_08_49_35",  # unknown
                "P067#2020_07_09_17_44_42",  # unknown
                "P068#2020_07_20_17_18_47",  # test
                "P068#2020_07_20_17_23_35",  # test
                "P071#2020_08_05_11_04_57",  # train
                "P071#2020_08_05_11_07_56",  # train
            ],
            "muscle": [
                "P053#2020_03_06_17_15_27",  # unknown
                "P053#2020_03_06_17_16_32",  # unknown
                "P053#2020_03_06_17_17_16",  # unknown
                "P060#2020_05_14_21_58_18",  # train
            ],
            "peritoneum": [],
            "major_vein": [
                "P060#2020_05_14_19_56_29",  # train
                "P058#2020_05_13_19_12_55",  # train
                "P058#2020_05_13_19_22_11",  # train
            ],
            "kidney_with_Gerotas_fascia": [
                "P078#2021_02_07_11_43_13",  # unknown, Cam shift
                "P078#2021_02_07_12_15_52",  # unknown, Cam shift
                "P080#2021_02_14_11_14_41",  # unknown, Cam shift
            ],
        }

        # Make sure that the correct annotation is used for the masks images (the semantic and not the polygon annotations)
        for label_name, names in self.masks_isolation_dataset.items():
            self.masks_isolation_dataset[label_name] = [f"{n}@semantic#annotator5" for n in names]

        # other_interesting_images = ["P058#2020_05_13_19_33_36", "P046#2020_02_07_09_32_20", "P058#2020_05_13_22_20_25"]
        # paper_exclusion = [
        #     "P041#2019_12_14_12_20_17",
        #     "P045#2020_02_05_16_53_26",
        #     "P058#2020_05_13_20_50_50",
        #     "P045#2020_02_05_16_55_22",
        #     "P058#2020_05_13_20_53_54",
        #     "P058#2020_05_13_19_12_55",
        #     "P058#2020_05_13_19_22_11",
        # ]
        # wrong_camera = ["P078#2021_02_07_11_43_13", "P078#2021_02_07_12_15_52", "P080#2021_02_14_11_14_41"]

        self.real_datasets = {
            "masks_isolation": self.masks_isolation_dataset,
        }

        self._results_dir = None

    @property
    def results_dir(self) -> MultiPath:
        if self._results_dir is None:
            if _path_env := os.getenv("PATH_HTC_RESULTS_CONTEXT", False):
                self._results_dir = unify_path(_path_env)
            else:
                # If no path is set, we just use the default results directory
                self._results_dir = settings.results_dir
                settings.log.info(
                    "The environment variable PATH_HTC_RESULTS_CONTEXT is not set. Files for the context"
                    f" project will be written to {self._results_dir.find_best_location()}"
                )

        return self._results_dir

    @property
    def paper_dir(self) -> MultiPath:
        target_dir = self.results_dir / "paper"
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    @property
    def best_transform_runs(self) -> dict[str, MultiPath]:
        # Best runs for each transformation (found via find_best_transform_run())
        return {
            "organ_transplantation": settings.training_dir / "image/2023-02-08_14-48-02_organ_transplantation_0.8",
            "cut_mix": settings.training_dir / "image/2023-02-08_17-08-57_cut_mix_1",
            "jigsaw": settings.training_dir / "image/2023-02-16_21-17-59_jigsaw_0.8",
            "random_erasing": settings.training_dir / "image/2023-02-08_12-06-44_random_erasing_0.4",
            "hide_and_seek": settings.training_dir / "image/2023-02-16_15-34-51_hide_and_seek_1",
            "elastic": settings.training_dir / "image/2023-02-08_09-40-59_elastic_0.6",
        }

    @property
    def best_transform_runs_rgb(self) -> dict[str, MultiPath]:
        return {
            "organ_transplantation": settings.training_dir / "image/2023-01-29_11-31-04_organ_transplantation_0.8_rgb",
        }

    @property
    def glove_runs(self) -> dict[str, MultiPath]:
        return {
            "baseline": settings.training_dir / "image/2023-02-21_23-14-44_glove_baseline",
            "organ_transplantation": (
                settings.training_dir / "image/2023-02-21_23-14-55_glove_organ_transplantation_0.8"
            ),
            "cut_mix": settings.training_dir / "image/2023-02-23_19-07-27_glove_cut_mix_1.0",
            "jigsaw": settings.training_dir / "image/2023-02-22_12-31-26_glove_jigsaw_0.8",
            "elastic": settings.training_dir / "image/2023-02-22_12-31-26_glove_elastic_0.6",
            "random_erasing": settings.training_dir / "image/2023-02-22_12-31-26_glove_random_erasing_0.4",
            "hide_and_seek": settings.training_dir / "image/2023-02-22_12-31-26_glove_hide_and_seek_1.0",
        }

    @property
    def glove_runs_rgb(self) -> dict[str, MultiPath]:
        return {
            "baseline": settings.training_dir / "image/2023-02-24_12-07-15_glove_baseline_rgb",
            "organ_transplantation": (
                settings.training_dir / "image/2023-02-24_14-27-15_glove_organ_transplantation_0.8_rgb"
            ),
        }


settings_context = SettingContext()
