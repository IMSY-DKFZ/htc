# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import os

from htc.settings import settings
from htc.utils.LabelMapping import LabelMapping
from htc.utils.MultiPath import MultiPath
from htc.utils.unify_path import unify_path


class SettingsSeg:
    def __init__(self):
        """Settings for the segmentation task."""
        self.label_mapping = LabelMapping(
            {
                # Some class labels are considered as belonging to the background class
                "background": 0,
                "blue_cloth": 0,
                "blanket": 0,
                "foil": 0,
                "metal": 0,
                "white_compress": 0,
                "anorganic_artifact": 0,
                "tube": 0,
                "syringe": 0,
                "glove": 0,
                "abdominal_linen": 0,
                # Main classes
                "heart": 1,
                "lung": 2,
                "liver": 3,
                "colon": 4,
                "small_bowel": 5,
                "stomach": 6,
                "spleen": 7,
                "gallbladder": 8,
                "bladder": 9,
                "omentum": 10,
                "peritoneum": 11,
                "skin": 12,
                "fat": 13,
                "pancreas": 14,
                "muscle": 15,
                "kidney": 16,
                "major_vein": 17,
                "kidney_with_Gerotas_fascia": 18,
                # These classes are ignored
                "unlabeled": settings.label_index_thresh,
                "overlap": settings.label_index_thresh + 1,
                "organic_artifact": settings.label_index_thresh + 2,
                "ignore": settings.label_index_thresh + 3,
                "unsure": settings.label_index_thresh + 4,
                "colon_peritoneum": settings.label_index_thresh + 5,
                # Ignored because they occur only the masks dataset
                "aorta": settings.label_index_thresh + 6,
                "bile_fluid": settings.label_index_thresh + 7,
                "bone": settings.label_index_thresh + 8,
                "cartilage": settings.label_index_thresh + 9,
                "lymph_nodes": settings.label_index_thresh + 10,
                "silicone_gloves_blue": settings.label_index_thresh + 11,
                "silicone_gloves_light_blue": settings.label_index_thresh + 12,
                "silicone_gloves_white": settings.label_index_thresh + 13,
                "veins": settings.label_index_thresh + 14,
                "diaphragm": settings.label_index_thresh + 15,
                "arteries": settings.label_index_thresh + 16,
                "ovary": settings.label_index_thresh + 17,
                "ureter": settings.label_index_thresh + 18,
                "blood": settings.label_index_thresh + 19,
                "lymph_fluid": settings.label_index_thresh + 20,
                "urine": settings.label_index_thresh + 21,
                # Ignored because they occur only the humans dataset
                "cauterization": settings.label_index_thresh + 22,
                "lig_teres_hep": settings.label_index_thresh + 23,
                "fat_subcutaneous": settings.label_index_thresh + 24,
                "fat_visceral": settings.label_index_thresh + 25,
                "meso": settings.label_index_thresh + 26,
                "esophagus": settings.label_index_thresh + 27,
            }
        )
        self.labels = self.label_mapping.label_names()

        self.model_comparison_timestamp = os.getenv("HTC_MODEL_COMPARISON_TIMESTAMP", "2022-02-03_22-58-44")
        self.dataset_size_timestamp = "2022-02-16_23-01-31"
        self.lr_experiment_timestamp = "2022-02-04_22-05-49"
        self.seed_experiment_timestamp = "2022-02-15_20-00-11"
        self.nsd_aggregation = "surface_dice_metric_image_mean"
        self.lr_default = 0.001
        self.lr_higher = 0.01
        self.lr_lower = 0.0001

        # Also used for ordering of the labels
        self.label_colors_paper = {
            "invalid": "#BBBBBB",
            "background": "#EEEEEE",
            "heart": "#1f77b4",
            "lung": "#aec7e8",
            "stomach": "#98df8a",
            "small_bowel": "#2ca02c",
            "colon": "#ffbb78",
            "liver": "#ff7f0e",
            "gallbladder": "#ff9896",
            "pancreas": "#f7b6d2",
            "spleen": "#d62728",
            "kidney": "#dbdb8d",
            "kidney_with_Gerotas_fascia": "#9edae5",
            "bladder": "#9467bd",
            "fat": "#e377c2",
            "skin": "#c49c94",
            "muscle": "#bcbd22",
            "omentum": "#c5b0d5",
            "peritoneum": "#8c564b",
            "major_vein": "#17becf",
        }
        self.labels_paper_renaming = {
            "invalid": "ignore",
            "major_vein": "major vein",
            "kidney_with_Gerotas_fascia": "kidney with<br>Gerota's fascia",
            "fat": "subcutaneous fat",
            "small_bowel": "small intestine",
        }
        self.modality_names = {
            "hsi": "HSI",
            "param": "TPI",  # Tissue Parameter Images (or Tivita Parameter Images xD)
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
        self.model_names = ["pixel", "superpixel_classification", "patch", "image"]
        self.n_algorithms = len(self.model_colors) * len(self.modality_names)

        self._results_dir = None

    @property
    def results_dir(self) -> MultiPath:
        if self._results_dir is None:
            if _path_env := os.getenv("PATH_HTC_RESULTS_SEMANTIC", False):
                self._results_dir = unify_path(_path_env)
            else:
                # If no path is set, we just use the default results directory
                self._results_dir = settings.results_dir
                settings.log.info(
                    "The environment variable PATH_HTC_RESULTS_SEMANTIC is not set. Files for the semantic"
                    f" segmentation project will be written to {self._results_dir.find_best_location()}"
                )

        return self._results_dir

    @property
    def paper_dir(self) -> MultiPath:
        target_dir = self.results_dir / "paper"
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    @property
    def nsd_tolerances_path(self) -> MultiPath:
        return self.results_dir / "rater_variability/nsd_tolerances.csv"


settings_seg = SettingsSeg()
