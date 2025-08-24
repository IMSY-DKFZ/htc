# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import os

from htc.settings import settings
from htc.utils.LabelMapping import LabelMapping
from htc.utils.MultiPath import MultiPath
from htc.utils.unify_path import unify_path


class SettingsSpecies:
    def __init__(self):
        self.label_mapping = LabelMapping({
            "background": 0,
            "metal": 0,
            "abdominal_linen": 0,
            "blue_cloth": 0,
            "blanket": 0,
            "white_compress": 0,
            "white_compress_dry": 0,
            "anorganic_artifact": 0,
            "silicone_gloves_light_blue": 0,
            "silicone_gloves_blue": 0,
            "silicone_gloves_white": 0,
            "glove": 0,
            "foil": 0,
            "syringe": 0,
            "stapler": 0,
            "magnets": 0,
            "sutures": 0,
            "stomach": 1,
            "small_bowel": 2,
            "meso": 2,
            "colon": 3,
            "liver": 4,
            "pancreas": 5,
            "kidney": 6,
            "spleen": 7,
            "omentum": 8,
            "lung": 9,
            "skin": 10,
            "peritoneum": 11,
        })
        self.label_mapping_organs = LabelMapping(self.label_mapping.mapping_name_index, zero_is_invalid=True)

        self.pig_aortic_labels = [
            "stomach",
            "small_bowel",
            "colon",
            "liver",
            "spleen",
            "gallbladder",
            "fat_visceral",
            "fat_subcutaneous",
            "muscle",
            "peritoneum",
            "skin",
        ]
        self.rat_aortic_labels = [
            "stomach",
            "small_bowel",
            "colon",
            "liver",
            "kidney",
            "spleen",
            "bladder",
            "omentum",
            "fat_visceral",
        ]
        self.pig_icg_labels = [
            "stomach",
            "small_bowel",
            "colon",
            "liver",
            "kidney",
            "spleen",
            "omentum",
            "skin",
            "pancreas",
            "peritoneum",
            "lung",
        ]
        self.rat_icg_labels = [
            "stomach",
            "small_bowel",
            "colon",
            "liver",
            "pancreas",
            "kidney",
            # "spleen",
            "bladder",
            "omentum",
            "peritoneum",
            "fat_visceral",
        ]

        self.malperfused_kidney_subjects = [
            "SPACE_000005",
            "SPACE_000026",
            "SPACE_000089",
            "SPACE_000130",
            "SPACE_000147",
            "SPACE_000148",
            "SPACE_000180",
            "SPACE_000195",
            "SPACE_000227",
            "SPACE_000231",
            "SPACE_000232",
        ]
        self.excluded_images = [
            "SPACE_000130#2021_02_22_20_54_24",  # The kidney is not fully implanted yet
        ]
        self.malperfused_labels = ["kidney"]
        self.malperfused_labels_extended = ["colon", "liver", "kidney", "spleen"]
        self.icg_labels = ["stomach", "small_bowel", "colon", "liver", "kidney", "omentum", "peritoneum", "pancreas"]

        self.n_nested_folds = 3
        self.species_malperfusion_projection = {
            "pig": "weights+bias_malperfusion_pig_kidney=P091,P095,P097,P098+aortic",
            "rat": "weights+bias_malperfusion_rat_subjects=R017,R019,R025,R029",
            "human": "weights+bias_malperfusion_human_subjects=all",
        }
        self.species_icg_projection = {
            "pig": "weights+bias_ICG_pig_subjects=P062,P072,P076,P113",
            "rat": "weights+bias_ICG_rat_subjects=R043,R048",
        }
        self.spec_names = {
            "pig": "pig_semantic-only_5folds_nested-*-2_mapping-12_seed-0.json",
            "rat": "rat_semantic-only_5folds_nested-*-2_mapping-12_seed-0.json",
            "human": "human_semantic-only_physiological-kidney_5folds_nested-*-2_mapping-12_seed-0.json",
        }
        self.species_colors = {
            "pig": "#44AA99",
            "rat": "#89CCED",
            "human": "#DDCB76",
        }
        self.species_colors_malperfused = {
            "pig": "#D8EFEC",
            "rat": "#DCF0FA",
            "human": "#F8F5E4",
        }
        self.xeno_learning_color = "#b86fdc"
        self.icg_color = "#9ACA3D"

        self.model_timestamp = os.getenv("HTC_MODEL_TIMESTAMP", "2025-06-07_20-24-33")

        self._results_dir = None

    @property
    def results_dir(self) -> MultiPath:
        if self._results_dir is None:
            if _path_env := os.getenv("PATH_HTC_RESULTS_SPECIES", False):
                self._results_dir = unify_path(_path_env)
            else:
                # If no path is set, we just use the default results directory
                self._results_dir = settings.results_dir
                settings.log.info(
                    "The environment variable PATH_HTC_RESULTS_SPECIES is not set. Files for the human project"
                    f" will be written to {self._results_dir.find_best_location()}"
                )

        return self._results_dir

    @property
    def paper_dir(self) -> MultiPath:
        target_dir = self.results_dir / "paper"
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir


settings_species = SettingsSpecies()
