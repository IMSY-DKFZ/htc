# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import os

from htc.settings import settings
from htc.utils.LabelMapping import LabelMapping
from htc.utils.MultiPath import MultiPath
from htc.utils.unify_path import unify_path


class SettingsRat:
    def __init__(self):
        self.label_mapping = LabelMapping(
            {
                "stomach": 0,
                "small_bowel": 1,
                "colon": 2,
                "liver": 3,
                "pancreas": 4,
                "kidney": 5,
                "spleen": 6,
                "bladder": 7,
                "omentum": 8,
                "lung": 9,
                "heart": 10,
                "cartilage": 11,
                "bone": 12,
                "skin": 13,
                "muscle": 14,
                "peritoneum": 15,
                "major_vein": 16,
                "kidney_with_Gerotas_fascia": 17,
            },
            unknown_invalid=True,
        )

        self.label_mapping_standardized = LabelMapping(
            {
                "stomach": 0,
                "small_bowel": 1,
                "colon": 2,
                "liver": 3,
                "pancreas": 4,
                "kidney": 5,
                "spleen": 6,
                "bladder": 7,
                "omentum": 8,
                "lung": 9,
                "pleura": 10,
                "trachea": 11,
                "heart": 12,
                "cartilage": 13,
                "bone": 14,
                "tendon": 15,
                "ligament_pat": 16,
                "skin": 17,
                "fur": 18,
                "muscle": 19,
                "fat_subcutaneous": 20,
                "peritoneum": 21,
                "aorta": 22,
                "major_vein": 23,
                "kidney_with_Gerotas_fascia": 24,
                "diaphragm": 25,
                "tube": 26,
                "ovary": 27,
                "fat_visceral": 28,
                "thymus": 29,
                "blood": 30,
            },
            unknown_invalid=True,
        )

        # Only for those subjects, we have standardized recordings for all organs (there are more subjects with standardized recordings but not for all organs)
        self.standardized_subjects = [
            "R002",
            "R003",
            "R014",
            "R015",
            "R016",
            "R017",
            "R018",
            "R019",
            "R020",
            "R021",
            "R022",
            "R023",
            "R024",
        ]

        self.best_run_standardized = "2024-02-23_14-31-38_median_31classes"

        self.labels_paper_renaming = {
            "small_bowel": "small bowel",
            "major_vein": "vena cava",
            "kidney_with_Gerotas_fascia": "kidney with\nGerota's fascia",
            "fat_visceral": "visceral fat",
            "ligament_pat": "ligament",
            "fat_subcutaneous": "subcutaneous tissue",
            "saliv_gland": "salivary gland",
            "vesic_gland": "vesicular gland",
        }

        self.label_colors = {
            "stomach": "#FF1202",
            "small_bowel": "#FF9001",
            "colon": "#FFDD00",
            "liver": "#7FFD03",
            "pancreas": "#02FFF2",
            "kidney": "#0475FF",
            "spleen": "#020197",
            "bladder": "#630605",
            "omentum": "#9900ED",
            "lung": "#ED00C9",
            "heart": "#FD8EEC",
            "cartilage": "#15E7C5",
            "bone": "#A35F01",
            "skin": "#A32121",
            "muscle": "#484848",
            "peritoneum": "#8C7FB8",
            "major_vein": "#BE14C4",
            "kidney_with_Gerotas_fascia": "#BEE7C5",
            "tube": "#BDE70A",
            "ovary": "#AB8600",
            "aorta": "#AB8600",
            "pleura": "#FFF893",
            "blood": "#830000",
            "fat_visceral": "#FFC494",
            "tendon": "#89BDFF",
            "ligament_pat": "#FFB46D",
            "thymus": "#D88CFC",
            "trachea": "#00E28E",
            "fur": "#FF7830",
            "fat_subcutaneous": "#E66E6E",
            "diaphragm": "#73AF00",
            "thyroid": "#B90C00",
            "saliv_gland": "#BC6A00",
            "vesic_gland": "#00469C",
            "teeth": "#448801",
            "urine": "#FFEF88",
        }

        self.colormap_straylight = {
            "no_straylight": "#688B51",
            "ceiling": "#4D6DA1",
            "OR-right": "#8B5958",
            "OR-situs": "#9E50A1",
            "OR-situs+ceiling": "#604961",
        }

        self._results_dir = None

        self.colormap_straylight = {
            "no_straylight": "#688B51",
            "ceiling": "#4D6DA1",
            "OR-right": "#8B5958",
            "OR-situs": "#9E50A1",
            "OR-situs+ceiling": "#604961",
        }

    @property
    def results_dir(self) -> MultiPath:
        if self._results_dir is None:
            if _path_env := os.getenv("PATH_HTC_RESULTS_RAT", False):
                self._results_dir = unify_path(_path_env)
            else:
                # If no path is set, we just use the default results directory
                self._results_dir = settings.results_dir
                settings.log.info(
                    "The environment variable PATH_HTC_RESULTS_RAT is not set. Files for the rat project"
                    f" will be written to {self._results_dir.find_best_location()}"
                )

        return self._results_dir

    @property
    def figures_dir(self) -> MultiPath:
        target_dir = self.results_dir / "figures"
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    @property
    def paper_dir(self) -> MultiPath:
        target_dir = self.results_dir / "paper"
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir


settings_rat = SettingsRat()
