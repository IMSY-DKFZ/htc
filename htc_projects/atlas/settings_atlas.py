# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import os

from htc.settings import settings
from htc.utils.LabelMapping import LabelMapping
from htc.utils.MultiPath import MultiPath
from htc.utils.unify_path import unify_path


class SettingsAtlas:
    def __init__(self):
        self.label_mapping = LabelMapping({
            "stomach": 0,
            "small_bowel": 1,
            "colon": 2,
            "liver": 3,
            "gallbladder": 4,
            "pancreas": 5,
            "kidney": 6,
            "lung": 7,
            "heart": 8,
            "cartilage": 9,
            "bile_fluid": 10,
            "kidney_with_Gerotas_fascia": 11,
            "major_vein": 12,
            "peritoneum": 13,
            "muscle": 14,
            "skin": 15,
            "bone": 16,
            "omentum": 17,
            "bladder": 18,
            "spleen": 19,
            "unlabeled": settings.label_index_thresh,
            "overlap": settings.label_index_thresh + 1,
            "organic_artifact": settings.label_index_thresh + 2,
            "ignore": settings.label_index_thresh + 3,
            "unsure": settings.label_index_thresh + 4,
            "colon_peritoneum": settings.label_index_thresh + 5,
            "aorta": settings.label_index_thresh + 6,
            "lymph_nodes": settings.label_index_thresh + 7,
            "silicone_gloves_blue": settings.label_index_thresh + 8,
            "silicone_gloves_light_blue": settings.label_index_thresh + 9,
            "silicone_gloves_white": settings.label_index_thresh + 10,
            "veins": settings.label_index_thresh + 11,
            "blue_cloth": settings.label_index_thresh + 12,
            "metal": settings.label_index_thresh + 13,
            "white_compress": settings.label_index_thresh + 14,
            "abdominal_linen": settings.label_index_thresh + 15,
            "diaphragm": settings.label_index_thresh + 16,
            "arteries": settings.label_index_thresh + 17,
            "ovary": settings.label_index_thresh + 18,
            "ureter": settings.label_index_thresh + 19,
            "blood": settings.label_index_thresh + 20,
            "lymph_fluid": settings.label_index_thresh + 21,
            "urine": settings.label_index_thresh + 22,
        })
        self.labels = self.label_mapping.label_names()
        self.figure_labels = [k.replace("_", " ") for k in self.labels]

        # Final dataset information
        self.valid_cameras = ["0102-00085_correct-1", "0202-00118_correct-1"]
        # Two images less than in the paper since we had to remove the following images:
        # P094#2021_04_30_09_16_58 because it is actually P094#2021_04_30_09_16_31 (was wrongly labelled)
        # P086#2021_04_15_19_54_39 because it is actually 2020_05_14_19_54_39 (was wrongly labelled)
        self.n_images = 9057
        self.n_subjects = 46
        self.paper_tag = "_labelling_paper_002"

        self.n_images_standardized = 5756
        self.n_subjects_standardized = 11
        self.paper_tag_standardized = "_labelling_standardized_final"

        self.best_run = "2021-07-31_03-46-04_generated_default_lr=0.0001,gamma=0.9,dropout=0.2,class_weight_method=softmin,oversampling=False,batch_size=20000"

        # Colors which are also used in the symbols
        self.label_colors = {
            "stomach": "#ff1100",
            "small_bowel": "#ff9100",
            "colon": "#ffdd00",
            "liver": "#80ff00",
            "gallbladder": "#04b50f",
            "pancreas": "#03fff2",
            "kidney": "#0374ff",
            "spleen": "#050099",
            "bladder": "#630505",
            "omentum": "#9a00ed",
            "lung": "#ed00c9",
            "pleura": "#FFF893",
            "trachea": "#00E28D",
            "thyroid": "#B90C00",
            "saliv_gland": "#BC6A00",
            "teeth": "#448801",
            "heart": "#ff8fee",
            "cartilage": "#16e7c5",
            "bone": "#A35F00",
            "tendon": "#89BDFF",
            "ligament_pat": "#FFB46D",
            "skin": "#a32121",
            "fur": "#FF7830",
            "muscle": "#4a4a4a",
            "fat_subcutaneous": "#E66E6E",
            "peritoneum": "#8c7fb8",
            "aorta": "#AB8600",
            "major_vein": "#be17c5",
            "kidney_with_Gerotas_fascia": "#bee7c5",
            "diaphragm": "#73AF00",
            "tube": "#BDE70A",
            "ovary": "#05B50D",
            "vesic_gland": "#00469C",
            "fat_visceral": "#FFC494",
            "thymus": "#D88CFC",
            "blood": "#830000",
            "bile_fluid": "#bee70e",
            "urine": "#FFEF88",
        }

        self.labels_paper_renaming = {
            "small_bowel": "small bowel",
            "major_vein": "major vein",
            "kidney_with_Gerotas_fascia": "kidney with Gerota's fascia",
            "bile_fluid": "bile fluid",
        }

        self._results_dir = None

    @property
    def results_dir(self) -> MultiPath:
        if self._results_dir is None:
            if _path_env := os.getenv("PATH_HTC_RESULTS_ATLAS", False):
                self._results_dir = unify_path(_path_env)
            else:
                # If no path is set, we just use the default results directory
                self._results_dir = settings.results_dir
                settings.log.info(
                    "The environment variable PATH_HTC_RESULTS_ATLAS is not set. Files for the tissue atlas project"
                    f" will be written to {self._results_dir.find_best_location()}"
                )

        return self._results_dir

    @property
    def paper_dir(self) -> MultiPath:
        target_dir = self.results_dir / "paper"
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir


settings_atlas = SettingsAtlas()
