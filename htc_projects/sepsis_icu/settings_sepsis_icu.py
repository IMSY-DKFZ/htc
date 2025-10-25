# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import os

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.LabelMapping import LabelMapping
from htc.utils.MultiPath import MultiPath
from htc.utils.unify_path import unify_path
from htc_projects.sepsis_icu.tables import first_inclusion


class SettingsSepsisICU:
    def __init__(self):
        """Settings for experiments related to the sepsis ICU confounder study."""
        self._results_dir = None

        self.disease_colors = {
            "sepsis": "#C75B50",
            "no_sepsis": "#A1C75A",
            "intermediate": "#FABE26",
            "post_sepsis_problems": "#FA9733",
            "unclear": "#AFAFAF",
        }

        self.survival_colors = {
            False: "#810F03",
            True: "#5D703A",
            "non-survivor": "#810F03",
            "survivor": "#5D703A",
        }

        self.shock_colors = {
            "no_shock": "#FA9733",
            "shock": "#9B4073",
        }

        self.model_colors = {
            "RGB": "#97249C",
            "TPI": "#C73186",
            "HSI": "#C7A729",
            "HSI, median": "#F57E34",
            "HSI + clinical data": "#70663D",
            "clinical data": "#2F969C",
            "Random Forest": "#2563F5",
            "SOFA": "#23585F",
            "VIS": "#4DC6D6",
            "PCT": "#23585F",
            "CRP": "#23585F",
            "SMS": "#4DC6D6",
            "CRT": "#4DC6D6",
            "SIRS": "#23585F",
            "qSOFA": "#4DC6D6",
            "APACHE_II": "#23585F",
            "NEWS": "#4DC6D6",
        }

        self.measurement_site_colors = {
            "palm + finger": "#9891FF",
            "palm": "#f89c74",
            "finger": "#6DE8AC",
        }

        self.task_mapping = {
            "sepsis": "sepsis_status",
            "survival": "survival_30_days_post_inclusion",
            "shock": "septic_shock",
            "septic_shock": "septic_shock",
        }
        self.status_mapping = {
            "no_sepsis": "no sepsis",
            "sepsis": "sepsis",
            False: "non survivor",
            True: "survivor",
        }

        self.sepsis_label_mapping = LabelMapping({
            "no_sepsis": 0,
            "sepsis": 1,
        })
        self.shock_label_mapping = LabelMapping({
            False: 0,
            True: 1,
        })
        self.survival_label_mapping = LabelMapping({
            False: 0,
            True: 1,
        })

        self.functional_parameter_mapping = {
            "sto2": "oxygen saturation",
            "nir": "perfusion index",
            "thi": "haemoglobin index",
            "twi": "water index",
        }

        self.exclusion_subjects = [70]  # subject 70 is excluded due to cerebral death

        self.metadata_selection_real_time = {
            "demographic": [
                "age",
                "sex",
                "weight",
                "type_weight_measurement",
            ],
            "diagnosis": [
                "renal_replacement_therapy",
                "ECMO",
                "impella",
                "liver_replacement_therapy",
            ],
            "vital": [
                "heart_frequency",
                "sinusrhythm",
                "MAP",
                "systolic_blood_pressure",
                "temperature",
                "SpO2",
            ],
            "BGA": [
                "pCO2",
                "PaO2",
                "SaO2",
                "Hb_BGA",
                "lactate",
                "pH",
                "type_BGA",
            ],
            "catecholamines": [
                "noradrenaline_dose_SOFA",
                "adrenaline_dose_SOFA",
                "vasopressor_dose_SOFA",
                "dobutamine_dose_SOFA",
                "milrinone_dose",
            ],
            "ventilation": [
                "ventilation_tubus_tracheostoma",  # ventilation at any other timepoint during the day or FiO2 larger than 40% (from COPRA)
                "ventilation_currently",  # ventilation at current timepoint
                "APRV",
                "FiO2",
                "PEEP",
                "PPEAK",
                "respiratory_frequency",
            ],
        }
        self.metadata_selection_comprehensive = self.metadata_selection_real_time | {
            "system_status": [
                "fluid_balance",
                "diuresis",
                "CVP",
            ],
            "comorbidities": [
                "arterial_hypertonia",
                "preexisting_cardiovascular_disease",
                "cardiac_insufficiency",
                "COPD",
                "other_pulmonary_disease",
                "nicotine_abuse",
                "organ_transplantation",
                "alcoholism",
                "chronic_kidney_failure",
                "acute_renal_failure",
                "diabetes_mellitus",
                "malignant_primary_disease",
                "PAOD",
            ],
            "medication": [
                "ACE_inhibitors",
                "ARBs",
                "beta_blockers",
                "antiplatelets",
                "anticoagulants",
                "corticoids",
                "immunosuppressive_drugs",
                "opioids",
                "number_other_drugs",
            ],
            "lab": [
                "creatinine",
                "GFR",
                "LDH",
                "bilirubin",
                "CRP",
                "leucocytes",
                "Hb_lab",
                "thrombosis",
                "hematocrit",
                "sodium",
                "potassium",
                "PCT",
            ],
            "microbiology": [
                "infection",
                "bacterial_infection",
                "gram_pos_bacterial_infection",
                "gram_neg_bacterial_infection",
                "viral_infection",
                "fungal_infection",
                "pathogen_1",
                "date_pathogen_1",
                "site_pathogen_1",
                "other_pathogens_1",
                "pathogen_2",
                "site_pathogen_2",
                "date_pathogen_2",
                "other_pathogens_2",
                "pathogen_3",
                "date_pathogen_3",
                "site_pathogen_3",
            ],
            "other": [
                "days_on_ICU",
            ],
            "tpi": [
                "median_sto2",
                "median_twi",
                "median_nir",
                "median_thi",
            ],
        }

        # Attribute to group mapping
        self.metadata_groups = {}
        for group, attributes in self.metadata_selection_comprehensive.items():
            for attribute in attributes:
                self.metadata_groups[attribute] = group

        self.metadata_scales = {
            "age": "ratio",
            "sex": "nominal",
            "weight": "ratio",
            "type_weight_measurement": "nominal",
            "type_skin": "nominal",
            "renal_replacement_therapy": "boolean",
            "ECMO": "boolean",
            "type_ECMO": "nominal",
            "impella": "boolean",
            "liver_replacement_therapy": "boolean",
            "heart_frequency": "ratio",
            "sinusrhythm": "boolean",
            "MAP": "ratio",
            "systolic_blood_pressure": "ratio",
            "temperature": "interval",
            "SpO2": "ratio",
            "pCO2": "ratio",
            "PaO2": "ratio",
            "Hb_BGA": "ratio",
            "lactate": "ratio",
            "pH": "ratio",
            "SaO2": "ratio",
            "type_BGA": "nominal",
            "noradrenaline_dose_SOFA": "ratio",
            "adrenaline_dose_SOFA": "ratio",
            "vasopressor_dose_SOFA": "ratio",
            "dobutamine_dose_SOFA": "ratio",
            "milrinone_dose": "ratio",
            "ventilation_tubus_tracheostoma": "boolean",
            "ventilation_currently": "boolean",
            "APRV": "boolean",
            "FiO2": "ratio",
            "PEEP": "ratio",
            "PPEAK": "ratio",
            "respiratory_frequency": "ratio",
            "straylight_blinds": "nominal",
            "straylight_ceiling": "nominal",
            "catecholamine_administration": "boolean",
            "Hb_lab": "ratio",
            "creatinine": "ratio",
            "GFR": "ratio",
            "LDH": "ratio",
            "bilirubin": "ratio",
            "CRP": "ratio",
            "leucocytes": "ratio",
            "thrombosis": "ratio",
            "hematocrit": "ratio",
            "sodium": "ratio",
            "potassium": "ratio",
            "PCT": "ratio",
            "GCS": "nominal",
            "RASS": "nominal",
            "mottling_score": "nominal",
            "recap_time": "nominal",
            "VIS": "ratio",
            "SOFA": "nominal",
            "SIRS": "boolean",
            "qSOFA": "boolean",
            "NEWS": "nominal",
            "APACHE_II": "nominal",
            "APACHE_II_COPRA": "nominal",
            "SOFA_COPRA": "nominal",
            "status_OP": "nominal",
            "severe_organ_insufficiency": "boolean",
            "immune_deficiency": "boolean",
            "focus": "nominal",
            "respiratory_focus": "boolean",
            "abdominal_focus": "boolean",
            "genitourinary_focus": "boolean",
            "skin_soft_tissue_focus": "boolean",
        }

        self.metadata_units = {
            "heart_frequency": "\\bpm",
            "MAP": "\\mmHg",
            "SpO2": "\\percent",
            "temperature": "\\degreeCelsius",
            "pCO2": "\\mmHg",
            "PaO2": "\\mmHg",
            "SaO2": "\\percent",
            "Hb_BGA": "\\g\\per\\deci\\litre",
            "lactate": "\\mg\\per\\deci\\litre",
            "weight": "\\kg",
            "noradrenaline_dose_SOFA": "\\ug\\per\\kg\\per\\minute",
            "adrenaline_dose_SOFA": "\\ug\\per\\kg\\per\\minute",
            "vasopressor_dose_SOFA": "\\Unit\\per\\kg\\per\\minute",
            "dobutamine_dose_SOFA": "\\ug\\per\\kg\\per\\minute",
            "FiO2": "\\percent",
            "respiratory_frequency": "\\minute\\tothe{-1}",
            "PEEP": "\\milli\\bar",
            "PPEAK": "\\milli\\bar",
            "Hb_lab": "\\g\\per\\deci\\litre",
            "creatinine": "\\mg\\per\\deci\\litre",
            "GFR": "\\ml\\per\\minute",
            "LDH": "\\Unit\\per\\litre",
            "bilirubin": "\\mg\\per\\deci\\litre",
            "CRP": "\\mg\\per\\litre",
            "leucocytes": "\\nano\\litre\\tothe{-1}",
            "thrombosis": "\\nano\\litre\\tothe{-1}",
            "PCT": "\\nano\\g\\per\\ml",
            "hematocrit": "\\percent",
            "sodium": "\\milli\\mole\\per\\liter",
            "potassium": "\\milli\\mole\\per\\liter",
        }

        self.metadata_paper_renaming = {
            "type_skin": "Fitzpatrick skin type",
            "type_weight_measurement": "type of weight measurement",
            "PaO2": "pO2",
            "SaO2": "sO2",
            "Hb_BGA": "Hb (BGA)",
            "noradrenaline_dose_SOFA": "noradrenaline dose",
            "adrenaline_dose_SOFA": "adrenaline dose",
            "vasopressor_dose_SOFA": "vasopressin dose",
            "dobutamine_dose_SOFA": "dobutamine dose",
            "ventilation_currently": "ventilation",
            "ventilation_tubus_tracheostoma": "invasive ventilation",
            "PPEAK": "P-peak",
            "Hb_lab": "Hb (lab)",
            "leucocytes": "leukocytes",
            "thrombosis": "platelets",
        }
        self.metadata_groups_renaming = {
            "demographic": "demographics",
            "vital": "vital signs",
            "BGA": "BGA measurements",
            "diagnosis": "organ replacement therapies",
            "ventilation": "ventilation parameters",
            "catecholamines": "dose of administered vasopressors and inotropes",
        }

        self.model_timestamp = os.getenv("HTC_MODEL_TIMESTAMP", "2025-03-07_13-00-00")

        self.clinical_scores = {
            "sepsis": {
                "1hr": ["NEWS", "recap_time", "mottling_score", "qSOFA"],
                "10hrs": ["CRP", "PCT", "SIRS", "SOFA"],
            },
            "survival": {
                "1hr": ["VIS"],
                "10hrs": ["SOFA", "APACHE_II"],
            },
        }

        self.page_width_inch = 7.25  # following the Science Advances guidelines
        self.font_size = 8
        self.font_size_small = 7
        self.font = "Myriad Pro"
        self.font_family = "sans-serif"

    @property
    def results_dir(self) -> MultiPath:
        if self._results_dir is None:
            if _path_env := os.getenv("PATH_HTC_RESULTS_SEPSIS_ICU", False):
                self._results_dir = unify_path(_path_env)
            else:
                # If no path is set, we just use the default results directory
                self._results_dir = settings.results_dir
                settings.log.info(
                    "The environment variable PATH_HTC_RESULTS_SEPSIS_ICU is not set. Files for the sepsis ICU project"
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
        target_dir = self.results_dir / "paper_figures"
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    @property
    def test_unclear_paths_palm(self) -> list[DataPath]:
        df = first_inclusion()
        df = df[df["sepsis_status"] != df["enforced_sepsis_status"]]
        assert df.subject_name.nunique() == 71
        df = df.query("label_name == 'palm'").reset_index(drop=True)
        assert sorted(df.enforced_sepsis_status.unique()) == [
            "no_sepsis",
            "sepsis",
        ], "Unclear sepsis status in enforced palm labels"
        paths = [DataPath.from_image_name(image_name) for image_name in df.image_name.unique()]

        return paths

    @property
    def test_unclear_paths_finger(self) -> list[DataPath]:
        df = first_inclusion()
        df = df[df["sepsis_status"] != df["enforced_sepsis_status"]]
        assert df.subject_name.nunique() == 71
        df = df.query("label_name == 'finger'").reset_index(drop=True)
        assert sorted(df.enforced_sepsis_status.unique()) == [
            "no_sepsis",
            "sepsis",
        ], "Unclear sepsis status in enforced finger labels"
        paths = [DataPath.from_image_name(image_name) for image_name in df.image_name.unique()]

        return paths


settings_sepsis_icu = SettingsSepsisICU()
