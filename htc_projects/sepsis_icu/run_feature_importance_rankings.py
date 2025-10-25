# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json

import numpy as np

from htc.utils.AdvancedJSONEncoder import AdvancedJSONEncoder
from htc_projects.sepsis_icu.baseline_methods import feature_selection
from htc_projects.sepsis_icu.settings_sepsis_icu import settings_sepsis_icu
from htc_projects.sepsis_icu.utils import config_from_baseline_name

if __name__ == "__main__":
    meta_attributes_dict = {}
    meta_attributes_dict["restricted"] = {
        1: "vital+demographic+BGA",
        10: "vital+demographic+BGA+lab",
    }
    meta_attributes_dict["all"] = {
        1: "demographic+vital+BGA+diagnosis+ventilation+catecholamines",
        10: "demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab",
    }

    for restricted in [False, True]:
        if restricted:
            meta_attributes_subdict = meta_attributes_dict["restricted"]
            name_str = "_restricted"
        else:
            meta_attributes_subdict = meta_attributes_dict["all"]
            name_str = ""

        ranking_dict = {}
        for target in ["sepsis", "survival"]:
            ranking_dict[target] = {}
            for timedelta, meta_attributes in meta_attributes_subdict.items():
                run_dir = f"random_forest_meta@{meta_attributes}"
                config = config_from_baseline_name(run_dir, target)
                overall_ranking = []
                for i in np.arange(5):
                    config["input/data_spec"] = config["input/data_spec"].replace("test-0.25", f"nested-{i}")
                    res = feature_selection(config)
                    overall_ranking.append(res["overall_ranking"])

                assert len(overall_ranking) == 5, "Incorrect number of folds"
                for j in np.arange(1, 5):
                    assert overall_ranking[0] == overall_ranking[j], (
                        f"Mismatch in feature rankings across nested folds for {target} at {timedelta} hrs (nested"
                        f" folds 0 and {j})"
                    )
                overall_ranking = overall_ranking[0]
                assert len(set(overall_ranking)) == len(overall_ranking), "Duplicate features in ranking"
                individual_attributes = meta_attributes.split("+")
                gt_len_ranking = 0
                for attribute in individual_attributes:
                    gt_len_ranking += len(settings_sepsis_icu.metadata_selection_comprehensive[attribute])
                assert len(overall_ranking) == gt_len_ranking, "Incorrect number of features in ranking"

                ranking_dict[target][timedelta] = overall_ranking

        savepath = settings_sepsis_icu.results_dir / f"feature_importance_rankings{name_str}.json"
        with savepath.open("w") as f:
            json.dump(ranking_dict, f, cls=AdvancedJSONEncoder, indent=4)
