# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pytest

from htc.model_processing.SinglePredictionTable import SinglePredictionTable
from htc.tivita.DataPath import DataPath


class TestSinglePredictionTable:
    @pytest.mark.serial
    def test_compute_table_paths(self) -> None:
        scores = []
        for test in [False, True]:
            table_predictor = SinglePredictionTable(
                model="image", run_folder="2023-02-08_14-48-02_organ_transplantation_0.8", test=test
            )

            path = DataPath.from_image_name("R002#2023_09_19_10_14_28#0202-00118@semantic#primary")
            df = table_predictor.compute_table_paths([path], ["DSC"])
            assert "dice_metric" in df.columns
            assert "dice_metric_image" in df.columns
            assert len(df) == 1
            assert df.iloc[0]["image_name"] == path.image_name()

            scores.append(df.iloc[0]["dice_metric_image"])

        assert len(scores) == 2
        assert scores[0] != pytest.approx(scores[1])
