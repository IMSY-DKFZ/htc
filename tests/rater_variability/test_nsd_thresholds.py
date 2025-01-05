# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest

from htc.rater_variability.run_nsd_thresholds import nsd_thresholds
from htc.settings_seg import settings_seg


@pytest.mark.skipif(not settings_seg.nsd_tolerances_path.exists(), reason="Precomputed NSD values are not available")
def test_nsd_thresholds() -> None:
    df_computed = nsd_thresholds()
    df_file = pd.read_csv(settings_seg.nsd_tolerances_path)

    assert df_computed["label_name"].tolist() == df_file["label_name"].tolist()
    assert pytest.approx(df_computed.reset_index(drop=True).drop(columns=["label_name"])) == df_file.drop(
        columns=["label_name"]
    ), "The precomputed NSD tolerances are not identical to the newly computed ones"
