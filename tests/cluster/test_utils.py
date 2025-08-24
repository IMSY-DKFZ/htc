# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from htc.cluster.utils import cluster_command


def test_cluster_command() -> None:
    cmd = cluster_command("my_command", memory="16G", excluded_hosts=["lsf22-gpu02", "lsf22-gpu07"])

    assert (
        cmd
        == """bsub -R "tensorcore" -R "select[hname!='lsf22-gpu02' && hname!='lsf22-gpu07']" -q gpu-lowprio-debian -gpu num=1:j_exclusive=yes:gmem=16G ./runner_htc.sh htc training my_command"""
    )
