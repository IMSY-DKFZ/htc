# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import hashlib
import inspect
import itertools
import re
from pathlib import Path
from types import MappingProxyType
from typing import Self
from zipfile import ZipFile

import pandas as pd
import torch
import torch.nn as nn

from htc.models.common.MetricAggregation import MetricAggregation
from htc.settings import settings
from htc.utils.Config import Config
from htc.utils.general import sha256_file
from htc.utils.helper_functions import run_info


class PostInitCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post__init__()
        return obj


class HTCModel(nn.Module, metaclass=PostInitCaller):
    # The models will appear in reverse order in the public README

    # Models from our MIA2022 paper
    known_models = MappingProxyType({
        "pixel@2022-02-03_22-58-44_generated_default_rgb_model_comparison": {
            "sha256": "93f359368e06e238d76351e0f9ad8818f15b7e75253051d0b48feeb153ead02f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/pixel@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip",
        },
        "pixel@2022-02-03_22-58-44_generated_default_parameters_model_comparison": {
            "sha256": "3d0360f28b4e3cb1c9120521cb88742ac8452ca3184bce5d95ff1b4b9dc8e489",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/pixel@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip",
        },
        "pixel@2022-02-03_22-58-44_generated_default_model_comparison": {
            "sha256": "f8d996d352a70edf6758903ecf678566154833b0e6c5bebbdfd6653a4abe17c4",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/pixel@2022-02-03_22-58-44_generated_default_model_comparison.zip",
        },
        "superpixel_classification@2022-02-03_22-58-44_generated_default_rgb_model_comparison": {
            "sha256": "56396eddd460524702b940c4aabfe470fbacbb6885a89b71441b8002bcb7a696",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/superpixel_classification@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip",
        },
        "superpixel_classification@2022-02-03_22-58-44_generated_default_parameters_model_comparison": {
            "sha256": "87a292b909a6f27a55965f2be42cc183332d91a634e23398930d3930ee3152f6",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/superpixel_classification@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip",
        },
        "superpixel_classification@2022-02-03_22-58-44_generated_default_model_comparison": {
            "sha256": "a27a0ed8a37b20663fc1fbf2ffe9e83a84fca0c5afb8c813974377b318f6e6ad",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/superpixel_classification@2022-02-03_22-58-44_generated_default_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_rgb_model_comparison": {
            "sha256": "d67353936110f4ace2ad73f704267d7f6cec9d1fff8df64029f9517fb28cca11",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_parameters_model_comparison": {
            "sha256": "e66328c9d3166a6fdded6bdc832cd882cacbc7faf98058b2b777831d420eb2b7",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_model_comparison": {
            "sha256": "b0a8f23a52f80b1277ec086b9938463f95c6651948eb8913dd1e7633a4bdc46e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_64_rgb_model_comparison": {
            "sha256": "e1d5c78a9d584b71a6353b618d435e1009eea2144590b2ba7045e55517ce16dd",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_64_rgb_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_64_parameters_model_comparison": {
            "sha256": "b6b086f9ef7014b6529d64be29b817c39e9a4db211de712db10f97220c034175",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_64_parameters_model_comparison.zip",
        },
        "patch@2022-02-03_22-58-44_generated_default_64_model_comparison": {
            "sha256": "ea4e0708e7b053e29797b0e607485de74499b7bcb144a68de3937fc8ca95b5c3",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_64_model_comparison.zip",
        },
        "image@2022-02-03_22-58-44_generated_default_rgb_model_comparison": {
            "sha256": "ee589c35cd9aa9fb201cc12e74a9133f1eb822b220a72b3252624f1c4402b636",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip",
        },
        "image@2022-02-03_22-58-44_generated_default_parameters_model_comparison": {
            "sha256": "d21c983dadd13aafab78e3c7dbad5ecfcaa362d5aa2a61833847c12c434eec65",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip",
        },
        "image@2022-02-03_22-58-44_generated_default_model_comparison": {
            "sha256": "c913639483efbf5536e46aea914099f15ae6a80ad21e5dd101c59ee55ee5c658",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2022-02-03_22-58-44_generated_default_model_comparison.zip",
        },
        # Models from our MICCAI2023 paper
        "image@2023-01-29_11-31-04_organ_transplantation_0.8_rgb": {
            "sha256": "9e322a8f13e3331a4d86b96eff63b2879b7d459c35b3894388b201d227a72986",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2023-01-29_11-31-04_organ_transplantation_0.8_rgb.zip",
        },
        "image@2023-02-08_14-48-02_organ_transplantation_0.8": {
            "sha256": "503258ade835f742759bafa63c50fa0925074189b762bfb391b4a2fecd4ff433",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2023-02-08_14-48-02_organ_transplantation_0.8.zip",
        },
        # Models from our NatureBME2025 paper
        "image@2025-03-09_19-38-10_baseline_human_nested-0-2": {
            "sha256": "f043372659f2e7db0913755dca713ff727e9623f672b50a805fc922e18377c98",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_baseline_human_nested-0-2.zip",
        },
        "image@2025-03-09_19-38-10_baseline_human_nested-1-2": {
            "sha256": "e8b09fbf176470dc0b112bda75c00c5ddf3bbaaf517523b855dc292f7715cc5e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_baseline_human_nested-1-2.zip",
        },
        "image@2025-03-09_19-38-10_baseline_human_nested-2-2": {
            "sha256": "e2a5dab3e895a99152d2fafa8658f48bfaeda811119f9454519bbdcd0264f38a",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_baseline_human_nested-2-2.zip",
        },
        "image@2025-03-09_19-38-10_baseline_pig_nested-0-2": {
            "sha256": "141701f7f0683402658d5bce7c5e32089d85849120f06a19ccd1f677d250900b",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_baseline_pig_nested-0-2.zip",
        },
        "image@2025-03-09_19-38-10_baseline_pig_nested-1-2": {
            "sha256": "d543466e998cccbe20be5c32b4b8756962ade7bf9c86e50aecd36948855cafaa",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_baseline_pig_nested-1-2.zip",
        },
        "image@2025-03-09_19-38-10_baseline_pig_nested-2-2": {
            "sha256": "6990b0f79a02ee456033e711b3d48c75862e0986582b9ba6458b9acba272d764",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_baseline_pig_nested-2-2.zip",
        },
        "image@2025-03-09_19-38-10_baseline_rat_nested-0-2": {
            "sha256": "314ca5b1c94a5b0bec6027f5f4fc14d906660855b80f4a2c571fa73b75b3f8ec",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_baseline_rat_nested-0-2.zip",
        },
        "image@2025-03-09_19-38-10_baseline_rat_nested-1-2": {
            "sha256": "8c3e87df7b30d78035a03f998313f2ec79d2dd93f3e293a9daa4f62ea4191579",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_baseline_rat_nested-1-2.zip",
        },
        "image@2025-03-09_19-38-10_baseline_rat_nested-2-2": {
            "sha256": "67750bf146c3013e71ad02ac5d4e6b5a0b760e44f88fb4ca439f6a3c03494427",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_baseline_rat_nested-2-2.zip",
        },
        "image@2025-03-09_19-38-10_joint_pig-p+rat-p2human_nested-0-2": {
            "sha256": "0f59d78910280aede87db26548079b65a89478a9aafb8902cc44fc3897bb12a1",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_joint_pig-p+rat-p2human_nested-0-2.zip",
        },
        "image@2025-03-09_19-38-10_joint_pig-p+rat-p2human_nested-1-2": {
            "sha256": "103da7443bf94c63abc4d1054a9c24a6eefad0b3c21c71f0acfa0dc903d96ed2",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_joint_pig-p+rat-p2human_nested-1-2.zip",
        },
        "image@2025-03-09_19-38-10_joint_pig-p+rat-p2human_nested-2-2": {
            "sha256": "907736eb8d8a3b823bb06562eab86e339969ad00cf8be2e128d60d855ccc6b1f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_joint_pig-p+rat-p2human_nested-2-2.zip",
        },
        "image@2025-03-09_19-38-10_projected-ICG_pig2rat_nested-0-2": {
            "sha256": "25ea884d0f81ed8613d47af924c34d8fd2af44da61c4d9622caf4fa661a190a1",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_projected-ICG_pig2rat_nested-0-2.zip",
        },
        "image@2025-03-09_19-38-10_projected-ICG_pig2rat_nested-1-2": {
            "sha256": "e49aefc6b7033cb5eed16334bd30490e789b7e33763bb2f1b4f494c2effa7353",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_projected-ICG_pig2rat_nested-1-2.zip",
        },
        "image@2025-03-09_19-38-10_projected-ICG_pig2rat_nested-2-2": {
            "sha256": "5af4e0f5deee17577cf0c93b2f2412ff0e4d4e002cbff5988a02258c03261ed3",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_projected-ICG_pig2rat_nested-2-2.zip",
        },
        "image@2025-03-09_19-38-10_projected-ICG_rat2pig_nested-0-2": {
            "sha256": "8e9abccf8f55b7a185c62e70a50099ddcebd7fe09566ce1e972f4e8bded82618",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_projected-ICG_rat2pig_nested-0-2.zip",
        },
        "image@2025-03-09_19-38-10_projected-ICG_rat2pig_nested-1-2": {
            "sha256": "2da2a39754d98842586a2256fabf590ea60595b1de78cd5792d8e7569b700087",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_projected-ICG_rat2pig_nested-1-2.zip",
        },
        "image@2025-03-09_19-38-10_projected-ICG_rat2pig_nested-2-2": {
            "sha256": "4804961a93e1ba7a5ca4c005741fff45338a12879da65695b2280b937e40c54e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_projected-ICG_rat2pig_nested-2-2.zip",
        },
        "image@2025-03-09_19-38-10_projected-malperfusion_pig2human_nested-0-2": {
            "sha256": "56d499de401422b745ce59d1a8accf7960bcfa2c4cce4310210afda97a2a0fbb",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_projected-malperfusion_pig2human_nested-0-2.zip",
        },
        "image@2025-03-09_19-38-10_projected-malperfusion_pig2human_nested-1-2": {
            "sha256": "3eb089186cef7e2b6b23c5336936a5424104e0dee909a43a04d810741aaa018d",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_projected-malperfusion_pig2human_nested-1-2.zip",
        },
        "image@2025-03-09_19-38-10_projected-malperfusion_pig2human_nested-2-2": {
            "sha256": "68273ff347281fd4442807d1300d5d625c5d64490a4ed03e0a4ed02edac8ab00",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_projected-malperfusion_pig2human_nested-2-2.zip",
        },
        "image@2025-03-09_19-38-10_projected-malperfusion_pig2rat_nested-0-2": {
            "sha256": "679a3541a3dbc0e09d33703739fc5949da3a86df075032144dbd1665f47c2204",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_projected-malperfusion_pig2rat_nested-0-2.zip",
        },
        "image@2025-03-09_19-38-10_projected-malperfusion_pig2rat_nested-1-2": {
            "sha256": "f16fda87d95e45e2ac435f367f1fff3de119a617557cc4b82fa9496af1da5c23",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_projected-malperfusion_pig2rat_nested-1-2.zip",
        },
        "image@2025-03-09_19-38-10_projected-malperfusion_pig2rat_nested-2-2": {
            "sha256": "f94a4a9a966cbfebe0b9f81a12e238a6c72782abc3720ed9f2ebb0286f7b516a",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_projected-malperfusion_pig2rat_nested-2-2.zip",
        },
        "image@2025-03-09_19-38-10_projected-malperfusion_rat2human_nested-0-2": {
            "sha256": "cdae0e1fb0c86acc4ab8cd3c18311368fed00e64245e5a31611aec65024eac4a",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_projected-malperfusion_rat2human_nested-0-2.zip",
        },
        "image@2025-03-09_19-38-10_projected-malperfusion_rat2human_nested-1-2": {
            "sha256": "b5f041dece5a76457c8a79f3f286b9e38a3137576c881d63e5b45fe922a7513a",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_projected-malperfusion_rat2human_nested-1-2.zip",
        },
        "image@2025-03-09_19-38-10_projected-malperfusion_rat2human_nested-2-2": {
            "sha256": "ea1c2e770cf9856ad78db8def6bdddb168dc4de5a18763e4ab078174312984e0",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_projected-malperfusion_rat2human_nested-2-2.zip",
        },
        "image@2025-03-09_19-38-10_projected-malperfusion_rat2pig_nested-0-2": {
            "sha256": "bf787201c9c4bf3ff9e037d6f4e4ba679a785e362a38ae1d4fc00b2b33de32ec",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_projected-malperfusion_rat2pig_nested-0-2.zip",
        },
        "image@2025-03-09_19-38-10_projected-malperfusion_rat2pig_nested-1-2": {
            "sha256": "a836156030be7feff3c630d1700d992c373a24388b7e807107035cb9308734de",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_projected-malperfusion_rat2pig_nested-1-2.zip",
        },
        "image@2025-03-09_19-38-10_projected-malperfusion_rat2pig_nested-2-2": {
            "sha256": "909c37ea0d8c92cb7672f5e02f03ddda9a2b457b5e01d61d37828ac772b5133a",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-09_19-38-10_projected-malperfusion_rat2pig_nested-2-2.zip",
        },
        # Models from our Science Advances 2025 paper (sepsis):
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-0-4_seed-0-2": {
            "sha256": "db3287125bf1358879b17e94eba30cd459d0cd5d6e97b61bf2f7d9859a47885b",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-0-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-0-4_seed-1-2": {
            "sha256": "077fa7fa2301190580db12737afdb366538a28a0ead4d9b7cdfc6d58fd3728e9",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-0-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-0-4_seed-2-2": {
            "sha256": "cfab5050c1867b23517630cfc0262582d272e99832660d6185f444399cfca09b",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-0-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-1-4_seed-0-2": {
            "sha256": "564128b42f1056c3640a0ab7fe76ad3fa344e8d71f20ddc0b1427c7852e627f3",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-1-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-1-4_seed-1-2": {
            "sha256": "bbf25a405a927faa87a69f31ffa61d79e6edc367a853e90aceaeac8af50c89b6",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-1-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-1-4_seed-2-2": {
            "sha256": "82bc5a4f9ff8bbcc13b7f96662abb08048d203b834aeadbc270e7ed5f3a99402",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-1-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-2-4_seed-0-2": {
            "sha256": "01f38fbb577170ee52a99b7f284dce13491530fcac4e75f1da433abf57a8a444",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-2-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-2-4_seed-1-2": {
            "sha256": "dfd67941f1b9f69ebf28c593ed5d7c520e28f6ad7a9863e3c63aea274f95e9cb",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-2-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-2-4_seed-2-2": {
            "sha256": "aa2b9e90f9c7e249b01aa12cf0d91d2353880ac7886bbe702fba11dd7b16a31d",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-2-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-3-4_seed-0-2": {
            "sha256": "6dd13dea7703a53d75b71627ff1e42d27bca16f660f2fbb82a8eb1e318a2abf9",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-3-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-3-4_seed-1-2": {
            "sha256": "0f27b479d1d35c57ae02399095b24cabe9abb864f422e7ea8c13c5b3fba3171d",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-3-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-3-4_seed-2-2": {
            "sha256": "875793a8c52fb9ad64d22962374820d22fc4be7508e4eb54679c15d6110e3268",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-3-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-4-4_seed-0-2": {
            "sha256": "f0f9a708d35bec99c2d5aa513441aa3b538a80f3a4c47976bfb65859dc7c5699",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-4-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-4-4_seed-1-2": {
            "sha256": "e8cf58445d613a84693433219318200f64f1637bc2b3c0685b15943b82799182",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-4-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-4-4_seed-2-2": {
            "sha256": "9ccba4d746c4f9af7bf20c6b17460c03ae56ece4bdafd03c96fb317b2ff45d27",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_nested-4-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-0-4_seed-0-2": {
            "sha256": "abdb6dc553b48e4f84be010ddbf8efdde3f65e7ae58c3de9d54cf3bc9c60b9e5",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-0-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-0-4_seed-1-2": {
            "sha256": "18ca256d1a0324c56d0d7623889f6ebf5fe3d9624a36ee1a6998191b17371ffe",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-0-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-0-4_seed-2-2": {
            "sha256": "b6faa48e1dc0c2b9f17b34680d8154ec91c1d91f90163b306ae8b493ebfce09f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-0-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-1-4_seed-0-2": {
            "sha256": "e509d8deef4bd889869885ed4b07682e62da5245c5ad09e97e5e7596ba58b8ee",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-1-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-1-4_seed-1-2": {
            "sha256": "e4e9351030dc7b8d7604253baae44c2bd958dd59dc7fcbdfa84231a5f707fc5c",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-1-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-1-4_seed-2-2": {
            "sha256": "8733d80a43a3803106c36db2773b3e7da7bee47496795dc8a684b7b341191bfa",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-1-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-2-4_seed-0-2": {
            "sha256": "7fb81a9b8ca905e8a525587b7aef1dd1325707e81e4df91da8e81259929ea327",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-2-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-2-4_seed-1-2": {
            "sha256": "b633b6f03690cda04a8af1a8b1d06ab59ec5657ab4e63a9e5a680596b9715edf",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-2-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-2-4_seed-2-2": {
            "sha256": "bbeb3a94ecabc950af4a83cbd1b85ec20bd6224ea09c0aaf0b12111b447858ca",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-2-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-3-4_seed-0-2": {
            "sha256": "a52c1101f2c693244d4f274550210e82cf52ebbb9829c55331c08d70756e7ad8",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-3-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-3-4_seed-1-2": {
            "sha256": "07991d7e59385b14ac16bc4f9b78ddcecddfde750afc6cef3ef04b4df617a7d5",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-3-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-3-4_seed-2-2": {
            "sha256": "a8a6aeca0488d2e4e1b9f0a69d08d712d4a90a74af6bb6b296b38c237b827f78",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-3-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-4-4_seed-0-2": {
            "sha256": "737554ddae1aa63cd177281fff4d15461bc7ef7ae1ebb079c997949677c1034a",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-4-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-4-4_seed-1-2": {
            "sha256": "849c37b15af600e9fd6e5d003929a4de1957965a759e725efeb7f8513becd9fa",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-4-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-4-4_seed-2-2": {
            "sha256": "7c169656dcd93c68d65a2d4e274f6c1fc05ec1d52ee28226882471ccdd51ce32",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_rgb_nested-4-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-0-4_seed-0-2": {
            "sha256": "0926182f159afe29647e9162ffdb93c6b2c301a9e9877e74a3ecf85a40fded31",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-0-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-0-4_seed-1-2": {
            "sha256": "98dd25f37035ab415aaa732372fa7e71ce7414e0720d21ce86f4daf43b43ffe7",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-0-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-0-4_seed-2-2": {
            "sha256": "73b8309a02dedbf57d20844f786718c12634d3fa8cc2d86874f683c461803a90",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-0-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-1-4_seed-0-2": {
            "sha256": "280fb67204d5936f16a8d3842bb86e21b009a49e9a975ce2e028151a2ebc77ad",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-1-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-1-4_seed-1-2": {
            "sha256": "1ce80169e4d7390e0e4e0200eddfa2407d10167fcbe887d459992c6378c72c90",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-1-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-1-4_seed-2-2": {
            "sha256": "5809d6b668c0cd95306257ca8e9f84615ab26a3c98ab970c9dcde77496e39765",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-1-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-2-4_seed-0-2": {
            "sha256": "eaa3a5315f337f1b0cd905eda5af9f9842d1a7b325d5641bef8fa7476b1cd78d",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-2-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-2-4_seed-1-2": {
            "sha256": "0f285c4d59529398b1d1f7261ec7475d9cbf69ad02a1b87f85b7b65d6f315f9b",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-2-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-2-4_seed-2-2": {
            "sha256": "29c4185571b2f3c7c7714745ac9266a4653a3d22b0eecc4043302da9750b473b",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-2-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-3-4_seed-0-2": {
            "sha256": "d77744ca144518d9750b5cc5774d81ba7c192090039c7f8b8be0df1ce5fb8b95",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-3-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-3-4_seed-1-2": {
            "sha256": "fa513886ffa682d96425ad6b53dfae271afba300a4a4a10152b01830a44ead4f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-3-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-3-4_seed-2-2": {
            "sha256": "34da89aa6ca76374f4d1d4fa4bb0b6f536f1346f0a05160574e944920731def4",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-3-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-4-4_seed-0-2": {
            "sha256": "55b5abcbfcb9d91f98ba7fa76000d4f9259dde273a8941cf2d7c64e4894c9d32",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-4-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-4-4_seed-1-2": {
            "sha256": "b79ebfdf92d443e4af0092fa4e8cb664e79ca5c805633fca602cd78b9d906c45",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-4-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-4-4_seed-2-2": {
            "sha256": "c41eadea8c5ded704d4eb4ac24358c9489f5afda863344d610a9fd85eb8d6c73",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_finger_image_tpi_nested-4-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-0-4_seed-0-2": {
            "sha256": "e5f130947abd39982291c4bb309dcd8dedccddb654a73ca0ae209e425d273a89",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-0-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-0-4_seed-1-2": {
            "sha256": "5ad652a46d037370d1f34416b8b7c697d19a9e34097e8b0435164a3bb7f88a29",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-0-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-0-4_seed-2-2": {
            "sha256": "23151927ae990a173600c7c28d7ed181efb51413b4eaa9ae913eeb6829a53cbc",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-0-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-1-4_seed-0-2": {
            "sha256": "059d5346473fab4c6e66225ba922db8d0bf872268eaaf3ae55e8826e5eb9d217",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-1-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-1-4_seed-1-2": {
            "sha256": "722a371ed27be824de7452d790cd1444cab61a4514ed0366efd831e7a5db45d0",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-1-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-1-4_seed-2-2": {
            "sha256": "d03c9c93e3333be2d42465251184917fd5c9eb6162f8ce044cd06812fe167843",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-1-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-2-4_seed-0-2": {
            "sha256": "067e3fe78e8bbdce93ac8512259110cc0c5c4a7dffbf54b7a09ccbdc37f54b55",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-2-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-2-4_seed-1-2": {
            "sha256": "2986e674dee4441ee72cb99624ed702f584d702b773e78118a84c47fe08fe497",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-2-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-2-4_seed-2-2": {
            "sha256": "82f8bd5a4a8c7590a6c68792891f7d85074e874c71da6d027cf79cd96ec3af15",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-2-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-3-4_seed-0-2": {
            "sha256": "523dbdcffa1bd21ec1a4c9baab9404fc444f5aca75b8d734d657ec88d184099d",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-3-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-3-4_seed-1-2": {
            "sha256": "952887ca8937f268aa70ed94c045c8aa4cdb9dfd34a8107180bfac500f18a77e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-3-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-3-4_seed-2-2": {
            "sha256": "605e0dd2c546e9248e44aab1adb153d67f2542128056a604fc13ec6a493cd106",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-3-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-4-4_seed-0-2": {
            "sha256": "36bbc36f1deeba6fb33f5f7382b6193574f05512219d56df71362125de5bec62",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-4-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-4-4_seed-1-2": {
            "sha256": "b25d5a824658384e3cd2527934731ece3930806350a34712f0aca3b2bb34ba18",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-4-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-4-4_seed-2-2": {
            "sha256": "a20b57fc305c7f4d51d4ff3dce960b5297e4846231c790c50d9aff13aaa7113a",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-4-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-0-4_seed-0-2": {
            "sha256": "276c5aa15b75d62ec13566e50186d69a4d50b1dba98d172d2e6b6656d805dfa0",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-0-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-0-4_seed-1-2": {
            "sha256": "bbd04683dd1d10973e360c795e22d3ca1fbb807bc33bc41741704b6dfdf87f82",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-0-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-0-4_seed-2-2": {
            "sha256": "ff367a93a1a25a759ef4a84b1a3c87a12ab1b77e14ad15d50e83e9af6a62643b",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-0-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-1-4_seed-0-2": {
            "sha256": "679ce594a8dd6dff1e5370bdd8521b864e9319203850a646a7fa81c38375049e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-1-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-1-4_seed-1-2": {
            "sha256": "9dff68ae38b23a7ae624f7dcb836809b5aea6670c451ff1e6b923ba272eafdea",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-1-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-1-4_seed-2-2": {
            "sha256": "75c31f5fdbce0eba20ec35feeeafe92b2e752779d43370d959a5069581525dfb",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-1-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-2-4_seed-0-2": {
            "sha256": "eecd3447d082c90ce366adb74e6faf3d4a46204880ed7b25cf18fede8e7ea01f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-2-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-2-4_seed-1-2": {
            "sha256": "b1733722cb66c161474a3fbef6467ab26bd12c86cd1fd77041d64885ee91fd3b",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-2-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-2-4_seed-2-2": {
            "sha256": "55b0cfd02006ce797d584974a13787d99265525fb6be49a260de55763e4dbe98",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-2-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-3-4_seed-0-2": {
            "sha256": "ef38c60b7e2341a00240395473cd55d3755eeae5724bf70f8fdaedc5ef00c247",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-3-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-3-4_seed-1-2": {
            "sha256": "9a223622f87d03976c12ea83d86ad25d968ca450d52fb15ce1446cb8bbb1a16c",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-3-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-3-4_seed-2-2": {
            "sha256": "2d1ad203662c8c2deb0fd78e82db2f490f28539d3f15526dddfb1de58d6e73b3",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-3-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-4-4_seed-0-2": {
            "sha256": "789e276373b46d316f1c02009ae8e78e3337e7b4578f640fa999e4901604ad93",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-4-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-4-4_seed-1-2": {
            "sha256": "eab711b744427af40260901aac575acaa2498b25c9ad36053dd75a8f8f78b62d",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-4-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-4-4_seed-2-2": {
            "sha256": "82a804143c22f6fbc81d1d5155b11951a0b2d225130b039725e0c56376957bab",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-4-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-0-4_seed-0-2": {
            "sha256": "f911f6413222a584e6e3dfd45deafda2f20784686f6c21f2f0aed5292e1b651f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-0-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-0-4_seed-1-2": {
            "sha256": "840da6946278118966cbeff509fb724d275ff4158accd5e69ed074e3fbee9caa",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-0-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-0-4_seed-2-2": {
            "sha256": "75082e8ba71453c66ac6d03da8ac767816c3d012b11f90d9c3bf74f6d2757eea",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-0-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-1-4_seed-0-2": {
            "sha256": "613eeb287952d59a589605eb16c0b90dd4a19302372c14262ffc7fdcea2dc895",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-1-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-1-4_seed-1-2": {
            "sha256": "f42cbab4ff5ebdbe2c091357e0a9a8444374a177d83cb040213979eecaefe101",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-1-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-1-4_seed-2-2": {
            "sha256": "8d8447c79144e5c69c813304a206429bb0512f4c26e13baf92585dd2e1c9e09f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-1-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-2-4_seed-0-2": {
            "sha256": "73a8fa9c8cd92d1854ebd39857076807558e89f6ed565b4708288ce2fe24ee99",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-2-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-2-4_seed-1-2": {
            "sha256": "2b23ebedcb17db98b644ec406d0c6db3ea1fcc0619f467e5997f768aa77f2e95",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-2-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-2-4_seed-2-2": {
            "sha256": "2e94be96134e024be1e4b36e933a7c04848ba3d03bd8c9e7e8f95e2d8f53921e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-2-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-3-4_seed-0-2": {
            "sha256": "5ab630260e1bc55a0d60c09f6f73e629c3281e19d9e3f06c79df82fcf741bbe3",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-3-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-3-4_seed-1-2": {
            "sha256": "a9f9f447e03fc5dd1e4ca0283f2ec39aad566a18a6a12aeb737b70cd35b9ce42",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-3-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-3-4_seed-2-2": {
            "sha256": "9807f29de0d87322125980e653d8b6f447e3c13be5faa32505b4e7bea6e044ab",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-3-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-4-4_seed-0-2": {
            "sha256": "1ddb1fa3675a5db5d442ce5feda4f26b180f589d3f21755fdc0e967bcd266117",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-4-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-4-4_seed-1-2": {
            "sha256": "d04507a9092d7ead49cdaa283a64a996e9d5fc75f996b93134d37b15b4e0672e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-4-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-4-4_seed-2-2": {
            "sha256": "0f13eed446b1f438ec9bf4a3cff6e41a9147a9c79d65584137724ceda3e27ccf",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_nested-4-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-0-4_seed-0-2": {
            "sha256": "f2948f326ab40a0163e2789d6c1fa8d3b944a48178bda2456b39ceb4ee535807",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-0-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-0-4_seed-1-2": {
            "sha256": "6217e536381069e74d479e348921be6423e67d2ee1b57083490e7c0142862b79",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-0-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-0-4_seed-2-2": {
            "sha256": "22a0295acb860b2f05aca07542576762ba5ce6d3cfa01a1603c8c3ea993206a2",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-0-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-1-4_seed-0-2": {
            "sha256": "29dfacb01b47b8b0d36f4cf65ddbeb12e0b647b4050ed2c8c9f9f4b614644996",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-1-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-1-4_seed-1-2": {
            "sha256": "92d4a5cda78baf758e9d95467c7b87448e98700fdedcdd23d956fc622419fec8",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-1-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-1-4_seed-2-2": {
            "sha256": "58bc555bdd29b0dc494a0a6552dda3c64d5acf0620676cbcacdb908421fea747",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-1-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-2-4_seed-0-2": {
            "sha256": "e4db580d4983a65390054ff98c5e60807c992b13d10286b2f64c0915854c795d",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-2-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-2-4_seed-1-2": {
            "sha256": "cc689940929c40a3ab15c98c37c9bc95a14dd63fac533eb3fb1424cf5c833ace",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-2-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-2-4_seed-2-2": {
            "sha256": "67672bcadb3e38b2cd25d0179cc92f9b873f28f14641b756a8f8ff5627729608",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-2-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-3-4_seed-0-2": {
            "sha256": "28435b6a4f2471e92c4b23ebb1439734bdc78f03731f98b18042c03f2e39f735",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-3-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-3-4_seed-1-2": {
            "sha256": "f3af5bcf05c6dd657ca68d74a827d0fa368805c986b8b11812b385969fd9da33",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-3-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-3-4_seed-2-2": {
            "sha256": "90024f1fd324c0d47b5f181dca89a44f0800677a153995c71d85d6ec32ab798e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-3-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-4-4_seed-0-2": {
            "sha256": "d7ffcfbddd7e329c37e347b10bdd09c3d43a2e18dfa153cd36f81e1de9ad3a66",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-4-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-4-4_seed-1-2": {
            "sha256": "965717cfc45a61ede7a1559f44070d71c54ff86171bf5f5c7ee1ad395529538e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-4-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-4-4_seed-2-2": {
            "sha256": "3efef110e500df0ea22cd7f47f1f4644a9be2e19219dfcdf87ee7c6d0d8cc60f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_rgb_nested-4-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-0-4_seed-0-2": {
            "sha256": "8f5721e022a79a2190cf8a8bb9436ea82ee5193602a42c58301b47011f0977d8",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-0-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-0-4_seed-1-2": {
            "sha256": "dacb4bcdee89ac723415945cd8d5322332f19d2e5c04a3fa19524db2f99622b4",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-0-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-0-4_seed-2-2": {
            "sha256": "7d4ba587b6819472ac233dbb7e093cd12f6d386ede97a8dbdc028488b8f8681a",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-0-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-1-4_seed-0-2": {
            "sha256": "ba2dc6323206e91b3ed15d89e21748922debfc1ac338033ec4df9cde012134a2",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-1-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-1-4_seed-1-2": {
            "sha256": "e74b756e4bda90e3fd2fb783de05e3f63f2c2f054da030cb8e1db14c65a635ce",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-1-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-1-4_seed-2-2": {
            "sha256": "338cf40b301df9b15e9c67f0ad35c617a8b505ce68a9191c23ea312f78cd1ff7",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-1-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-2-4_seed-0-2": {
            "sha256": "c0f3c9d0888db373524aceb8b9187ac532a8031779abb4d89db2687efe21dc55",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-2-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-2-4_seed-1-2": {
            "sha256": "8e92d7a09819824327e9d1c8a1be2458a7d9d46961b5b290aa6c603aa2b033ef",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-2-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-2-4_seed-2-2": {
            "sha256": "cdac3c8294a1afaa6e47ecd4c6f18ce2faf447be590f1e17afc6172bfa89e0c7",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-2-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-3-4_seed-0-2": {
            "sha256": "32a66cdd0cd77ea1c6700a76773f179263427888d0a15a24c9c1438bf9856c99",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-3-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-3-4_seed-1-2": {
            "sha256": "f9f0e6de9d48e716d11b551c51c82b5a32a3be18abaa4d9dbe483eceeb28f677",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-3-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-3-4_seed-2-2": {
            "sha256": "d2275fe8bb49841a790ca2d37e935852ef04efb37d57be31146151819bfdfc3b",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-3-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-4-4_seed-0-2": {
            "sha256": "6712b7bc4f8965fef46c10b9196c37ddc9e43638c2f732283829adaff7208847",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-4-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-4-4_seed-1-2": {
            "sha256": "3b7f671569b0d049a5ba04f318aa78c23cf3a7dd95c7117509de5a0105cd4e77",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-4-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-4-4_seed-2-2": {
            "sha256": "e301bcb3f6d84cd92d652002ae459a9b5526f4306ca6443e7b5bd449107ed559",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_image_tpi_nested-4-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-0-4_seed-0-2": {
            "sha256": "e7ec2ff4a99735be4ec6f23d9a008eb4e913321e0def9166496e3db3829468f3",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-0-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-0-4_seed-1-2": {
            "sha256": "5b4f143cc563f53e0d2b09f28549dc96b32a1eefb86d041341e645aa362596eb",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-0-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-0-4_seed-2-2": {
            "sha256": "37f7f133ae99a804f8b330dfeaaabbd341c5d1768bd4157f604829d621ef69d7",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-0-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-1-4_seed-0-2": {
            "sha256": "213780e6a7786b2c946ba4f7a7195dcbdfb85930e1322e9e07ce9892b34c2f5f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-1-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-1-4_seed-1-2": {
            "sha256": "3ee63a9cf1fbab388f6af7694bcb7bd7ac889c39e727fa6a20810f3f9706297f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-1-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-1-4_seed-2-2": {
            "sha256": "17203646aec58cb5531f4ccad0f54b12e57178d488a25fa345f6d3a4c3b8ea60",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-1-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-2-4_seed-0-2": {
            "sha256": "93f123346ab1926391c2ce3c5b667b3d45fee950bc2092c79e0fa40070038fc8",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-2-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-2-4_seed-1-2": {
            "sha256": "5e38e262f2e2e77317f33a8a16e0b11bcaf9620dc86252229a548208e371beb0",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-2-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-2-4_seed-2-2": {
            "sha256": "4fe5bad2cc71edf75e7916f77bb53864632364878febe098b42c5987f2d89dc9",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-2-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-3-4_seed-0-2": {
            "sha256": "ff6c958bc817fdacb026d5607e64b81f339085c722900ea1fa50bec0775ce401",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-3-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-3-4_seed-1-2": {
            "sha256": "e953d7a67b06787b07439f1688385e882324a797f3c3ca92ec727e6f973c9a37",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-3-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-3-4_seed-2-2": {
            "sha256": "98b8bfcad9b6332049dab347f67a7ea79097fb535577bd4b75ead437ea3765cf",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-3-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-4-4_seed-0-2": {
            "sha256": "2a76793a76f2be2ffc662f07ba6a8df90c4d05c44137d5e09eb755872040a25a",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-4-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-4-4_seed-1-2": {
            "sha256": "9afb4e2580dcaf4ee8e0a1cf7c163fea43409c9c06c6d92b87596ec06f25eb8f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-4-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-4-4_seed-2-2": {
            "sha256": "cac701ec75bb55dbf542171f8477aa6e89293a7eace7759c8183ce81727e542f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_sepsis-inclusion_palm_stacked_image_nested-4-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-0-4_seed-0-2": {
            "sha256": "1e46380d151b58a3c41e490c3ed6ede82190ab1b57df40c46cac0eeba04f6e00",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-0-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-0-4_seed-1-2": {
            "sha256": "237d769f130f13a495533fb702c7cb91c8ecf91fce13fc4b7228f188afbf73fa",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-0-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-0-4_seed-2-2": {
            "sha256": "95bae016a796efb35de6cf2673627220beb667a123a92d1339b1a2672bc0c045",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-0-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-1-4_seed-0-2": {
            "sha256": "7e55e8a53784049ce0e6e10ad19e7e1063a6c643e1ece7477df107ccf11f0dc6",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-1-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-1-4_seed-1-2": {
            "sha256": "de769a81c52433d08fe738c270376266f288ce9252904b92c02e67d434ff3ed1",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-1-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-1-4_seed-2-2": {
            "sha256": "9f12f5d4b1556f2df1d4809a6a7e7e600017950922372f4729144d0b074e53a7",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-1-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-2-4_seed-0-2": {
            "sha256": "72c3e714e5faa77cc2228edeffbd770f3b93bee4d166a999b6174d99c75433b9",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-2-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-2-4_seed-1-2": {
            "sha256": "c28ce42b766c4b06721f3b7af4783a47f89f8b6ce55be45df0770a73a5099534",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-2-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-2-4_seed-2-2": {
            "sha256": "e806c25ad32b74c527319dba085604354b5cc69b23e12e1c6311b2fc1d51d6e3",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-2-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-3-4_seed-0-2": {
            "sha256": "1c12e1fd2020586ff70bc4d1b878e320b1e208b7deb25e76e53d939bc160586d",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-3-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-3-4_seed-1-2": {
            "sha256": "7f1a6b4c2fd7d793d1c56b44ce4d40cc5f245e782cf7a1995b4374ccbe9983e1",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-3-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-3-4_seed-2-2": {
            "sha256": "bd449ed442dd65ddf8456aed401d4a7b0773e5a9886e3d642e0a4f431d377121",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-3-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-4-4_seed-0-2": {
            "sha256": "eb28e83682e3232e545bd0dd3b3680dc3faafb64f845b56d9e6415d0041e6ff7",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-4-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-4-4_seed-1-2": {
            "sha256": "02d1e6a5af57f0057c6cd282fc5bc8e51cecf5cb2d6de14528c1dca80b6df29b",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-4-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-4-4_seed-2-2": {
            "sha256": "e647fa9d3cc1123f1949ceb14ef1f890838d7352d5e9d9da1aeb63bf195378c9",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_nested-4-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-0-4_seed-0-2": {
            "sha256": "624b8af78c69fe2ab8b2fd54a917e5932d6fcde9fe67557c5fb196acbe26ccc4",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-0-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-0-4_seed-1-2": {
            "sha256": "0c4e7bb2a1d699a87b3e12282ee98725ee2d774d1b462f7f887378d7a26fdb1e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-0-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-0-4_seed-2-2": {
            "sha256": "bb16c2bbea90d36d0d194a35c8fecf88a36d70e1529890ecd9355ca8276a2950",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-0-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-1-4_seed-0-2": {
            "sha256": "126f670f3561f0e13957e0be30026344773590230276a027918c9beeb2ccfded",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-1-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-1-4_seed-1-2": {
            "sha256": "6caac64f3e9ba2a4152da88981edf362e628d2955d0905adbbccfa6e223ce139",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-1-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-1-4_seed-2-2": {
            "sha256": "74fc300b3c9eda0f3bcfd57ce4c0cb6b70fd33becdcc4c4ec1fdd952500b2aff",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-1-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-2-4_seed-0-2": {
            "sha256": "8652fc9666100e797e70aba03911d48c35b54e2303194f9115941ec9259f67ba",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-2-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-2-4_seed-1-2": {
            "sha256": "779146735bda211e25bf96a820a1db953d7726c289ae1044a1923524eb7f0bce",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-2-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-2-4_seed-2-2": {
            "sha256": "6f40d7c1e7f7012185ff452afac4ba2213847edfa67d81a4d28d83658e919edd",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-2-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-3-4_seed-0-2": {
            "sha256": "b467ea0300ca59abea5edb1ab5ccd522a367adbc053e5be9e0fdf75267c42b69",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-3-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-3-4_seed-1-2": {
            "sha256": "0882be38d06d3722cb9ccc6a687ddd45e788b0d0f5aed11d17b2cb0b913868b9",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-3-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-3-4_seed-2-2": {
            "sha256": "2e0bbaab7381f2c83a3bb27ee3aa34cf25bc6b980736cf139163938c8e10218e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-3-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-4-4_seed-0-2": {
            "sha256": "3e77e077c3ea277f702607d12d50796ba78ec10485578a9fdd1ec7d3f6b71d59",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-4-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-4-4_seed-1-2": {
            "sha256": "3889125d8d02796021ad56fe4f0714fdaaf52dc3b141dc5958279acb5ab38912",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-4-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-4-4_seed-2-2": {
            "sha256": "81cde6afb58ba91a5f94e2ade57305a16936c7605c078d164d1092d2b2767799",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_rgb_nested-4-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-0-4_seed-0-2": {
            "sha256": "f0584cd898889ca1d8569f8a6fae6161a89b772aa15153cb29d0a5b8862de8b0",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-0-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-0-4_seed-1-2": {
            "sha256": "fc87963f7e8311aa8ee0e58f26c75106b6714bc63b93d51d9d34ad646f789735",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-0-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-0-4_seed-2-2": {
            "sha256": "5dd69a3cd5e2f02dcb4f14f846cfb21c5d0cf59de1acc368f80cf1daa17e8fed",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-0-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-1-4_seed-0-2": {
            "sha256": "3f087750ded0823342f977691b03469743017bb8961b7c64c74ce958f8a6d2c7",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-1-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-1-4_seed-1-2": {
            "sha256": "47b26b64ebaab7c5496c82e9b3c2ea6b03e68ff64f3d3c79951db62c6db4fe5a",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-1-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-1-4_seed-2-2": {
            "sha256": "7520410351cf067d8b976d9611f8a266288b78d1e11871fbd949f3a801288c09",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-1-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-2-4_seed-0-2": {
            "sha256": "cb638b6a9b26074dd495a70678db08828b2d34a1daad4057841d033f68329ffa",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-2-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-2-4_seed-1-2": {
            "sha256": "a7bdde2a05ffe3575f496ac6a77875b4ce7d0821faac4d4fca568a4202fa1c55",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-2-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-2-4_seed-2-2": {
            "sha256": "faf3f62d69b556c6c2a444b4573dc339db9c9386d02a54ec9722e60e84fb72fd",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-2-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-3-4_seed-0-2": {
            "sha256": "651b1239721fcfdd5f6549ddd3fd6ea4d06ff6a245ce95ecd75713d1f537c6b0",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-3-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-3-4_seed-1-2": {
            "sha256": "901a8b9e4d3dd3c8488e9f897b361b9182cc8c0484ec45fc5bb1acabd0377de9",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-3-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-3-4_seed-2-2": {
            "sha256": "8cf4cc03b8485688c994648fc3800d16a8981067603e67b5ca6743e2a1f96473",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-3-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-4-4_seed-0-2": {
            "sha256": "211d9e1ca6c8b2b7771c615112f8f16d9f51b994bb0296e85e62c68c03a61a75",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-4-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-4-4_seed-1-2": {
            "sha256": "d03f23bb18dd152e33cf046a54dbac4d77b4ad422e77572ef84730c62a3c7023",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-4-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-4-4_seed-2-2": {
            "sha256": "048a5562c7dad5e1babae30c794ac6903cd00203b85c59761b5d22fedc9c1920",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_finger_image_tpi_nested-4-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-0-4_seed-0-2": {
            "sha256": "3b335b3a174536c74e9837bedec2a9732c005312b2fce5c0f6dce3e639a289e8",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-0-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-0-4_seed-1-2": {
            "sha256": "7ca6b6071b933df24d2940ded9d8d03eec4a89d5620237410ffa775f8b903b50",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-0-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-0-4_seed-2-2": {
            "sha256": "92104d69a0a97a8ce17b7f1f761de078923e480bcdb1b492b05efbe6ab116326",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-0-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-1-4_seed-0-2": {
            "sha256": "fd3cac1e6db229efcaafc85964bf6fdd5b0fa82ed5bbf6dda70de552714c3e8c",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-1-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-1-4_seed-1-2": {
            "sha256": "6a305cf402a2b34429e7bc3935579716b87dc2900dd79b4aea13cde2bc9b6870",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-1-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-1-4_seed-2-2": {
            "sha256": "3d136932ae422cb32fae0bc072914f93c6fa26156693fcd5c4631e1957f6a6ba",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-1-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-2-4_seed-0-2": {
            "sha256": "ba2fdd91f71677e74520caac7f3e18fcd94a1ef5a08b24a2ae419c9ad0605aff",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-2-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-2-4_seed-1-2": {
            "sha256": "9f0480aed55ff5913a3548ce4384994ccaa134e3541cc088523075153d941975",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-2-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-2-4_seed-2-2": {
            "sha256": "88fc3a1a993108584ef8eb11c88eb32b667b9156fe1003e59fec156d8315f12e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-2-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-3-4_seed-0-2": {
            "sha256": "6f2b8d3051ebf91de83851f1560f6c614924ad4e5946c85ef07cab5c9ef23a25",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-3-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-3-4_seed-1-2": {
            "sha256": "14d1ce812bbc3bf4d4e3b0fbb270c4ffb22ce1e658c276b83d80d2f658212286",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-3-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-3-4_seed-2-2": {
            "sha256": "b55ca10fa2c0f6b367ea83e57ceea71c1274e9191813c5071a3d28a31ff19a07",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-3-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-4-4_seed-0-2": {
            "sha256": "64e04f20f63763454f41cb8a1af0e01c2f331358b90030e54bc59e6d0e4e77a8",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-4-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-4-4_seed-1-2": {
            "sha256": "157e57cf46032df2b29b88cfafa80033025dac91d71873dc7edb4eb3bf1d06ee",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-4-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-4-4_seed-2-2": {
            "sha256": "a2b159aa5108f7c3963051a40d92d50049f1195d611ed771fcd0f705afdb7a25",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab_nested-4-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-0-4_seed-0-2": {
            "sha256": "61138282ddaa5491c7fe596317478e7860cedf3bad56e53c8e734c31e962a0c4",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-0-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-0-4_seed-1-2": {
            "sha256": "3a5fad6efa8e268fdb3e2f6fdcbfffc28f130ab3b82b20f7ad829776b2e4cab5",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-0-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-0-4_seed-2-2": {
            "sha256": "391f8c515e26a79553fd5a465a442badf52cff7ce8d1ca11021fc9a2410c344e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-0-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-1-4_seed-0-2": {
            "sha256": "8d17cf7a5f3ae6a5b768d37bed194fecf09bcfdea17b67bb29ace88a91aca3f7",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-1-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-1-4_seed-1-2": {
            "sha256": "0de1351b64a7506a168337c7b637248fc3e01b4d7c68a9cb83f7e01cb3787b4e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-1-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-1-4_seed-2-2": {
            "sha256": "76df6031e5005f86bada98a38ee26834aa50cb8d9ec8494f079ee3c823abda07",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-1-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-2-4_seed-0-2": {
            "sha256": "a5f604edf47cb29e491327021ccec7698eca8b109aefaa8f82cdde96f85549b2",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-2-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-2-4_seed-1-2": {
            "sha256": "48ba1245a7c521fa815e521a8ce2e1b2509b7a438339511e9777f0765b97be71",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-2-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-2-4_seed-2-2": {
            "sha256": "93a1e21c6cefa02e1e180328a71c974f5a57596cad30fe15fde701adc14cf60a",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-2-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-3-4_seed-0-2": {
            "sha256": "a0c9664e45563b43068e8f3a9b197d26cce1fb3d7e7c0fa565a8722d0ac922bb",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-3-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-3-4_seed-1-2": {
            "sha256": "c086ea55312b198bd2cb7a70950da82dbbde4020491b59df091c97325f1d7976",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-3-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-3-4_seed-2-2": {
            "sha256": "d4dd952e6dccfe29430b98cd4fdb60a19ee8a584543c85647a1dc67ba615dfec",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-3-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-4-4_seed-0-2": {
            "sha256": "0a2fc8cc6f6662f21dd0226185c86ccfe090ffc17c10a7917940ec7841df2877",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-4-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-4-4_seed-1-2": {
            "sha256": "07e708dad3de6e8fdb7a63613053f741c1368b15ebb2091be92545aa8f40ed1f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-4-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-4-4_seed-2-2": {
            "sha256": "91118f2ec1418374a3d9316068ed4e07a0cd240794c07bf8f48353b02fd86bd7",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image-meta_demographic+vital+BGA+diagnosis+ventilation+catecholamines_nested-4-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-0-4_seed-0-2": {
            "sha256": "0f8a94f37dc2e694137608db4cf7f37a95cb0e7309f715761e059182c662b0f4",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-0-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-0-4_seed-1-2": {
            "sha256": "2e9dd061d90cd2d0016175065d3a005334fa8942b91a45492e2299965d14bc6d",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-0-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-0-4_seed-2-2": {
            "sha256": "0a8001d50a51776dc832eeac0a488f9077c844c8305343a3ca813d8a6f9c76e0",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-0-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-1-4_seed-0-2": {
            "sha256": "6bb31aa5f9f9d99673e0ba6513e720d2abd3a5dfa979fecae2650704a608f293",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-1-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-1-4_seed-1-2": {
            "sha256": "71b42486a494df46accb4fe0651eee4bcc03da6c1bd600290d09ad7007f41d59",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-1-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-1-4_seed-2-2": {
            "sha256": "bb731fe5d3a45eafef13ce8d7e0a69d9e2680c8b6ecf750f0cdbb1f225fb6df2",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-1-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-2-4_seed-0-2": {
            "sha256": "041cd70e64d01a426ad67175f558d7d34c2af70eefaffbe6f49e80891eca4125",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-2-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-2-4_seed-1-2": {
            "sha256": "1a958b0c5e5b17af8e79ab98efd0e057002e20dcfd483d4f6f6a6c2aa966148c",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-2-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-2-4_seed-2-2": {
            "sha256": "6541a88fce96d43450598267f2b414b1d53464a7566ef2a58ce3e7a38dc7d687",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-2-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-3-4_seed-0-2": {
            "sha256": "a41c8cefdee4327a96d54bf7976392fdb7c6c3df225fcd4875c96932caee89df",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-3-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-3-4_seed-1-2": {
            "sha256": "301dd9a91ec0b79e9bd3d11f06b94ba2082e819a9933c0114813d787538ba2c0",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-3-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-3-4_seed-2-2": {
            "sha256": "b6cdd8059d1205acc9aadff5e95b9845ddb261329f1c8dce0b47c6b87a1665ca",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-3-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-4-4_seed-0-2": {
            "sha256": "9420fb46153887dd2af3a3062b97511fae33d7decd2d835f4c089daa4ed336cf",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-4-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-4-4_seed-1-2": {
            "sha256": "637865287175d32675cc3e9d4093d2dc621e1c672b3ddd7cc7f94b47d2ef6b29",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-4-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-4-4_seed-2-2": {
            "sha256": "0b9b36d9cbb680690e21100a2dae4529ec997a8eb58a277994d55b830c77cf34",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_nested-4-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-0-4_seed-0-2": {
            "sha256": "f230430e508a4362ac8f85a8237e28158a0ff78176e4b0cbc2542ad37ebfbf2e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-0-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-0-4_seed-1-2": {
            "sha256": "8cb76d35283e7734c4d98c40b76a77a28d6afd5a9bf1e230b1d67ad3532f85ba",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-0-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-0-4_seed-2-2": {
            "sha256": "ff3df4dea85d7bc5fb859b76e26a8fa1d5447a01cff1c6a5bcaea46affd75e54",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-0-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-1-4_seed-0-2": {
            "sha256": "0834dd208031d208dfa37493411a02a13f06525fd9e76c0c66cac64792118ea9",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-1-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-1-4_seed-1-2": {
            "sha256": "776748c4ee1f8a8db6d78b2c0502618b37ffda347d1b1cc68b9de3e6dcd54fea",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-1-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-1-4_seed-2-2": {
            "sha256": "8f82be5ee3ed7ffc92159ca6925d1e239a1f9760e4d1f1a1a9a3c99cb31a42ee",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-1-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-2-4_seed-0-2": {
            "sha256": "73db903af0525bf6bae984587da5f01eadf5210ff964eabd3d0323bc48a3ac99",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-2-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-2-4_seed-1-2": {
            "sha256": "a48cc577b497fd3509388d91e228a79b0796352a40597a0e44746a6c26f2c963",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-2-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-2-4_seed-2-2": {
            "sha256": "ca40634d952639b34c1b54dc5911f83fc34e828bec946000fb45b7b947343f4d",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-2-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-3-4_seed-0-2": {
            "sha256": "a535aae8c12021a898a60cac514a599e341b138632a629618a3d11856c97f2b1",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-3-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-3-4_seed-1-2": {
            "sha256": "9d8aa64c4ec910acebb04cdfb799aea272c13f3af7e84248aff1aa4ca88e3ba6",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-3-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-3-4_seed-2-2": {
            "sha256": "fe85a33b619b21939b9e1489888a13fd7d9158880ef297e928a9e00bace7707a",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-3-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-4-4_seed-0-2": {
            "sha256": "fc4dfd43405bfce2f15f0b6685ed3cce7d23b214e060257048815257ce1d1da9",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-4-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-4-4_seed-1-2": {
            "sha256": "19bf378fac36971774f68df35b484c245b59aa82b9956be89722100affec58bc",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-4-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-4-4_seed-2-2": {
            "sha256": "7955dc783acfcb84889d5925c452556383b4319781d13cc55958cbbc3b93ff4c",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_rgb_nested-4-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-0-4_seed-0-2": {
            "sha256": "3dc7c4476921af50134e07af1fbfbcc2481351a28b459a07c94a3bac60be5eb6",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-0-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-0-4_seed-1-2": {
            "sha256": "436f7bdab2939fa321068ea2ef6922c29b7e66bbb5fe8cb49284c5718cd82be5",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-0-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-0-4_seed-2-2": {
            "sha256": "c4c2f9fd95e1335e635466639a974e1254111b23bf27c26eea0d26932aaf9224",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-0-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-1-4_seed-0-2": {
            "sha256": "48afa8ebf5f9f0e48804006309f35134fbe95abb05e2a36deed7492b0d85a169",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-1-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-1-4_seed-1-2": {
            "sha256": "3038e032fb96fc6ac918f1a1f8c48c76eda67f0c1994dc31970e1333c55e751c",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-1-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-1-4_seed-2-2": {
            "sha256": "93ad17c007d8f859f6b9f60f5165210d80e340902049f87114aeae5fc4bb7fca",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-1-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-2-4_seed-0-2": {
            "sha256": "27505653717b407d36cc6bd810091b83950309fe7dd4f23a5e41388bd88f04b7",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-2-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-2-4_seed-1-2": {
            "sha256": "1795911f551aa4d094436d7662ff1a928af9ee4a0b238a430033e33e4eef8b67",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-2-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-2-4_seed-2-2": {
            "sha256": "e4c84d75bc71ae95b42feecfc0417b3be6aed3df89d9ff75b9ffbfefc10aab5b",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-2-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-3-4_seed-0-2": {
            "sha256": "c5ae84f8bea547c0dd12152a299b652a311f1c156d8c9534b2cd42d069d2a759",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-3-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-3-4_seed-1-2": {
            "sha256": "df2175864057fd961a45a0420fb00414f0a4cb2c0810be20b2d3614865b98df4",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-3-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-3-4_seed-2-2": {
            "sha256": "58f0a7fd141ae1dfb24a166365358c64f95c79dfbf1ac6417d10db2e1a9ed8bc",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-3-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-4-4_seed-0-2": {
            "sha256": "1d72aa5c416aefc4a18340263032d8aa3d21bcd55fcd265755b95ae6a33b37d1",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-4-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-4-4_seed-1-2": {
            "sha256": "40c9a76c85f8e01da8c382484f2c8edeaa38c493e4a92470dbb07a1293dccd9f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-4-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-4-4_seed-2-2": {
            "sha256": "b208fb0e656deb43ccf9603df68e5a4a73b41ac77fa278f6846c1dc30966f8d6",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_image_tpi_nested-4-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-0-4_seed-0-2": {
            "sha256": "49d6058321da435b704fdb7d5447d9e7d697b14ae5797e8486b053568ec4b46b",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-0-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-0-4_seed-1-2": {
            "sha256": "2c6f3f23cc623a288f4c4facb22222aaebceba6958d8a982146e093c09376456",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-0-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-0-4_seed-2-2": {
            "sha256": "23a91351cf1e40544717b57ba847b5c92e07cda5a4b1dedcea6320e098b97351",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-0-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-1-4_seed-0-2": {
            "sha256": "50f97d9cb28b469837e1ee1b75aef3356dfd4a2cf75a262d047fa300142cd034",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-1-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-1-4_seed-1-2": {
            "sha256": "128ba35693091ef0c208105eaffd810c7c291819dfaf9f186ca7ab74b49d9a84",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-1-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-1-4_seed-2-2": {
            "sha256": "455ec95d9efea965515cc42ac31d97aa1ce09b722da1e4eb8b31174509d7bdac",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-1-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-2-4_seed-0-2": {
            "sha256": "29fff2fe8929057998b5b83ea5a629ac891998da293d17c1df198adc769f1b65",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-2-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-2-4_seed-1-2": {
            "sha256": "38bf18859f03706835596dad8a82951f850b678504554219384121e3194dcf63",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-2-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-2-4_seed-2-2": {
            "sha256": "2c3161f516303912bad25e1bfda9e7368e3e57c58e3118c92436202740f5db36",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-2-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-3-4_seed-0-2": {
            "sha256": "f9eb7f893b11aba10f27230a4e75dfc101573e12b1ae7dc8e9002ca7c277856e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-3-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-3-4_seed-1-2": {
            "sha256": "3153e9a2395172ce052cba74c0e431bd0e4386f861a13ae6f975768e7e8686e9",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-3-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-3-4_seed-2-2": {
            "sha256": "930b161fb2bed80f3c8042c761facb9ec0246c6e68f84dbb1f52e72afeb2a652",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-3-4_seed-2-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-4-4_seed-0-2": {
            "sha256": "0d924657b78034b42f69a6c8540f5f44f2a38e9de4f5c4a61d6da7efaf98978e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-4-4_seed-0-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-4-4_seed-1-2": {
            "sha256": "cff335c59736fce843374d75cd15846dbd12c0ba3a028e1b558615ca2bc09a42",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-4-4_seed-1-2.zip",
        },
        "image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-4-4_seed-2-2": {
            "sha256": "9d695060deb9324d05f2181ece5e9dccc0caceb17ab5db24d4de8226a3bef15f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2025-03-07_13-00-00_survival-inclusion_palm_stacked_image_nested-4-4_seed-2-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-0-4_seed-0-2": {
            "sha256": "3e0d761ebca1e4f468c7894539055a21e5321619af7c14832b7bcff304a44991",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-0-4_seed-0-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-0-4_seed-1-2": {
            "sha256": "62399b64f77e0838949b1da38bae80d84cf59d3e59d6c4f987e21f13dd4bf2a7",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-0-4_seed-1-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-0-4_seed-2-2": {
            "sha256": "e04ac7e92c48ef9d57a955daff64f4a8f37d779c1a52111fc88ecb1f4a5e3c68",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-0-4_seed-2-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-1-4_seed-0-2": {
            "sha256": "7c636bc04acc48b0d90ec24c45dfc54daa43c36567e879894fbe46ddb72fcac4",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-1-4_seed-0-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-1-4_seed-1-2": {
            "sha256": "a717538d58256532d10517284c3df3aa2f7fdd1ef55c3d4bf94ba6ca93e2219a",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-1-4_seed-1-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-1-4_seed-2-2": {
            "sha256": "9e13d3d8ab83ef0ffc3c570f4accd00b11439c54e9b0882e5e132f398e01aea1",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-1-4_seed-2-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-2-4_seed-0-2": {
            "sha256": "e717cb17fae6c0773fd431b6f7f92bccc78f88e711f7ecd6bb6ae5225e13025e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-2-4_seed-0-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-2-4_seed-1-2": {
            "sha256": "ba91dcf32aab500a9ae3d223f2693d77884d8463a512d044ed3e074922f81a62",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-2-4_seed-1-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-2-4_seed-2-2": {
            "sha256": "c2aad824caec609f7c584e20ae4bb0c254f632d45e433f13298a40cbaa9d7f8f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-2-4_seed-2-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-3-4_seed-0-2": {
            "sha256": "4beec1e324a84e961740dfe4832d3b9d3125a5f77a6a7a24c3812f59d7993521",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-3-4_seed-0-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-3-4_seed-1-2": {
            "sha256": "e6e829660759e09de37cba03f10332d872fb51483ca8bc500d69abb45c396fa4",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-3-4_seed-1-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-3-4_seed-2-2": {
            "sha256": "98b7c39bb02293ccb3835cc585ee0524833083b793ad00e45656f70293165ce2",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-3-4_seed-2-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-4-4_seed-0-2": {
            "sha256": "8cbd940576ba1b9c4b2a25ca8607d3c13ef5ac36c15084e7f6061223a977408b",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-4-4_seed-0-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-4-4_seed-1-2": {
            "sha256": "261fe5f9e447cd2929cfa24e2a9c17c2713b4e1b1c592c254c0469306281fcd1",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-4-4_seed-1-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-4-4_seed-2-2": {
            "sha256": "6ce96bc98d7fe885593c7e1a8b6feb9da6fd11a7ea6e137dd62be34e6442bc6f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_finger_median_nested-4-4_seed-2-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-0-4_seed-0-2": {
            "sha256": "0152cc63665e6c44f0d9956f5dfcbfa02fa29f7f3935446a5a0db6bdbaae65e2",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-0-4_seed-0-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-0-4_seed-1-2": {
            "sha256": "d3a8d0945a366bad6ab2a82b6f13ea3afbf47db2ad925761343e3d2e56d70a82",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-0-4_seed-1-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-0-4_seed-2-2": {
            "sha256": "e66707af09110f9b515b53082fe9cdae7c10d288a03f3868948569aab34f31ce",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-0-4_seed-2-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-1-4_seed-0-2": {
            "sha256": "1afe2c2daf6eec8f33357c391f0987a5ecf694074de0fb9d34979a4062ecd7b1",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-1-4_seed-0-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-1-4_seed-1-2": {
            "sha256": "0e4a5cc9cb5a4ce2aaa13b72c3fea1b3fa92cebd917607ca9f7caa77ec696d8c",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-1-4_seed-1-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-1-4_seed-2-2": {
            "sha256": "9cc4923316d80c3367c664987e9c4140a2a1f3e47f8d78c8b5ab7c1a17ed48e1",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-1-4_seed-2-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-2-4_seed-0-2": {
            "sha256": "ef042b42cd610363c1a5d14fdee331215c4d71355ff957bac7f74ec730771a88",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-2-4_seed-0-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-2-4_seed-1-2": {
            "sha256": "77a3019b4b900a26a6085468c6d5e7c7b540b6e4fdbf36ad1fc7c69a6db81a85",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-2-4_seed-1-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-2-4_seed-2-2": {
            "sha256": "cf7f4d402c0304077933be1b225b75f3c1fafdde1388734b71dc23a0095ba149",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-2-4_seed-2-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-3-4_seed-0-2": {
            "sha256": "cafe8da5d5499c67a917297a5f87c9121579f1de5552dee9d97b71ad4dbdc530",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-3-4_seed-0-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-3-4_seed-1-2": {
            "sha256": "73d596c9e706d761444f5a08be74cd1654dfa67123ff56159f611c12aed7b117",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-3-4_seed-1-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-3-4_seed-2-2": {
            "sha256": "94d500db9b75465603f427234c1654642cbad5fa500c9dd2fb01f6eb0dc08beb",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-3-4_seed-2-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-4-4_seed-0-2": {
            "sha256": "20e7ca2d423080dd3a87182ce18f4fd13f521a2dd42aefb84bd18ecab4834a05",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-4-4_seed-0-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-4-4_seed-1-2": {
            "sha256": "49a658b024bdd902fbfdb8eb6da30137af7d482d8584ec00fe94e78854578db2",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-4-4_seed-1-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-4-4_seed-2-2": {
            "sha256": "9f04b45d8b338fd61db1bc737ea52304a961ea23180d01e54a44ea49b2bb99b5",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_sepsis-inclusion_palm_median_nested-4-4_seed-2-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-0-4_seed-0-2": {
            "sha256": "7fbdd2b1b785f258a90c688da2c4a3e19c3e9b439c5841643d0f35abe2108d4e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-0-4_seed-0-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-0-4_seed-1-2": {
            "sha256": "5a439427291c6287567b2fd2bfbb6cb7369b4c72bd5c6a83dd5e13ec53c0c2b3",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-0-4_seed-1-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-0-4_seed-2-2": {
            "sha256": "d1ff48686e92a97c0de426391b471ed9f010ec457b2a6e0c07e44210b5a4e29f",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-0-4_seed-2-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-1-4_seed-0-2": {
            "sha256": "ff3c86bbe2547c50dd29664b72fb976e5e3d4491918adfe5449ca455fdbe6fc1",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-1-4_seed-0-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-1-4_seed-1-2": {
            "sha256": "6dc13c8f19e397844696d726918d1975d96486d8333ace4ce639e837b68caf6e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-1-4_seed-1-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-1-4_seed-2-2": {
            "sha256": "6e92fb4a56879c5b204824bf3cf4b9a58f0eaa696370e2aced9564e085c9b136",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-1-4_seed-2-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-2-4_seed-0-2": {
            "sha256": "1935eaa088b403a8552fa30d42aacb1c382936b9a3575e8a5497e6139fa4e659",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-2-4_seed-0-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-2-4_seed-1-2": {
            "sha256": "7c30fe10b54ecc18eb3c3984aad4f8ad52f0a1e130a841bc12a5e4e623c7304d",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-2-4_seed-1-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-2-4_seed-2-2": {
            "sha256": "069a6825c85cbc5fb2bce6f9a8f8c4df1663496da3294db49490c457de915075",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-2-4_seed-2-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-3-4_seed-0-2": {
            "sha256": "2177c5635ad80f4feb17e4b88714a2628d71e67d5eb96f8984974b74dcbd7a58",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-3-4_seed-0-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-3-4_seed-1-2": {
            "sha256": "fac26baa0990176842b5eaebf4b7d8ab035a2ac62b89b96c2767407c3f3c0f73",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-3-4_seed-1-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-3-4_seed-2-2": {
            "sha256": "0c033eab42824987f8019b6248cffba2aeef7b9328b2a1c61fbb2e4fb35116b0",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-3-4_seed-2-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-4-4_seed-0-2": {
            "sha256": "beba5c6825dda14a55df80612128ba73ed823360115e6cc12d8338c3565b78b8",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-4-4_seed-0-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-4-4_seed-1-2": {
            "sha256": "92140b76eaebeaf5fe720f60027dc9bd1a201ba6fd041d2c5f4ca959436f676e",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-4-4_seed-1-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-4-4_seed-2-2": {
            "sha256": "f05498c381888b76734385470eea2537543817adae13ee75ea6cd33fcf47e8dc",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_finger_median_nested-4-4_seed-2-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-0-4_seed-0-2": {
            "sha256": "529e206afb9c95e0e842629b825b845eaecaa5298f51cc0b0e6a88dbc3be97c2",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-0-4_seed-0-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-0-4_seed-1-2": {
            "sha256": "b461896a4add1ce1aca54e0d13028b98294795a91e79b28cae77dcd9496d8e4a",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-0-4_seed-1-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-0-4_seed-2-2": {
            "sha256": "43b7388ff99823a4862170c886ae6d416eb8681437ba0a1da681a21c45783f68",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-0-4_seed-2-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-1-4_seed-0-2": {
            "sha256": "a77685681aea9ab565285c8f40bbd8195a1279cd0c18a44d4b2d47eba189538b",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-1-4_seed-0-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-1-4_seed-1-2": {
            "sha256": "6a7fb7ec62cce877dea5305f61e53eb5acea694e89d36e765092b89827552075",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-1-4_seed-1-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-1-4_seed-2-2": {
            "sha256": "48a8ec32b7ba7a3f1b8c7dbac7b9b12317de37085ed440d1919bfb050832b264",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-1-4_seed-2-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-2-4_seed-0-2": {
            "sha256": "2f7d52f3fb01d33b625c0844d80b33dd128d6c871f75c8b0ee568462a2bae65d",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-2-4_seed-0-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-2-4_seed-1-2": {
            "sha256": "c2fc85b7d9e6b9127a0abada8e55b1622839175c636b77959b0cd5f7e7132492",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-2-4_seed-1-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-2-4_seed-2-2": {
            "sha256": "c231ff0e32d69e5b1d4a5cdca6c00459f29a8f43a601e1fcca29ac77c5be576a",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-2-4_seed-2-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-3-4_seed-0-2": {
            "sha256": "ecc6a58cb2f606afd9d11f0ea3c059d6775dd838de6963ae96d30c5927d1084c",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-3-4_seed-0-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-3-4_seed-1-2": {
            "sha256": "747f6ca0c830224e10b85393280b736fe57a6a0f116ad590edab40f91c726114",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-3-4_seed-1-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-3-4_seed-2-2": {
            "sha256": "3da0ee4a65091dd148b326d064b8a39f967581162ebc4ebe99be2a73121ff7fc",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-3-4_seed-2-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-4-4_seed-0-2": {
            "sha256": "5b061862d1d923d67fad9ac02d36db87183ab81e93febfa8bb03a2e2b0685782",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-4-4_seed-0-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-4-4_seed-1-2": {
            "sha256": "9e1b1c3a07d01a716fd39876b58b3a4368a5d6001baac499b0d43cd57d58c2a8",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-4-4_seed-1-2.zip",
        },
        "median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-4-4_seed-2-2": {
            "sha256": "02a067346eb2cc18b4f4655fbc78ca8086e1c7ce82ea3d1e8e33645d73f84472",
            "url": "https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/median_pixel@2025-03-07_13-00-00_survival-inclusion_palm_median_nested-4-4_seed-2-2.zip",
        },
    })

    def __init__(self, config: Config, fold_name: str = None):
        """
        Base class for all model classes. It implements some logic to automatically replace the weights of a network with the weights from another pretrained network.

        If a model class inherits from this class it becomes very easy to use the same model class again with pretrained weights. You only need to specify the pretrained network in the config and then the weights get replaced automatically. In the config, you have two options to reference the pretrained network (see also the config.schema file):
        - With its path (absolute or relative to the training directory): `config["model/pretrained_model/path"] = "image/my_training_run"`
        - With a dictionary of the relevant properties: `config["model/pretrained_model"] = {"model": "image", "run_folder": "my_training_run"}`

        In case more than one fold is found in the run folder, the one with the highest score will be used per default. Alternatively, you can also specify the fold explicitly (either by appending it to the path or via the `model/pretrained_model/fold_name` config attribute).

        If the pretrained network is set in either way, the `_load_pretrained_model()` method of this class gets called at the end of your `__init__` (via the `__post__init__` method). Usually, we don't want all the weights of the pretrained network, for example, the segmentation head is normally not useful if the number of classes in the new task is different from the number of classes of the pretrained task. This is why those weights are skipped by default. You can control what should be skipped in your model by altering the `skip_keys_pattern` attribute of this class:
        >>> class MyModel(HTCModel):
        ...     def __init__(self, config: Config):
        ...         super().__init__(config)
        ...         self.skip_keys_pattern.add("decoder")
        >>> my_model = MyModel(Config({}))
        >>> sorted(my_model.skip_keys_pattern)
        ['classification_head', 'decoder', 'heads.heads', 'segmentation_head']

        Args:
            config: Configuration object of the training run.
            fold_name: the name of the fold being trained. This is used to find the correct parameter for temperature scaling.
        """
        super().__init__()
        self.config = config
        self.fold_name = fold_name

        # Default keys to load/skip for pretraining
        # Subclasses can modify these sets by adding elements to it or replacing them
        # If the key starts with "model.", then this usually corresponds to the name of the attribute in the lightning class
        # load_keys_pattern defines a dictionary with search and replace rules
        self.load_keys_pattern = {"model.": ""}
        # If any of the following names occurs in the state dict, they will be skipped
        self.skip_keys_pattern = {"segmentation_head", "classification_head", "heads.heads"}

        # Check for every model once whether the input is properly L1 normalized
        self._normalization_handle = self.register_forward_pre_hook(self._normalization_check)

    def __post__init__(self):
        if self.config["model/pretrained_model"]:
            self._load_pretrained_model()

        # We initialize temperature scaling only after the pretrained model may be loaded because that could change the fold name
        if self.config.get("post_processing/calibration") is not None:
            factors = self.config["post_processing/calibration/scaling"]
            biases = self.config["post_processing/calibration/bias"]
            if self.fold_name not in factors or self.fold_name not in biases:
                settings.log.warning(
                    "Found temperature scaling parameters in the config but not for the requested fold"
                    f" {self.fold_name} (temperature scaling will not be applied)"
                )
            else:
                self.register_buffer("_temp_factor", torch.tensor(factors[self.fold_name]), persistent=False)
                self.register_buffer("_temp_bias", torch.tensor(biases[self.fold_name]), persistent=False)
                if (nll_prior := self.config.get("post_processing/calibration/nll_prior")) is not None:
                    self.register_buffer("_nll_prior", torch.tensor(nll_prior), persistent=False)
                else:
                    self._nll_prior = None
                self.register_forward_hook(self._temperature_scaling)

    def _temperature_scaling(
        self, module: nn.Module, module_in: tuple, output: torch.Tensor | dict[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if type(output) == dict:
            logits = output["class"]
        else:
            logits = output

        # Actual scaling (based on: https://github.com/luferrer/psr-calibration/blob/master/psrcal/calibration.py)
        scores = logits * self._temp_factor + self._temp_bias
        if self._nll_prior is not None:
            scores = scores - self._nll_prior  # - because nll = -log(prior)

        scores = scores - torch.logsumexp(scores, axis=-1, keepdim=True)

        if type(output) == dict:
            output["class"] = scores
        else:
            output = scores

        return output

    def _normalization_check(self, module: nn.Module, module_in: tuple) -> None:
        if self.config["input/n_channels"] == 100 and (
            self.config["input/preprocessing"] == "L1" or self.config["input/normalization"] == "L1"
        ):
            features = module_in[0]

            # Find the channel dimension
            channel_dim = None
            for dim, length in enumerate(features.shape):
                if length == self.config["input/n_channels"]:
                    channel_dim = dim
                    break

            # It is possible that we cannot find the channel dimensions, e.g. for the pixel model if an input != 100 is passed to the model
            # It is also ok if a spectrum only contains zeros since this is done by some augmentations
            if channel_dim is not None:
                channel_sum = features.abs().sum(dim=channel_dim)
                # Either all values must be close to 1 (or 0)
                all_valid = torch.all(
                    torch.isclose(channel_sum, torch.tensor(1.0, device=features.device), atol=0.1)
                    | torch.isclose(channel_sum, torch.tensor(0.0, device=features.device), atol=0.1)
                )

                # Or the mean/std must fit on average for the non-zero elements (because single pixels may be off)
                nonzeros = channel_sum[channel_sum.nonzero(as_tuple=True)]
                if nonzeros.nelement() == 0:
                    mean = torch.tensor(torch.nan, device=features.device)
                    std = torch.tensor(torch.nan, device=features.device)
                    average_valid = False
                else:
                    mean = nonzeros.mean()
                    std = nonzeros.std(correction=0)  # We may encounter single-element tensors here
                    average_valid = torch.isclose(
                        mean, torch.tensor(1.0, device=features.device), atol=0.01
                    ) and torch.isclose(std, torch.tensor(0.0, device=features.device), atol=0.01)

                if not (all_valid or average_valid):
                    settings.log.warning(
                        f"The model {module.__class__.__name__} expects L1 normalized input but the features"
                        f" ({features.shape = }) do not seem to be L1 normalized:\naverage per pixel ="
                        f" {mean}\nstandard deviation per pixel ="
                        f" {std}\nThis check is only performed for the first batch."
                    )

        # We only perform this check for the first batch
        self._normalization_handle.remove()

    def _load_pretrained_model(self) -> None:
        model_path = None
        pretrained_dir = None
        map_location = None if torch.cuda.is_available() else "cpu"

        if self.config["model/pretrained_model/path"]:
            possible_locations = HTCModel._get_possible_locations(Path(self.config["model/pretrained_model/path"]))
            for location in possible_locations:
                if location.is_dir():
                    pretrained_dir = location
                elif location.is_file():
                    model_path = location
                    break
        else:
            pretrained_dir = HTCModel.find_pretrained_run(
                self.config["model/pretrained_model/model"],
                self.config["model/pretrained_model/run_folder"],
            )
        assert pretrained_dir is not None or model_path is not None, (
            f"Could not find the pretrained model as specified in the config: {self.config['model/pretrained_model']}"
        )

        if model_path is None:
            if self.config["model/pretrained_model/fold_name"]:
                if pretrained_dir.name.startswith("fold"):
                    assert pretrained_dir.name == self.config["model/pretrained_model/fold_name"], (
                        f"The found pretrained directory {pretrained_dir} does not match the fold name"
                        f" {self.config['model/pretrained_model/fold_name']}"
                    )
                else:
                    pretrained_dir = pretrained_dir / self.config["model/pretrained_model/fold_name"]

            model_path = HTCModel.best_checkpoint(pretrained_dir)

        assert model_path is not None, "Could not find the best model"
        pretrained_model = torch.load(model_path, map_location=map_location)

        # Change state dict keys
        model_dict = self.state_dict()
        num_keys_loaded = 0
        skipped_keys = []

        for k in pretrained_model["state_dict"].keys():
            if any(skip_key_pattern in k for skip_key_pattern in self.skip_keys_pattern):
                skipped_keys.append(k)
                continue

            for pattern, replace in self.load_keys_pattern.items():
                if pattern in k:
                    # If the input channels are different then use the same trick as used in segmentation_models library
                    # e.g. in case of 3 channels new_weight[:, i] = pretrained_weight[:, i % 3]

                    new_in_channel = model_dict[k.replace(pattern, replace)].shape
                    pretrained_in_channel = pretrained_model["state_dict"][k].shape

                    if new_in_channel != pretrained_in_channel:
                        new_in_channel, pretrained_in_channel = new_in_channel[1], pretrained_in_channel[1]
                        for c in range(new_in_channel):
                            model_dict[k.replace(pattern, replace)][:, c] = pretrained_model["state_dict"][k][
                                :, c % pretrained_in_channel
                            ]

                        model_dict[k.replace(pattern, replace)] = (
                            model_dict[k.replace(pattern, replace)] * pretrained_in_channel
                        ) / new_in_channel
                    else:
                        model_dict[k.replace(pattern, replace)] = pretrained_model["state_dict"][k]
                    num_keys_loaded += 1

        if self.fold_name is None:
            self.fold_name = model_path.parent.name

        # Load the new weights
        self.load_state_dict(model_dict)
        if num_keys_loaded == 0:
            settings.log.warning(f"No key has been loaded from the pretrained dir: {pretrained_dir}")
        elif num_keys_loaded + len(skipped_keys) != len(model_dict):
            settings.log.warning(
                f"{num_keys_loaded} keys were changed in the model ({len(skipped_keys)} keys were skipped:"
                f" {skipped_keys}) but the model contains {len(model_dict)} keys. This means that some parameters of"
                " the model remain unchanged"
            )
        else:
            settings.log.info(
                f"Successfully loaded the pretrained model ({len(skipped_keys)} keys were skipped: {skipped_keys})."
            )

    @classmethod
    def pretrained_model(
        cls,
        model: str = None,
        run_folder: str = None,
        path: str | Path = None,
        fold_name: str = None,
        n_classes: int = None,
        n_channels: int = None,
        pretrained_weights: bool = True,
        **model_kwargs,
    ) -> Self | list[Self]:
        """
        Load a pretrained segmentation model.

        You can directly use this model to train a network on your data. The weights will be initialized with the weights from the pretrained network, except for the segmentation head which is initialized randomly (and may also be different in terms of number of classes, depending on your data). The returned instance corresponds to the calling class (e.g. `ModelImage`) and you can also find it in the third column of the pretrained models table (cf. readme).

        For example, load the pretrained model for the image-based segmentation network:
        >>> from htc import ModelImage, Normalization
        >>> run_folder = "2022-02-03_22-58-44_generated_default_model_comparison"  # HSI model
        >>> model = ModelImage.pretrained_model(model="image", run_folder=run_folder)  # doctest: +ELLIPSIS
        [...]
        >>> input_data = torch.randn(1, 100, 480, 640)  # NCHW
        >>> input_data = Normalization(channel_dim=1)(input_data)  # Model expects L1 normalized input
        >>> model(input_data).shape
        torch.Size([1, 19, 480, 640])

        It is also possible to have a different number of classes as output or a different number of channels as input:
        >>> model = ModelImage.pretrained_model(
        ...     model="image", run_folder=run_folder, n_classes=3, n_channels=10
        ... )  # doctest: +ELLIPSIS
        [...]
        >>> input_data = torch.randn(1, 10, 480, 640)  # NCHW
        >>> model(input_data).shape
        torch.Size([1, 3, 480, 640])

        The patch-based models also use the `ModelImage` class but with a different input (here using the patch_64 model):
        >>> run_folder = "2022-02-03_22-58-44_generated_default_64_model_comparison"  # HSI model
        >>> model = ModelImage.pretrained_model(model="patch", run_folder=run_folder)  # doctest: +ELLIPSIS
        [...]
        >>> input_data = torch.randn(1, 100, 64, 64)  # NCHW
        >>> input_data = Normalization(channel_dim=1)(input_data)  # Model expects L1 normalized input
        >>> model(input_data).shape
        torch.Size([1, 19, 64, 64])

        The procedure is the same for the superpixel-based segmentation network but this time also using a different calling class (`ModelSuperpixelClassification`):
        >>> from htc import ModelSuperpixelClassification
        >>> run_folder = "2022-02-03_22-58-44_generated_default_model_comparison"  # HSI model
        >>> model = ModelSuperpixelClassification.pretrained_model(
        ...     model="superpixel_classification", run_folder=run_folder
        ... )  # doctest: +ELLIPSIS
        [...]
        >>> input_data = torch.randn(2, 100, 32, 32)  # NCHW
        >>> input_data = Normalization(channel_dim=1)(input_data)  # Model expects L1 normalized input
        >>> model(input_data).shape
        torch.Size([2, 19])

        And also the pixel network:
        >>> from htc import ModelPixel
        >>> run_folder = "2022-02-03_22-58-44_generated_default_model_comparison"  # HSI model
        >>> model = ModelPixel.pretrained_model(model="pixel", run_folder=run_folder)  # doctest: +ELLIPSIS
        [...]
        >>> input_data = torch.randn(2, 100)  # NC
        >>> input_data = Normalization(channel_dim=1)(input_data)  # Model expects L1 normalized input
        >>> model(input_data)["class"].shape
        torch.Size([2, 19])

        For the pixel model, you can specify a different number of classes but you do not need to set the number of input channels because the underlying convolutional operations directly operate along the channel dimension. Hence, you can just supply input data with a different number of channels and it will work as well.
        >>> model = ModelPixel.pretrained_model(model="pixel", run_folder=run_folder, n_classes=3)  # doctest: +ELLIPSIS
        [...]
        >>> input_data = torch.randn(2, 90)  # NC
        >>> model(input_data)["class"].shape
        torch.Size([2, 3])

        Retrieve a collection of models as list:
        >>> models = ModelImage.pretrained_model(
        ...     model="image", run_folder="2025-03-09_19-38-10_baseline_rat_nested-*-2"
        ... )  # doctest: +ELLIPSIS
        [...]
        >>> len(models)
        3

        Args:
            model: Basic model type like image or pixel (first column in the pretrained models table). This corresponds to the folder name in the first hierarchy level of the training directory.
            run_folder: Name of the training run from which the weights should be loaded, e.g. to select HSI or RGB models (fourth column in the pretrained models table). This corresponds to the folder name in the second hierarchy level of the training directory. If the run folder contains a wildcard `*` to indicate a collection of runs (e.g. "2025-03-09_19-38-10_baseline_rat_nested-*-2"), this function will return a list of models from this collection.
            path: Alternatively of specifying the model and run folder, you can also specify the path to the run directory, the fold directory or the path to the checkpoint file (*.ckpt) directly.
            fold_name: Name of the validation fold which defines the trained network of the run. If None, the model with the highest metric score will be used.
            n_classes: Number of classes for the network output. If None, uses the same setting as in the trained network (e.g. 18 organ classes + background for the organ segmentation networks).
            n_channels: Number of channels of the input. If None, uses the same settings as in the trained network (e.g. 100 channels). This is inspired by the timm library (https://timm.fast.ai/models#How-is-timm-able-to-use-pretrained-weights-and-handle-images-that-are-not-3-channel-RGB-images?), i.e. it repeats the weights according to the desired number of channels. Please not that this does not take any semantic of the input into account, e.g. the wavelength range or the filter functions of the camera.
            pretrained_weights: If True, overwrite the weights of the model with the weights from the pretrained model, i.e. make use of the pretrained model. If False, will still load (and download) the model but keep the weights randomly initialized. This mainly ensures that the same config is used for the pretrained model.
            model_kwargs: Additional keyword arguments passed to the model instance.

        Returns: Instance of the calling model class initialized with the pretrained weights. The model object will be an instance of `torch.nn.Module`.
        """

        def _construct_model(run_dir: Path) -> Self:
            config = Config(run_dir / "config.json")

            if pretrained_weights:
                if path is not None:
                    config["model/pretrained_model/path"] = path
                else:
                    config["model/pretrained_model/model"] = model
                    config["model/pretrained_model/run_folder"] = run_dir.name

            if fold_name is not None:
                config["model/pretrained_model/fold_name"] = fold_name
            if n_classes is not None:
                config["input/n_classes"] = n_classes
            if n_channels is not None:
                assert model != "pixel", (
                    "The parameter n_channels cannot be used with the pixel model. The number of channels are solely"
                    " determined by the input (see examples)"
                )
                config["input/n_channels"] = n_channels

            return cls(config, **model_kwargs)

        run_dir = HTCModel.find_pretrained_run(model, run_folder, path)
        if isinstance(run_dir, list):
            return [_construct_model(r) for r in run_dir]
        else:
            return _construct_model(run_dir)

    @staticmethod
    def find_pretrained_run(
        model_name: str = None, run_folder: str = None, path: str | Path = None
    ) -> Path | list[Path]:
        """
        Searches for a pretrained run either in the local results directory, in the local PyTorch model cache directory or it will attempt to download the model. For the local results directory, the following folders are searched:
        - `results/training/<model_name>/<run_folder>`
        - `results/pretrained_models/<model_name>/<run_folder>`
        - `results/<model_name>/<run_folder>`
        - `<model_name>/<run_folder>` (relative/absolute path)

        Args:
            model_name: Basic model type like image or pixel.
            run_folder: Name of the training run directory (e.g. 2022-02-03_22-58-44_generated_default_model_comparison). If the run folder contains a wildcard `*` to indicate a collection of runs (e.g. "2025-03-09_19-38-10_baseline_rat_nested-*-2"), this function will return all matching run directories.
            path: Alternatively to model_name and run_folder, you can also specify the path to the run directory (may also be relative to the results directory in one of the folders from above). If the path points to the fold directory or the checkpoint file (*.ckpt), the corresponding run directory will be returned.

        Returns: Path to the requested training run (run directory usually starting with a timestamp).
        """
        if path is not None:
            if type(path) is str:
                path = Path(path)

            possible_locations = HTCModel._get_possible_locations(path)
            for location in possible_locations:
                if location.is_dir():
                    if location.name.startswith("fold"):
                        # At this point, we are only interested in the run directory and not the fold directory
                        location = location.parent

                    if model_name is not None:
                        assert location.parent.name == model_name, (
                            f"The found location {location} does not match the given model_name {model_name}"
                        )
                    if run_folder is not None:
                        assert location.name == run_folder, (
                            f"The found location {location} does not match the given run_folder {run_folder}"
                        )

                    return location
                elif location.is_file():
                    # From the checkpoint file to the run directory
                    return location.parent.parent

            raise ValueError(
                f"Could not find the pretrained model. Tried the following locations: {possible_locations}"
            )
        else:
            assert path is None, "The path parameter is not used if model_name and run_folder are specified"
            assert model_name is not None and run_folder is not None, (
                "Please specify model_name and run_folder (e.g. in your config via the keys"
                " `model/pretrained_model/model` and `model/pretrained_model/run_folder`) if no path is given"
            )

            def _load_run_folder(run_folder: str) -> Path:
                # Option 1: local results directory
                if settings.results_dir is not None:
                    possible_locations = HTCModel._get_possible_locations(Path(model_name) / run_folder)
                    for run_dir in possible_locations:
                        if run_dir.is_dir():
                            settings.log_once.info(f"Found pretrained run in the local results dir at {run_dir}")
                            return run_dir

                # Option 2: local hub dir (cache folder)
                hub_dir = Path(torch.hub.get_dir()) / "htc_checkpoints"
                run_dir = hub_dir / model_name / run_folder
                if run_dir.is_dir():
                    settings.log_once.info(f"Found pretrained run in the local hub dir at {run_dir}")
                    return run_dir

                # Option 3: download the model to the local hub dir
                name = f"{model_name}@{run_folder}"
                assert name in HTCModel.known_models, (
                    f"Could not find the training run for {model_name}/{run_folder} (neither locally nor as download option)"
                )
                model_info = HTCModel.known_models[name]

                hub_dir.mkdir(parents=True, exist_ok=True)

                # Download the archive containing all trained models for the run (i.e. a model per fold)
                zip_path = hub_dir / f"{name}.zip"
                settings.log.info(f"Downloading pretrained model {name} since it is not locally available")
                torch.hub.download_url_to_file(model_info["url"], zip_path)

                # Extract the archive in the models dir with the usual structure (e.g. image/run_folder/fold_name)
                with ZipFile(zip_path) as f:
                    f.extractall(hub_dir)
                zip_path.unlink()

                assert run_dir.is_dir(), "run folder not available even after download"

                # Check file contents to catch download errors
                hash_cat = ""
                for f in sorted(run_dir.rglob("*"), key=lambda x: str(x).lower()):
                    if f.is_file():
                        hash_cat += sha256_file(f)

                hash_folder = hashlib.sha256(hash_cat.encode()).hexdigest()
                if model_info["sha256"] != hash_folder:
                    settings.log.error(
                        f"The hash of the folder (hash of the file hashes, {hash_folder}) does not match the expected hash"
                        f" ({model_info['sha256']}). The download of the model was likely not successful. The downloaded"
                        f" files are not deleted and are still available at {hub_dir}. Please check the files manually"
                        " (e.g. for invalid file sizes). If you want to re-trigger the download process, just delete the"
                        f" corresponding run directory {run_dir}"
                    )
                else:
                    settings.log.info(f"Successfully downloaded the pretrained run to the local hub dir at {run_dir}")

                return run_dir

            if "*" in run_folder:
                # Find all * from left to right
                max_indices = []
                for match in re.findall(r"\*-(\d+)", run_folder):
                    max_indices.append(int(match))

                assert len(max_indices) > 0, (
                    f"Could not infer any maximum index from {run_folder} for the run folder collection. The collection must for example be be named nested-0-2, nested-1-2, nested-2-2."
                )

                run_folders = []
                for indices in itertools.product(*[range(i + 1) for i in max_indices]):
                    nested_run_folder = run_folder
                    for i in indices:
                        nested_run_folder = nested_run_folder.replace("*", str(i), 1)

                    assert "*" not in nested_run_folder, f"Could not replace all * in {run_folder} with {indices}"
                    run_folders.append(_load_run_folder(nested_run_folder))

                return run_folders
            else:
                return _load_run_folder(run_folder)

    @staticmethod
    def best_checkpoint(path: Path) -> Path:
        """
        Searches for the best model checkpoint path within the given run directory across all available folds.

        Args:
            path: The path to the training run or to a specific fold.

        Returns: The path to the best model checkpoint.
        """
        # Choose the model with the highest dice score
        checkpoint_paths = sorted(path.rglob("*.ckpt"))
        if len(checkpoint_paths) == 1:
            model_path = checkpoint_paths[0]
        else:
            table_path = path / "validation_table.pkl.xz"
            if table_path.exists():
                # Best model based on the validation table
                df_val = pd.read_pickle(table_path)
                df_val = df_val.query("epoch_index == best_epoch_index and dataset_index == 0")
                config = Config(path / "config.json")

                # Best model per fold for surgical scene segmentation is based on dice metric
                if "dice_metric" in df_val.columns:
                    agg = MetricAggregation(df_val, config=config)
                    df_best = agg.grouped_metrics(domains=["fold_name", "epoch_index"])
                    df_best = df_best.groupby(["fold_name", "epoch_index"], as_index=False)["dice_metric"].mean()
                    df_best = df_best.sort_values(by=agg.metrics, ascending=False, ignore_index=True)
                    fold_dir = path / df_best.iloc[0].fold_name

                # Best model per fold for sepsis diagnosis and mortality prediction defaults to "fold_0"
                else:
                    fold_dir = path / "fold_0"

                checkpoint_paths = sorted(fold_dir.rglob("*.ckpt"))
                if len(checkpoint_paths) == 1:
                    model_path = checkpoint_paths[0]
                else:
                    checkpoint_paths = sorted(fold_dir.rglob(f"epoch={df_best.iloc[0].epoch_index}*.ckpt"))
                    assert len(checkpoint_paths) == 1, (
                        f"More than one checkpoint found for the epoch {df_best.iloc[0].epoch_index}: {checkpoint_paths}"
                    )
                    model_path = checkpoint_paths[0]
            else:
                model_path = checkpoint_paths[0]
                settings.log.warning(
                    f"Could not find the validation table at {table_path} but this is required to automatically"
                    f" determine the best model. The first found checkpoint will be used instead: {model_path}"
                )

        return model_path

    @staticmethod
    def markdown_table_segmentation() -> str:
        """
        Generate a markdown table with all known pretrained models for surgical scene segmentation (used in the README).

        If you want to update the table in the README, simple call this function on the command line:
        ```bash
        python -c "from htc import HTCModel; print(HTCModel.markdown_table_segmentation())"
        ```
        and replace the table in the README with the resulting table.
        """
        from htc.models.image.ModelImage import ModelImage
        from htc.models.pixel.ModelPixel import ModelPixel
        from htc.models.pixel.ModelPixelRGB import ModelPixelRGB
        from htc.models.superpixel_classification.ModelSuperpixelClassification import ModelSuperpixelClassification

        table_lines = [
            "| model type | modality | class | run folder |",
            "| ----------- | ----------- | ----------- | ----------- |",
        ]
        model_lines = []

        for name, download_info in HTCModel.known_models.items():
            model_type, run_folder = name.split("@")
            model_info = run_info(settings.training_dir / model_type / run_folder)

            if "2025-03-07_13-00-00" in run_folder:
                # Skip sepsis diagnosis and mortality prediction models
                continue

            if model_type == "superpixel_classification":
                ModelClass = ModelSuperpixelClassification
            elif model_type == "pixel":
                if "parameters" in run_folder or "rgb" in run_folder:
                    ModelClass = ModelPixelRGB
                else:
                    ModelClass = ModelPixel
            else:
                ModelClass = ModelImage

            # Check that the model can be loaded
            model = ModelClass.pretrained_model(model_type, run_folder=run_folder)
            assert isinstance(model, nn.Module)

            class_name = ModelClass.__name__
            class_path = Path(inspect.getfile(ModelClass)).relative_to(settings.src_dir)

            run_folder_md = ""
            if "nested" in run_folder:
                if "nested-0" not in run_folder:
                    # We only add one collection item to the table (not all nested runs)
                    continue

                run_folder_collection = re.sub(r"nested-\d+", "nested-*", run_folder)

                run_folders = HTCModel.find_pretrained_run(model_type, run_folder_collection)
                links = ", ".join(
                    f"[{i}]({HTCModel.known_models[f'{model_type}@{f.name}']['url']})"
                    for i, f in enumerate(run_folders)
                )
                run_folder_md = f"`{run_folder_collection}` (outer folds: {links})"

            else:
                run_folder_md = f"[`{run_folder}`]({download_info['url']})"

            model_lines.append(
                f"| {model_type} | {model_info['model_type']} | [`{class_name}`](./{class_path}) | {run_folder_md} |"
            )

        table_lines += reversed(model_lines)
        return "\n".join(table_lines)

    @staticmethod
    def markdown_table_sepsis_icu() -> str:
        """
        Generate a markdown table with all known pretrained models for sepsis diagnosis and mortality prediction (used in the README).

        If you want to update the table in the README, simple call this function on the command line:
        ```bash
        python -c "from htc import HTCModel; print(HTCModel.markdown_table_sepsis_icu())"
        ```
        and replace the table in the README with the resulting table.
        """
        from htc.models.pixel.ModelPixel import ModelPixel
        from htc.models.superpixel_classification.ModelSuperpixelClassification import ModelSuperpixelClassification

        table_lines = [
            "| model type | modality | class | run folder |",
            "| ----------- | ----------- | ----------- | ----------- |",
        ]
        model_lines = []

        for name in HTCModel.known_models.keys():
            model_type, run_folder = name.split("@")
            model_info = run_info(settings.training_dir / model_type / run_folder)

            if "2025-03-07_13-00-00" not in run_folder:
                # Skip surgical scene segmentation models
                continue

            if model_type == "median_pixel":
                ModelClass = ModelPixel
            elif model_type == "image":
                ModelClass = ModelSuperpixelClassification

            # Check that the model can be loaded
            model = ModelClass.pretrained_model(model_type, run_folder=run_folder)
            assert isinstance(model, nn.Module)

            class_name = ModelClass.__name__
            class_path = Path(inspect.getfile(ModelClass)).relative_to(settings.src_dir)

            run_folder_md = ""
            if "nested-0" not in run_folder:
                # We only add one collection item to the table (not all nested runs)
                continue

            # In the sepsis project, we have repeated runs across three different seeds - here as well, we only want to show one of them
            if "seed-0" not in run_folder:
                continue

            run_folder_collection = re.sub(r"nested-\d+", "nested-*", run_folder)
            run_folder_collection = re.sub(r"seed-\d+", "seed-*", run_folder_collection)
            run_folder_md = f"`{run_folder_collection}`"

            model_lines.append(
                f"| {model_type} | {model_info['model_type']} | [`{class_name}`](./{class_path}) | {run_folder_md} |"
            )

        table_lines += reversed(model_lines)
        return "\n".join(table_lines)

    @staticmethod
    def _get_possible_locations(path: Path) -> list[Path]:
        return [
            settings.training_dir / path,
            settings.results_dir / path,
            settings.results_dir / "pretrained_models" / path,
            path,
        ]
