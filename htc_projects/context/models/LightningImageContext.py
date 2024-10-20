# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from htc.models.image.LightningImage import LightningImage
from htc_projects.context.models.ContextEvaluationMixin import ContextEvaluationMixin


class LightningImageContext(ContextEvaluationMixin, LightningImage):
    pass
