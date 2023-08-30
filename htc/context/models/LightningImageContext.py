# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from htc.context.models.ContextEvaluationMixin import ContextEvaluationMixin
from htc.models.image.LightningImage import LightningImage


class LightningImageContext(ContextEvaluationMixin, LightningImage):
    pass
