# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch.nn.functional as F

from htc.models.common.loss import KLDivLossWeighted, SuperpixelLoss


class TestKLDivLossWeighted:
    def test_weight(self) -> None:
        predictions = torch.tensor([[10, 1, 2], [2, 5, 1]], dtype=torch.float32)
        predictions = F.log_softmax(predictions, dim=1)
        targets = torch.tensor([[0.9, 0.05, 0.05], [0, 1, 0]], dtype=torch.float32)

        loss_equal1 = F.kl_div(predictions, targets, reduction="batchmean")
        loss_equal2 = KLDivLossWeighted(weight=torch.ones(3))(predictions, targets)
        assert loss_equal1 == pytest.approx(loss_equal2)

        loss_equal1 = F.kl_div(predictions, targets, reduction="batchmean")
        loss_equal2 = KLDivLossWeighted(weight=torch.ones(3) / 3)(predictions, targets)
        assert loss_equal1 == pytest.approx(loss_equal2)

        weights = torch.tensor([0, 0, 1])
        loss_weighted = KLDivLossWeighted(weight=weights)(predictions, targets)

        predictions[0, 0] = 10  # This has no effect since the corresponding weight is zero
        loss_weighted2 = KLDivLossWeighted(weight=weights)(predictions, targets)
        assert loss_weighted == pytest.approx(loss_weighted2)

    def test_ce_identical(self) -> None:
        torch.manual_seed(0)
        # The weighted kl div loss is only identical to the weighted ce loss when one-hot targets are used

        predictions = torch.tensor([[10, 1, 2], [2, 5, 1]], dtype=torch.float32)
        predictions_log = F.log_softmax(predictions, dim=1)
        targets_prob = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)
        targets = targets_prob.argmax(dim=1)

        ce_loss = F.cross_entropy(predictions, targets)
        kl_loss = F.kl_div(predictions_log, targets_prob, reduction="batchmean")
        assert ce_loss == pytest.approx(kl_loss)

        weights = torch.tensor([0.4, 0.2, 0.4])
        ce_loss = F.cross_entropy(predictions, targets, weight=weights)
        kl_loss = KLDivLossWeighted(weight=weights)(predictions_log, targets_prob)
        assert ce_loss == pytest.approx(kl_loss)

        predictions = torch.rand(10, 3)
        predictions_log = F.log_softmax(predictions, dim=1)
        targets = torch.randint(0, 3, (10,))
        targets_prob = F.one_hot(targets).type(torch.float32)

        ce_loss = F.cross_entropy(predictions, targets, weight=weights)
        kl_loss = KLDivLossWeighted(weight=weights)(predictions_log, targets_prob)
        assert ce_loss == pytest.approx(kl_loss)


class TestSuperpixelLoss:
    def test_loss(self):
        spx_loss = SuperpixelLoss()

        spxs = torch.tensor([0, 0, 1, 1, 2, 2, 2], dtype=torch.int64)
        predictions = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [0.5, 0.5]], dtype=torch.float32)
        predictions_better = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0]], dtype=torch.float32)
        predictions_worse = torch.tensor(
            [[0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [0.5, 0.5], [0.5, 0.5]], dtype=torch.float32
        )
        predictions_worst = torch.tensor(
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=torch.float32
        )

        loss = spx_loss(predictions, spxs)
        loss_better = spx_loss(predictions_better, spxs)
        loss_worse = spx_loss(predictions_worse, spxs)
        loss_worst = spx_loss(predictions_worst, spxs)

        assert loss > loss_better, "predictions are more unique --> lower loss"
        assert loss < loss_worse, "predictions are less unique --> higher loss"
        assert loss_worst == pytest.approx(0.5), "max gini loss on max impurity"
