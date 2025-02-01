// SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
// SPDX-License-Identifier: MIT

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> spxs_predictions(torch::Tensor& spxs,
                                                          torch::Tensor& labels,
                                                          torch::Tensor& mask,
                                                          int n_classes) {
    auto shape = labels.sizes();
    spxs = spxs.flatten();
    labels = labels.flatten();
    mask = mask.flatten();

    auto spxs_a = spxs.accessor<int64_t, 1>();
    auto labels_a = labels.accessor<int64_t, 1>();
    auto mask_a = mask.accessor<bool, 1>();

    // Count for each superpixel which labels the corresponding pixels have
    auto spx_label_counts = torch::zeros({spxs.max().item<int64_t>() + 1, n_classes}, torch::kInt32);
    auto spx_label_counts_a = spx_label_counts.accessor<int, 2>();

    // Iterate over the image
    for (int i = 0; i < spxs_a.size(0); ++i) {
        if (mask_a[i]) {
            spx_label_counts_a[spxs_a[i]][labels_a[i]] += 1;
        }
    }

    // The label of the superpixel is the mode of the labels, i.e. the max count
    auto spx_label = spx_label_counts.argmax(1);  // The index of the max count corresponds to the label of the
                                                  // superpixel (mask-only superpixels are assigned to the background)
    auto spx_label_a2 = spx_label.accessor<int64_t, 1>();

    // Project the calculated labels for each superpixel back to the image
    auto predictions = torch::empty(shape[0] * shape[1], torch::kInt64);
    auto predictions_a = predictions.accessor<int64_t, 1>();

    for (int i = 0; i < spxs_a.size(0); ++i) {
        predictions_a[i] = spx_label_a2[spxs_a[i]];
    }

    return std::make_tuple(predictions.reshape(shape), spx_label_counts);
}
