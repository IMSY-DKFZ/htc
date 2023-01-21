// SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
// SPDX-License-Identifier: MIT

#include <torch/extension.h>

torch::Tensor segmentation_mask(const torch::Tensor& label_image, std::map<std::tuple<int, int, int>, int>& color_mapping) {
    auto seg = torch::empty({label_image.size(0), label_image.size(1)}, torch::kUInt8);
    
    auto seg_a = seg.accessor<unsigned char, 2>();
    auto label_a = label_image.accessor<unsigned char, 3>();
    
    for (int row = 0; row < label_a.size(0); ++row) {
        for (int col = 0; col < label_a.size(1); ++col) {
            auto pixel = label_a[row][col];
            auto color = std::make_tuple(pixel[0], pixel[1], pixel[2]);

            if (color_mapping.contains(color)) {
                auto label = color_mapping[color];
                seg_a[row][col] = label;
            }
        }
    }
    
    return seg;
}
