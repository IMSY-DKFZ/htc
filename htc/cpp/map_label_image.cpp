// SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
// SPDX-License-Identifier: MIT

#include <torch/extension.h>

torch::Tensor map_label_image(const torch::Tensor& label_image, std::unordered_map<int64_t, std::tuple<float, float, float, float>>& label_color_mapping) {
    auto mapped_image = torch::empty({label_image.size(0), label_image.size(1), 4}, torch::kFloat32);
    
    auto label_image_a = label_image.accessor<int64_t, 2>();
    auto mapped_image_a = mapped_image.accessor<float, 3>();
    
    for (int row = 0; row < label_image_a.size(0); ++row) {
        for (int col = 0; col < label_image_a.size(1); ++col) {
            auto label = label_image_a[row][col];
            const auto color = label_color_mapping[label];
            mapped_image_a[row][col][0] = std::get<0>(color);
            mapped_image_a[row][col][1] = std::get<1>(color);
            mapped_image_a[row][col][2] = std::get<2>(color);
            mapped_image_a[row][col][3] = std::get<3>(color);
        }
    }
    
    return mapped_image;
}
