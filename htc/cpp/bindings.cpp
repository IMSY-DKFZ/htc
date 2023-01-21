// SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
// SPDX-License-Identifier: MIT

#include <torch/extension.h>
#include "tensor_mapping.h"
#include "hierarchical_bootstrapping.h"

std::tuple<torch::Tensor, torch::Tensor> spxs_predictions(torch::Tensor& spxs, torch::Tensor& labels, torch::Tensor& mask, int n_classes);
torch::Tensor segmentation_mask(const torch::Tensor& label_image, std::map<std::tuple<int, int, int>, int>& color_mapping);
std::vector<std::vector<int>> kfold_combinations(const std::vector<int>& subject_indices, const std::map<int, std::vector<int>>& subject_labels, int min_labels, int n_groups = 5);
torch::Tensor nunique(const torch::Tensor& in, int64_t dim);
torch::Tensor map_label_image(const torch::Tensor& label_image, std::unordered_map<int64_t, std::tuple<float, float, float, float>>& label_color_mapping);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spxs_predictions", &spxs_predictions, "Superpixel prediction based on the modal value.");
  m.def("segmentation_mask", &segmentation_mask, "Creates a segmentation mask based on a label image and a corresponding color mapping.");
  m.def("tensor_mapping_integer", &tensor_mapping_integer, "Remaps values of a tensor based on a dict.");
  m.def("tensor_mapping_floating", &tensor_mapping_floating, "Remaps values of a tensor based on a dict.");
  m.def("kfold_combinations", &kfold_combinations, "Evaluate kfold combinations.");
  m.def("nunique", &nunique, "Counts unique elements along dim.");
  m.def("map_label_image", &map_label_image, "Create color images based on label images (map label ids to colors).");
  m.def("hierarchical_bootstrapping", &hierarchical_bootstrapping, "Create hierarchical_bootstrapping combinations.");
}
