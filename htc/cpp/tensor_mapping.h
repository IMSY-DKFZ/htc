// SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
// SPDX-License-Identifier: MIT

#include <torch/extension.h>

template<typename map_t>
torch::Tensor tensor_mapping(torch::Tensor& tensor, std::unordered_map<map_t, map_t>& mapping) {
    torch::TensorIteratorConfig iter_config;
    iter_config.add_input(tensor);
    auto iter = iter_config.build();

    AT_DISPATCH_ALL_TYPES(tensor.scalar_type(), /*name=*/"tensor_mapping", [&] {
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
            char* in_data = data[0];

            const char* end = in_data + n * strides[0];
            char* p = in_data;
            while (p != end) {
                auto* value = reinterpret_cast<scalar_t*>(p);
                if (mapping.contains(*value)) {
                    *value = mapping[*value];
                }

                p += strides[0];
            }
        };

        iter.for_each(loop);
    });

    return tensor;
}

// We need two functions for the mapping in Python since we need to call one of the functions depending on the type of the mapping (which is a dict in Python)
// Bot mappings have the largest type so that there are no type conversion issues (the tensor type can only be <= the mapping type and casting to a higher type is ok)
torch::Tensor tensor_mapping_integer(torch::Tensor& tensor, std::unordered_map<int64_t, int64_t>& mapping) {
    return tensor_mapping<int64_t>(tensor, mapping);
}

torch::Tensor tensor_mapping_floating(torch::Tensor& tensor, std::unordered_map<double, double>& mapping) {
    return tensor_mapping<double>(tensor, mapping);
}
