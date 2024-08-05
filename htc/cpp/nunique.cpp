// SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
// SPDX-License-Identifier: MIT

#include <torch/extension.h>
#include <ATen/native/ReduceOpsUtils.h>

torch::Tensor nunique(const torch::Tensor& in, int64_t dim) {
    // make_reduction resizes the output tensor
    torch::Tensor out = torch::empty({0}, torch::kInt64);  // We store the counts in a long tensor
    auto iter = torch::native::make_reduction(/*name=*/"nunique", out, in, dim, /*keepdim=*/false, /*in_dtype=*/in.scalar_type(), /*out_dtype=*/torch::kInt64);

    // AT_DISPATCH_ALL_TYPES automatically templates all versions of this function and chooses the right one based on the tensor type (https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h)
    // scalar_t is the type of the tensor
    // See also this example: https://github.com/pytorch/pytorch/blob/96c549d1c62462a74a41c99552e8f8a37aaa0793/aten/src/ATen/native/Sorting.cpp#L318
    AT_DISPATCH_ALL_TYPES(in.scalar_type(), /*name=*/"nunique", [&] {
        // Based on this example: https://labs.quansight.org/blog/2021/04/pytorch-tensoriterator-internals-update/index.html
        auto loop = [](char** data, const int64_t* strides, int64_t n) {
            char* out_data = data[0];
            char* in_data = data[1];

            // We use a simple set to find the unique elements in the reduce dimension (it turns out to be quite fast: https://stackoverflow.com/a/24477023/2762258)
            std::unordered_set<scalar_t> values;

            const char* end = in_data + n * strides[1];
            char* p = in_data;
            while (p != end) {
                values.insert(*reinterpret_cast<scalar_t*>(p));
                p += strides[1];
            }

            *reinterpret_cast<int64_t*>(out_data) = values.size();
        };

        // Unfortunately, we need to execute the loop in serial because the unordered_set is not thread safe
        iter.serial_for_each(loop, {0, iter.numel()});
    });

    return out;
}
