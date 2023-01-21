// SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
// SPDX-License-Identifier: MIT

#include <torch/extension.h>
#include <random>

using Cam2Subjects = std::unordered_map<int64_t, std::vector<int64_t>>;
using Subject2Images = std::unordered_map<int64_t, std::vector<int64_t>>;
using Cam2Subjects2Images = std::unordered_map<int64_t, Subject2Images>;

torch::Tensor hierarchical_bootstrapping(Cam2Subjects2Images mapping, int n_subjects, int n_images, int n_bootstraps, unsigned int seed) {
    std::mt19937 gen(seed); // Offers a good uniform distribution (https://www.boost.org/doc/libs/1_61_0/doc/html/boost_random/reference.html#boost_random.reference.generators)
    
    auto n_cams = mapping.size();
    auto bootstraps = torch::empty({n_bootstraps, static_cast<int64_t>(n_cams * n_subjects * n_images)}, torch::kInt64);
    auto bootstraps_a = bootstraps.accessor<int64_t, 2>();
    
    // Cache cam_to_subjects vector mapping for later use (we don't want to do this all over again inside the bootstrap loop)
    Cam2Subjects cam_to_subjects;
    for (const auto &[camera_index, subject_to_images]: mapping) {
        cam_to_subjects[camera_index].reserve(subject_to_images.size());
        for (auto const& p: subject_to_images) {
            cam_to_subjects[camera_index].push_back(p.first);
        }
    }
    
    for (int b = 0; b < n_bootstraps; ++b) {
        int col = 0;

        for (auto &[camera_index, subject_to_images]: mapping) {
            std::vector<int64_t>& subjects = cam_to_subjects[camera_index];
            
            std::uniform_int_distribution<> random_subject(0, subjects.size() - 1);
            
            for (int subject_index = 0; subject_index < n_subjects; ++subject_index) {
                auto& subject = subjects[random_subject(gen)];
                auto& images = subject_to_images[subject];
                std::uniform_int_distribution<> random_image(0, images.size() - 1);
                
                for (int image_index = 0; image_index < n_images; ++image_index) {
                    bootstraps_a[b][col++] = images[random_image(gen)];
                }
            }
        }
    }
    
    return bootstraps;
}
