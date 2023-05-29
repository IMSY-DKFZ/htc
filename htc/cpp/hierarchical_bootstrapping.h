// SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
// SPDX-License-Identifier: MIT

#include <torch/extension.h>
#include <random>
#include <unordered_map>

using Domain2Subjects = std::unordered_map<int64_t, std::vector<int64_t>>;
using Subject2Images = std::unordered_map<int64_t, std::vector<int64_t>>;
using Domain2Subjects2Images = std::unordered_map<int64_t, Subject2Images>;
using Label2Subjects2Images = std::unordered_map<int64_t, Subject2Images>;

torch::Tensor hierarchical_bootstrapping(Domain2Subjects2Images& mapping, int n_subjects, int n_images, int n_bootstraps, unsigned int seed) {
    std::mt19937 gen(seed); // Offers a good uniform distribution (https://www.boost.org/doc/libs/1_61_0/doc/html/boost_random/reference.html#boost_random.reference.generators)
    
    auto n_domains = mapping.size();
    auto bootstraps = torch::empty({n_bootstraps, static_cast<int64_t>(n_domains * n_subjects * n_images)}, torch::kInt64);
    auto bootstraps_a = bootstraps.accessor<int64_t, 2>();
    
    // Cache domain2subjects vector mapping for later use (we don't want to do this all over again inside the bootstrap loop)
    Domain2Subjects domain2subjects;
    for (const auto &[domain_index, subject2images]: mapping) {
        domain2subjects[domain_index].reserve(subject2images.size());
        for (auto const& p: subject2images) {
            domain2subjects[domain_index].push_back(p.first);
        }
    }
    
    for (int b = 0; b < n_bootstraps; ++b) {
        int col = 0;

        for (auto &[domain_index, subject2images]: mapping) {
            std::vector<int64_t>& subjects = domain2subjects[domain_index];
            
            std::uniform_int_distribution<> random_subject(0, subjects.size() - 1);
            
            for (int subject_index = 0; subject_index < n_subjects; ++subject_index) {
                auto& subject = subjects[random_subject(gen)];
                auto& images = subject2images[subject];
                std::uniform_int_distribution<> random_image(0, images.size() - 1);
                
                for (int image_index = 0; image_index < n_images; ++image_index) {
                    bootstraps_a[b][col++] = images[random_image(gen)];
                }
            }
        }
    }
    
    return bootstraps;
}

torch::Tensor hierarchical_bootstrapping_labels(Domain2Subjects2Images& domain_mapping, Label2Subjects2Images& label_mapping, int n_labels, int n_bootstraps, unsigned int seed) {
    std::mt19937 gen(seed); // Offers a good uniform distribution (https://www.boost.org/doc/libs/1_61_0/doc/html/boost_random/reference.html#boost_random.reference.generators)

    auto n_domains = domain_mapping.size();
    auto bootstraps = torch::empty({ n_bootstraps, static_cast<int64_t>(n_domains * n_labels) }, torch::kInt64);
    auto bootstraps_a = bootstraps.accessor<int64_t, 2>();

    // Cache domain2subjects vector mapping for later use (we don't want to do this all over again inside the bootstrap loop)
    Domain2Subjects domain2subjects;
    for (const auto& [domain_index, subject2images] : domain_mapping) {
        domain2subjects[domain_index].reserve(subject2images.size());
        for (auto const& p : subject2images) {
            domain2subjects[domain_index].push_back(p.first);
        }
    }

    // List of possible labels
    std::vector<int64_t> labels;
    labels.reserve(label_mapping.size());
    for (auto& item : label_mapping) {
        labels.push_back(item.first);
    }
    std::uniform_int_distribution<> random_label(0, labels.size() - 1);

    for (int b = 0; b < n_bootstraps; ++b) {
        int col = 0;

        // For each label, we select per domain one subject and one image and repeat this process n_labels times
        while (col < bootstraps.size(1)) {
            auto label = labels[random_label(gen)];
            auto& label_subjects = label_mapping[label];

            for (auto& [domain_index, subject2images] : domain_mapping) {
                std::vector<int64_t>& subjects_domain = domain2subjects[domain_index];

                // Select the subjects which have images of the current label
                std::vector<int64_t> subjects;
                subjects.reserve(subjects_domain.size());
                std::copy_if(subjects_domain.begin(), subjects_domain.end(), std::back_inserter(subjects), [&](auto s) {
                    return label_subjects.contains(s);
                });

                // Select random subject
                std::uniform_int_distribution<> random_subject(0, subjects.size() - 1);
                auto subject = subjects[random_subject(gen)];

                // Select random image
                auto& images = label_subjects[subject];
                std::uniform_int_distribution<> random_image(0, images.size() - 1);
                bootstraps_a[b][col++] = images[random_image(gen)];
            }
        }
    }

    return bootstraps;
}
