// SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
// SPDX-License-Identifier: MIT

#include <torch/extension.h>
#include <random>
#include <unordered_map>

using Domain2Subjects = std::unordered_map<int64_t, std::vector<int64_t>>;
using Subject2Images = std::unordered_map<int64_t, std::vector<int64_t>>;
using Domain2Subjects2Images = std::unordered_map<int64_t, Subject2Images>;
using Label2Subjects2Images = std::unordered_map<int64_t, Subject2Images>;
using Label2Images = std::unordered_map<int64_t, std::vector<int64_t>>;
using Image2Labels = std::unordered_map<int64_t, std::vector<int64_t>>;

torch::Tensor hierarchical_bootstrapping(Domain2Subjects2Images& mapping,
                                         int n_subjects,
                                         int n_images,
                                         int n_bootstraps,
                                         unsigned int seed) {
    // Offers a good uniform distribution
    // (https://www.boost.org/doc/libs/1_61_0/doc/html/boost_random/reference.html#boost_random.reference.generators)
    std::mt19937 gen(seed);

    auto n_domains = mapping.size();
    auto bootstraps =
        torch::empty({n_bootstraps, static_cast<int64_t>(n_domains * n_subjects * n_images)}, torch::kInt64);
    auto bootstraps_a = bootstraps.accessor<int64_t, 2>();

    // Cache domain2subjects vector mapping for later use (we don't want to do this all over again inside the bootstrap
    // loop)
    Domain2Subjects domain2subjects;
    for (const auto& [domain_index, subject2images] : mapping) {
        domain2subjects[domain_index].reserve(subject2images.size());
        for (auto const& p : subject2images) {
            domain2subjects[domain_index].push_back(p.first);
        }
    }

    for (int b = 0; b < n_bootstraps; ++b) {
        int col = 0;

        for (auto& [domain_index, subject2images] : mapping) {
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

Domain2Subjects construct_domain_subjects_mapping(const Domain2Subjects2Images& domain_subjects_images_mapping) {
    Domain2Subjects domain2subjects;
    for (const auto& [domain_index, subject2images] : domain_subjects_images_mapping) {
        domain2subjects[domain_index].reserve(subject2images.size());
        for (auto const& p : subject2images) {
            domain2subjects[domain_index].push_back(p.first);
        }
    }

    return domain2subjects;
}

Label2Subjects2Images construct_label_subjects_images_mapping(
    const Domain2Subjects2Images& domain_subjects_images_mapping,
    const Label2Images& label_images_mapping) {
    Label2Subjects2Images label_subjects_images_mapping;
    for (const auto& [label, label_images] : label_images_mapping) {
        for (int64_t label_image : label_images) {
            // Search for the current image in the domain mapping
            bool found = false;
            for (const auto& [domain_index, subject2images] : domain_subjects_images_mapping) {
                for (const auto& [subject, images] : subject2images) {
                    if (std::find(images.begin(), images.end(), label_image) != images.end()) {
                        label_subjects_images_mapping[label][subject].push_back(label_image);
                        found = true;
                        break;
                    }
                }
                if (found) {
                    break;
                }
            }
        }
    }

    return label_subjects_images_mapping;
}

Image2Labels construct_image_labels_mapping(const Label2Images& label_images_mapping) {
    Image2Labels image_labels_mapping;
    for (const auto& [label, images] : label_images_mapping) {
        for (int64_t image : images) {
            image_labels_mapping[image].push_back(label);
        }
    }

    return image_labels_mapping;
}

torch::Tensor hierarchical_bootstrapping_labels(Domain2Subjects2Images& domain_subjects_images_mapping,
                                                Label2Images& label_images_mapping,
                                                int n_labels,
                                                int n_bootstraps,
                                                bool oversampling,
                                                unsigned int seed) {
    // Offers a good uniform distribution
    // (https://www.boost.org/doc/libs/1_61_0/doc/html/boost_random/reference.html#boost_random.reference.generators)
    std::mt19937 gen(seed);

    auto n_domains = domain_subjects_images_mapping.size();
    auto bootstraps = torch::empty({n_bootstraps, static_cast<int64_t>(n_domains * n_labels)}, torch::kInt64);
    auto bootstraps_a = bootstraps.accessor<int64_t, 2>();

    // Cache common mappings for later use (we don't want to do this all over again inside the bootstrap loop)
    Domain2Subjects domain_subjects_mapping = construct_domain_subjects_mapping(domain_subjects_images_mapping);
    Label2Subjects2Images label_subjects_images_mapping =
        construct_label_subjects_images_mapping(domain_subjects_images_mapping, label_images_mapping);
    Image2Labels image_labels_mapping = construct_image_labels_mapping(label_images_mapping);

    // List of possible labels
    std::vector<int64_t> labels;
    labels.reserve(label_images_mapping.size());
    for (auto& item : label_images_mapping) {
        labels.push_back(item.first);
    }
    std::uniform_int_distribution<> random_label(0, labels.size() - 1);

    // Keep track of how many times each label has been selected
    std::unordered_map<int64_t, int64_t> label_counts;
    for (int64_t label : labels) {
        label_counts[label] = 0;
    }

    for (int b = 0; b < n_bootstraps; ++b) {
        int col = 0;

        // For each label, we select per domain one subject and one image and repeat this process n_labels times
        while (col < bootstraps.size(1)) {
            // First select a label
            int64_t label;
            if (oversampling) {
                // Find all labels which have the current least occurrence (there might be multiple labels with the same
                // count)
                std::unordered_map<int64_t, std::vector<int64_t>> min_count_labels;
                int64_t min_count = std::numeric_limits<int64_t>::max();
                for (auto& [l, count] : label_counts) {
                    min_count_labels[count].push_back(l);
                    if (count < min_count) {
                        min_count = count;
                    }
                }

                auto& possible_labels = min_count_labels[min_count];
                if (possible_labels.size() > 1) {
                    // From the labels with the lowest count, select one randomly
                    std::uniform_int_distribution<> random_possible_label(0, possible_labels.size() - 1);
                    label = possible_labels[random_possible_label(gen)];
                } else {
                    // If there is only one possible label, we do not need to select anything randomly
                    label = possible_labels[0];
                }
            } else {
                label = labels[random_label(gen)];
            }

            auto& label_subjects = label_subjects_images_mapping[label];

            for (auto& [domain_index, subject2images] : domain_subjects_images_mapping) {
                std::vector<int64_t>& subjects_domain = domain_subjects_mapping[domain_index];

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
                int64_t image = images[random_image(gen)];
                bootstraps_a[b][col++] = image;

                if (oversampling) {
                    // Update label counts for all the labels which appear in the selected image
                    for (int64_t label : image_labels_mapping[image]) {
                        label_counts[label]++;
                    }
                }
            }
        }
    }

    return bootstraps;
}
