// SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <map>
#include <vector>

class KFoldCombinations {
   public:
    KFoldCombinations(const std::map<int, std::vector<int>>& subject_labels, int min_labels, int n_groups)
        : subject_labels(subject_labels),
          min_labels(min_labels),
          n_groups(n_groups) {}

    std::vector<std::vector<int>> find_groups(const std::vector<int>& subject_indices) {
        std::vector<int> last(this->n_groups * this->n_group_size);
        this->group(last, subject_indices, 0);

        return this->combinations;
    }

   private:
    void group(std::vector<int>& last, const std::vector<int>& input, int step) {
        if (step == this->n_groups - 1) {
            last[step * 3] = input[0];
            last[step * 3 + 1] = input[1];
            last[step * 3 + 2] = input[2];

            if (check_fold(last)) {
                this->combinations.push_back(last);
            }
        } else {
            last[step * 3] = input[0];
            for (size_t i = 1; i < input.size() - 1; i++) {
                last[step * 3 + 1] = input[i];
                for (size_t j = i + 1; j < input.size(); j++) {
                    last[step * 3 + 2] = input[j];

                    std::vector<int> input_new(input.size() - this->n_group_size);
                    std::remove_copy_if(input.begin(), input.end(), input_new.begin(), [&](int value) {
                        return value == input[0] || value == input[i] || value == input[j];
                    });

                    this->group(last, input_new, step + 1);
                }
            }
        }
    }

    bool check_fold(const std::vector<int>& fold) {
        constexpr int MAX_SIZE_LABELS = 256;  // We do not support label ids > 255
        int n_labels = 0;
        for (int g = 0; g < this->n_groups; g++) {
            std::array<int, MAX_SIZE_LABELS> unique_labels = {0};
            for (int p = 0; p < this->n_group_size; p++) {
                auto label_indices = fold[g * this->n_group_size + p];
                for (auto label : this->subject_labels.at(label_indices)) {
                    unique_labels[label]++;
                }
            }
            n_labels += std::count_if(unique_labels.begin(), unique_labels.end(), [](int count) { return count > 0; });
        }

        return n_labels >= this->min_labels;
    }

    std::map<int, std::vector<int>> subject_labels;
    int min_labels;
    int n_groups;
    int n_group_size = 3;
    std::vector<std::vector<int>> combinations;
};

std::vector<std::vector<int>> kfold_combinations(const std::vector<int>& subject_indices,
                                                 const std::map<int, std::vector<int>>& subject_labels,
                                                 int min_labels,
                                                 int n_groups = 5) {
    return KFoldCombinations(subject_labels, min_labels, n_groups).find_groups(subject_indices);
}
