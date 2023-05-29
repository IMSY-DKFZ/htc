// SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
// SPDX-License-Identifier: MIT

#include <torch/torch.h>
#include <vector>
#include "ParallelExecution.h"

struct MaskParameters {
	MaskParameters(int offset_left, int offset_top, int delta_horizontal, int delta_vertical)
		: offset_left(offset_left), offset_top(offset_top), delta_horizontal(delta_horizontal), delta_vertical(delta_vertical)
	{}

	int offset_left;
	int offset_top;
	int delta_horizontal;
	int delta_vertical;
};

using MaskParametersDict = std::map<std::string, std::map<std::string, int64_t>>;

class ColorcheckerAutomask {
public:
	ColorcheckerAutomask(const at::Tensor& rot_image, std::string cc_board, int square_size, int safety_margin, int square_dist_vertical, int square_dist_horizontal)
		: rot_image(rot_image), cc_board(std::move(cc_board)), square_size(square_size), safety_margin(safety_margin), square_dist_vertical(square_dist_vertical), square_dist_horizontal(square_dist_horizontal)
	{
		this->square_size_expanded = this->square_size + this->safety_margin;
		this->img_height = static_cast<int>(this->rot_image.size(0));
		this->img_width = static_cast<int>(this->rot_image.size(1));
	}

	MaskParametersDict find_mask() {
		int64_t best_param_index = -1;
		MaskParametersDict masks;
		int64_t n_rows;
		int64_t n_cols;

		if (this->cc_board == "cc_passport") {
			// Search for the left part on the left image side
			this->generate_parameters(/*offset_left_min_start=*/0, /*offset_left_stop=*/this->img_width / 2);
			n_rows = 6;
			n_cols = 4;
		}
		else {
			this->generate_parameters(/*offset_left_min_start=*/0);
			n_rows = 4;
			n_cols = 6;
		}

		auto score_values = at::empty({ static_cast<int64_t>(this->parameters.size()), n_rows * n_cols }, torch::kFloat32);

		ParallelExecution pe;
		pe.parallel_for(0, n_rows * n_cols - 1, [&](const size_t i) {
			int row = static_cast<int>(i / n_cols);
			int col = static_cast<int>(i % n_cols);
			auto score = this->compute_chip_scores(row, col);
			score_values.index_put_({ at::indexing::Slice(), static_cast<int>(i) }, score);
		});

		// The best score across all colorchecker chips
		best_param_index = score_values.sum(1).argmin().item().to<int64_t>();
		auto& best_param = this->parameters[best_param_index];

		masks["mask_0"] = {
			{"offset_left", best_param.offset_left + this->safety_margin / 2},
			{"offset_top", best_param.offset_top + this->safety_margin / 2},
			{"square_size", this->square_size},
			{"square_dist_horizontal", this->square_dist_horizontal + this->safety_margin + best_param.delta_horizontal},
			{"square_dist_vertical", this->square_dist_vertical + this->safety_margin + best_param.delta_vertical},
		};
		
		if (this->cc_board == "cc_passport") {
			// Search for the right part on the right image side
			this->generate_parameters(/*offset_left_min_start=*/this->img_width / 2, /*offset_left_stop=*/this->img_width);
			score_values = at::empty({ static_cast<int64_t>(this->parameters.size()), n_rows * n_cols }, torch::kFloat32);

			pe.parallel_for(0, n_rows * n_cols - 1, [&](const size_t i) {
				int row = static_cast<int>(i / n_cols);
				int col = static_cast<int>(i % n_cols);
				auto score = this->compute_chip_scores(row, col);
				score_values.index_put_({ at::indexing::Slice(), static_cast<int>(i) }, score);
			});

			// The best score across all colorchecker chips
			best_param_index = score_values.sum(1).argmin().item().to<int64_t>();
			best_param = this->parameters[best_param_index];

			masks["mask_1"] = {
				{"offset_left", best_param.offset_left + this->safety_margin / 2},
				{"offset_top", best_param.offset_top + this->safety_margin / 2},
				{"square_size", this->square_size},
				{"square_dist_horizontal", this->square_dist_horizontal + this->safety_margin + best_param.delta_horizontal},
				{"square_dist_vertical", this->square_dist_vertical + this->safety_margin + best_param.delta_vertical},
			};
		}

		return masks;
	}

	at::Tensor visualize_search_area() {
		at::Tensor area = at::zeros({ this->img_height, this->img_width }, torch::kInt32);
		auto area_a = area.accessor<int32_t, 2>();

		int64_t n_rows;
		int64_t n_cols;
		if (this->cc_board == "cc_passport") {
			// Search for the left part on the left image side
			this->generate_parameters(/*offset_left_min_start=*/0, /*offset_left_stop=*/this->img_width / 2);
			n_rows = 6;
			n_cols = 4;
		}
		else {
			this->generate_parameters(/*offset_left_min_start=*/0);
			n_rows = 4;
			n_cols = 6;
		}

		// We mark every position in the image where we place a square (the square goes to the right and to the bottom)
		auto fill_parameters = [&]() {
			for (int row = 0; row < n_rows; row++) {
				for (int col = 0; col < n_cols; col++) {
					for (size_t i = 0; i < this->parameters.size(); i++) {
						const MaskParameters& p = this->parameters[i];

						// Global coordinates
						int chip_row = p.offset_top + row * (this->square_size_expanded + this->square_dist_vertical + p.delta_vertical);
						int chip_col = p.offset_left + col * (this->square_size_expanded + this->square_dist_horizontal + p.delta_horizontal);

						area_a[chip_row][chip_col] += 1;
					}
				}
			}
		};
		fill_parameters();

		if (this->cc_board == "cc_passport") {
			this->generate_parameters(/*offset_left_min_start=*/this->img_width / 2, /*offset_left_stop=*/this->img_width);
			fill_parameters();
		}

		return area;
	}

private:
	at::Tensor compute_chip_scores(int row, int col) {
		// We need to know the maximal area we want to look at for this chip
		auto chip_row_min = this->offset_top_min + row * (this->square_size_expanded + this->square_dist_vertical + this->delta_vertical_min);
		auto chip_row_max = this->offset_top_max + row * (this->square_size_expanded + this->square_dist_vertical + this->delta_vertical_max);
		auto chip_col_min = this->offset_left_min + col * (this->square_size_expanded + this->square_dist_horizontal + this->delta_horizontal_min);
		auto chip_col_max = this->offset_left_max + col * (this->square_size_expanded + this->square_dist_horizontal + this->delta_horizontal_max);

		// Select the search area for the color chip
		assert(chip_row_min >= 0 && chip_row_max + this->square_size_expanded <= this->rot_image.size(0));
		assert(chip_col_min >= 0 && chip_col_max + this->square_size_expanded <= this->rot_image.size(1));
		auto chip_area = this->rot_image.index({
			at::indexing::Slice(chip_row_min, chip_row_max + this->square_size_expanded),
			at::indexing::Slice(chip_col_min, chip_col_max + this->square_size_expanded),
			at::indexing::Slice(),
		});

		// The pooling layers expect NCHW input
		chip_area = chip_area.permute({ 2, 0, 1 }).unsqueeze_(0);

		// Compute the variance for each square in the search area
		// Note: we do not compute the standard deviation since we are only interested in the minimum and hence don't need the sqrt
		auto chip_area_squared = chip_area.pow(2);
		auto variance = at::avg_pool2d(chip_area_squared, /*kernel_size=*/this->square_size_expanded, /*stride=*/1) - at::avg_pool2d(chip_area, /*kernel_size=*/this->square_size_expanded, /*stride=*/1).pow(2);

		// Emphasize consistency across worst spectral channel
		variance = std::get<0>(variance.squeeze_(0).max(0));
		auto variance_a = variance.accessor<float, 2>();

		// Now we just map the parameters to the corresponding variance values
		auto scores = at::empty({ static_cast<int64_t>(this->parameters.size()) }, torch::kFloat32);
		auto scores_a = scores.accessor<float, 1>();
		for (size_t i = 0; i < this->parameters.size(); i++) {
			const MaskParameters& p = this->parameters[i];

			// We compute the index of the chip first on the global level and then transfer it to the local coordinate system of the search area/variance (e.g. chip_row_min)
			int chip_row = p.offset_top + row * (this->square_size_expanded + this->square_dist_vertical + p.delta_vertical) - chip_row_min;
			int chip_col = p.offset_left + col * (this->square_size_expanded + this->square_dist_horizontal + p.delta_horizontal) - chip_col_min;
			assert(chip_row >= 0 && chip_row <= variance_a.size(0) - 1);
			assert(chip_col >= 0 && chip_col <= variance_a.size(1) - 1);
			scores_a[i] = variance_a[chip_row][chip_col];
		}

		return scores;
	}

	void generate_parameters(int offset_left_min_start, int offset_left_stop = -1) {
		this->parameters.clear();

		if (this->cc_board == "cc_passport") {
			// The passport colorchecker contains one part on the left and one part on the right which are optimized separately
			assert(offset_left_stop > 0);
			this->offset_left_max = offset_left_stop - 4 * this->square_size_expanded - 3 * (this->square_dist_horizontal + this->delta_horizontal_max);
			this->offset_top_max = this->img_height - 6 * this->square_size_expanded - 5 * (this->square_dist_vertical + this->delta_vertical_max);
		}
		else {
			// The offsets iterate over the complete image dimensions minus the colorchecker mask region
			this->offset_left_max = this->img_width - 6 * this->square_size_expanded - 5 * (this->square_dist_horizontal + this->delta_horizontal_max);
			this->offset_top_max = this->img_height - 4 * this->square_size_expanded - 3 * (this->square_dist_vertical + this->delta_vertical_max);
		}
		this->offset_left_min = offset_left_min_start;
		this->offset_top_min = 0;

		// Image margin which we do not consider
		this->offset_left_min += this->offset_ignore;
		this->offset_top_min += this->offset_ignore;
		this->offset_left_max -= this->offset_ignore;
		this->offset_top_max -= this->offset_ignore;

		assert(this->offset_left_min < this->offset_left_max);
		assert(this->offset_top_min < this->offset_top_max);

		// We generate a vector with all the different parameter combinations and then later search for the combination with the best score
		for (int delta_horizontal = this->delta_horizontal_min; delta_horizontal <= this->delta_vertical_max; delta_horizontal++) {
			for (int delta_vertical = this->delta_vertical_min; delta_vertical <= this->delta_horizontal_max; delta_vertical++) {
				for (int offset_left = this->offset_left_min; offset_left <= this->offset_left_max; offset_left++) {
					for (int offset_top = this->offset_top_min; offset_top <= this->offset_top_max; offset_top++) {
						this->parameters.emplace_back(offset_left, offset_top, delta_horizontal, delta_vertical);
					}
				}
			}
		}

		assert(this->parameters.size() > 0);
	}

	// User variables
	at::Tensor rot_image;
	std::string cc_board;
	int square_size;
	int safety_margin;
	int square_dist_vertical;
	int square_dist_horizontal;

	// Internal variables
	int img_height;
	int img_width;
	std::vector<MaskParameters> parameters;
	int offset_ignore = 0;
	int offset_left_min = 0;
	int offset_left_max = 0;
	int offset_top_min = 0;
	int offset_top_max = 0;
	int delta_vertical_min = -2;
	int delta_vertical_max = 2;
	int delta_horizontal_min = -2;
	int delta_horizontal_max = 2;
	int square_size_expanded;
};

MaskParametersDict colorchecker_automask(const at::Tensor& rot_image, const std::string& cc_board, int square_size, int safety_margin, int square_dist_vertical, int square_dist_horizontal) {
	return ColorcheckerAutomask(rot_image, cc_board, square_size, safety_margin, square_dist_vertical, square_dist_horizontal).find_mask();
}
at::Tensor colorchecker_automask_search_area(const at::Tensor& rot_image, const std::string& cc_board, int square_size, int safety_margin, int square_dist_vertical, int square_dist_horizontal) {
	return ColorcheckerAutomask(rot_image, cc_board, square_size, safety_margin, square_dist_vertical, square_dist_horizontal).visualize_search_area();
}
