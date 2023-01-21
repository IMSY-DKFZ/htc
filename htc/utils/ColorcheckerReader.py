# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import itertools

import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import rotate

from htc.tivita.DataPath import DataPath
from htc.utils.import_extra import requires_extra
from htc.utils.LabelMapping import LabelMapping

try:
    from deskew import determine_skew

    _missing_library = ""
except ImportError:
    _missing_library = "deskew"


class ColorcheckerReader:
    label_colors_classic = {
        "dark_skin": "#735244",
        "light_skin": "#c29682",
        "blue_sky": "#627a9d",
        "foliage": "#576c43",
        "blue_flower": "#8580b1",
        "bluish_green": "#67bdaa",
        "orange": "#d67e2c",
        "purplish_blue": "#505ba6",
        "moderate_red": "#c15a63",
        "purple": "#5e3c6c",
        "yellow_green": "#9dbc40",
        "orange_yellow": "#e0a32e",
        "blue": "#383d96",
        "green": "#469449",
        "red": "#af363c",
        "yellow": "#e7c71f",
        "magenta": "#bb5695",
        "cyan": "#0885a1",
        "white": "#f3f3f2",
        "neutral_8": "#c8c8c8",
        "neutral_65": "#a0a0a0",
        "neutral_5": "#7a7a7a",
        "neutral_35": "#555555",
        "black": "#343434",
    }
    label_colors_video = {
        "confetti": "#ece85c",
        "froly": "#f18171",
        "lavender_magenta": "#f08eed",
        "cornflower_blue": "#747df1",
        "blizzard_blue": "#9eebed",
        "pastel_green": "#90e168",
        "tobacco_brown": "#6d5840",
        "antique_brass": "#c59064",
        "gold_sand": "#e8c49a",
        "tumbleweed": "#deb282",
        "pancho": "#ecccab",
        "desert_sand": "#edceb0",
        "merlin": "#45433e",
        "chicago": "#605f5a",
        "lemon_grass": "#98968b",
        "dawn": "#aba89c",
        "ash": "#c4c0b5",
        "westar": "#dfdbd1",
        "iridium": "#3b3a35",
        "dune": "#363531",
        "davy_grey": "#575853",
        "moon_mist": "#dcd8cc",
        "timberwolf": "#e0dcd0",
        "white_rock": "#eee8dd",
    }
    label_colors_passport = label_colors_classic | label_colors_video

    label_mapping_classic = {
        "background": 0,
        "dark_skin": 1,
        "light_skin": 2,
        "blue_sky": 3,
        "foliage": 4,
        "blue_flower": 5,
        "bluish_green": 6,
        "orange": 7,
        "purplish_blue": 8,
        "moderate_red": 9,
        "purple": 10,
        "yellow_green": 11,
        "orange_yellow": 12,
        "blue": 13,
        "green": 14,
        "red": 15,
        "yellow": 16,
        "magenta": 17,
        "cyan": 18,
        "white": 19,
        "neutral_8": 20,
        "neutral_65": 21,
        "neutral_5": 22,
        "neutral_35": 23,
        "black": 24,
    }
    label_mapping_passport = label_mapping_classic | {
        "confetti": 25,
        "froly": 26,
        "lavender_magenta": 27,
        "cornflower_blue": 28,
        "blizzard_blue": 29,
        "pastel_green": 30,
        "tobacco_brown": 31,
        "antique_brass": 32,
        "gold_sand": 33,
        "tumbleweed": 34,
        "pancho": 35,
        "desert_sand": 36,
        "merlin": 37,
        "chicago": 38,
        "lemon_grass": 39,
        "dawn": 40,
        "ash": 41,
        "westar": 42,
        "iridium": 43,
        "dune": 44,
        "davy_grey": 45,
        "moon_mist": 46,
        "timberwolf": 47,
        "white_rock": 48,
    }
    label_mapping_classic = LabelMapping(
        label_mapping_classic,
        zero_is_invalid=True,
        label_colors=label_colors_classic,
    )
    label_mapping_passport = LabelMapping(
        label_mapping_passport,
        zero_is_invalid=True,
        label_colors=label_colors_passport,
    )

    @requires_extra(_missing_library)
    def __init__(self, img_dir: DataPath, cc_board: str, rot_angle: float = None):
        """
        The purpose of this class is to generate annotation masks for hyperspectral colorchecker images.

        There is two annotation options:
        - `ColorcheckerReader.create_automask()`: Automatically generates an annotation mask.
        - `ColorcheckerReader.create_mask()`: Generates a custom annotation mask.

        >>> from htc.settings import settings
        >>> from htc.utils.ColorcheckerReader import ColorcheckerReader
        >>> from htc.tivita.DataPath import DataPath
        >>> cc_board = "cc_classic"
        >>> img_dir = settings.data_dirs.studies / "2021_03_31_yellow_filter_colorchecker/0202-00118/2022_12_25_colorchecker_MIC1/2022_12_25_20_45_29"
        >>> img_dir.exists()
        True
        >>> img_dir = DataPath(img_dir)
        >>> cc_reader = ColorcheckerReader(img_dir, cc_board)
        >>> automask = cc_reader.create_automask()
        >>> custom_mask_params = dict(square_size=70, square_dist_horizontal=34, square_dist_vertical=30, offset_top=60, offset_left=24)
        >>> custom_mask = cc_reader.create_mask(custom_mask_params)

        Args:
            img_dir: DataPath to the image directory (timestamp folder).
            cc_board: String describing the type of colorchecker board which is either the[colorchecker classic](https://www.xrite.com/de/categories/calibration-profiling/colorchecker-classic), referred to as "cc_classic" or a combination of the [colorchecker classic mini](https://www.xrite.com/categories/calibration-profiling/colorchecker-classic-family/colorchecker-classic-mini) and the video color chips of the [colorchecker passport video](https://www.xrite.com/categories/calibration-profiling/colorchecker-passport-video), referred to as "cc_passport".
            rot_angle: The rotation of the colorchecker board is corrected. The rotation angle is automatically determined by default, but if the determined rotation angle is not satisfying, a custom rotation angle rot_angle (in degrees) can be input.
        """
        self.img_dir = img_dir
        self.cube = torch.from_numpy(self.img_dir.read_cube())
        self.img_height, self.img_width = self.cube.shape[0:2]

        if rot_angle is None:
            rgb_img = self.img_dir.read_rgb_reconstructed()
            self.rot_angle = determine_skew(rgb2gray(rgb_img))
        else:
            self.rot_angle = rot_angle

        assert (
            -30 <= self.rot_angle <= 30
        ), f"Rotation angle of {self.rot_angle} is not applied, doublechecking is needed!"
        self.rot_cube = torch.from_numpy(rotate(self.cube, self.rot_angle, resize=False, mode="reflect"))
        self.rot_rgb = torch.from_numpy(
            rotate(self.img_dir.read_rgb_reconstructed(), self.rot_angle, resize=False, mode="reflect")
        )

        assert cc_board in [
            "cc_passport",
            "cc_classic",
        ], f"cc_board should be either cc_passport or cc_classic, but {cc_board} was given"
        self.cc_board = cc_board

        if self.cc_board == "cc_passport":
            self.square_size = 26
            self.safety_margin = 12  # used to dilate the square size during mask optimization
            self.square_dist_horizontal = 36 - self.safety_margin
            self.square_dist_vertical = 35 - self.safety_margin

        if self.cc_board == "cc_classic":
            self.square_size = 64
            self.safety_margin = 12
            self.square_dist_horizontal = 42 - self.safety_margin
            self.square_dist_vertical = 38 - self.safety_margin

        self.mask_params = {}

    def create_mask(
        self,
        mask_params: dict[str, int],
    ) -> torch.Tensor:
        """
        Generates a custom annotation mask from a given set of mask parameters.

        Args:
            mask_params: Specifies the 5 mask parameters `offset_left` and `offset_top` (x- and y-coordinate of the origin of the colorchecker mask),  `square_size` (edge length of the square annotation for an individual color chip), `square_dist_horizontal` and`square_dist_vertical` (distance between neighboring square annotations along the x- and y-axis) (cf. htc/utils/colorchecker_mask_sketch.svg).

        Returns: The custom annotation mask of the image.
        """
        spatial_shape = self.img_dir.dataset_settings["spatial_shape"]
        mask = torch.zeros(spatial_shape, dtype=torch.int32)

        if self.cc_board == "cc_passport":
            for row in range(4):
                for col in range(6):
                    idx = row * 6 + (5 - col) + 1
                    left_min = mask_params["offset_left"] + row * (
                        mask_params["square_size"] + mask_params["square_dist_horizontal"]
                    )
                    top_min = mask_params["offset_top"] + col * (
                        mask_params["square_size"] + mask_params["square_dist_vertical"]
                    )
                    mask[
                        top_min : top_min + mask_params["square_size"], left_min : left_min + mask_params["square_size"]
                    ] = idx

        else:
            for row in range(4):
                for col in range(6):
                    idx = row * 6 + col + 1
                    top_min = mask_params["offset_top"] + row * (
                        mask_params["square_size"] + mask_params["square_dist_vertical"]
                    )
                    left_min = mask_params["offset_left"] + col * (
                        mask_params["square_size"] + mask_params["square_dist_horizontal"]
                    )

                    mask[
                        top_min : top_min + mask_params["square_size"], left_min : left_min + mask_params["square_size"]
                    ] = idx

        if self.rot_angle is not None:
            mask = rotate(mask.numpy(), -self.rot_angle, resize=False, mode="constant", cval=0, order=0)
            mask = torch.from_numpy(mask)

        return mask

    def compute_mask_score(
        self, offset_left: int, offset_top: int, delta_horizontal: int, delta_vertical: int
    ) -> float:
        """
        During automated annotation mask generation, this function computes a score for a given set of mask parameters. Across all mask parameter sets, the set with the smallest score will later be selected to generate an optimal automask. See htc.utils.colorchecker_mask_sketch.svg for a visualization of the mask parameters.

        Args:
            offset_left: x-coordinate of the origin of the colorchecker mask.
            offset_top: y-coordinate of the origin of the colorchecker mask.
            delta_horizontal: Deviation of the mask parameter square_dist_horizontal in pixels.
            delta_vertical: Deviation of the mask parameter square_dist_vertical in pixels.

        Returns: Score for the given set of mask parameters.
        """
        stds = torch.empty(24, self.rot_rgb.shape[-1], dtype=self.rot_rgb.dtype)
        square_size = self.square_size + self.safety_margin

        if self.cc_board == "cc_passport":
            for row in range(4):
                for col in range(6):
                    idx = row * 6 + col
                    left_min = offset_left + row * (square_size + self.square_dist_horizontal + delta_horizontal)
                    top_min = offset_top + col * (square_size + self.square_dist_vertical + delta_vertical)
                    spectra = self.rot_rgb[top_min : top_min + square_size, left_min : left_min + square_size, :]
                    assert spectra.shape == (square_size, square_size, 3), spectra.shape
                    stds[idx, :] = torch.std(spectra, dim=(0, 1))

        else:
            for row in range(4):
                for col in range(6):
                    idx = row * 6 + col
                    top_min = offset_top + row * (square_size + self.square_dist_vertical + delta_vertical)
                    left_min = offset_left + col * (square_size + self.square_dist_horizontal + delta_horizontal)
                    spectra = self.rot_rgb[top_min : top_min + square_size, left_min : left_min + square_size, :]
                    stds[idx, :] = torch.std(spectra, dim=(0, 1))

        assert not stds.isnan().any()
        return stds.max(dim=1).values.sum().numpy()  # emphasize consistency across worst spectral channel

    def automask_helper(
        self,
        offset_left_range: np.ndarray,
        offset_top_range: np.ndarray,
        deltas_horizontal_range: np.ndarray,
        deltas_vertical_range: np.ndarray,
    ) -> torch.Tensor:
        """
        During the automated annotation mask generation, the mask parameters will be varied in the given ranges to generate an optimal annotation mask. See htc.utils.colorchecker_mask_sketch.svg for a visualization of the mask parameters.

        Args:
            offset_left_range: Range in which x-coordinate of the origin of the colorchecker mask will be varied.
            offset_top_range: Range in which y-coordinate of the origin of the colorchecker mask will be varied.
            deltas_horizontal_range: Range in which deviation from `square_dist_horizontal` will be varied.
            deltas_vertical_range: Range in which deviation from `square_dist_vertical` will be varied.

        Returns: Annotation mask of the image which is optimal according to the score.
        """
        offsets_left = []
        offsets_top = []
        deltas_horizontal = []
        deltas_vertical = []
        for ol, ot, dh, dv in itertools.product(
            offset_left_range, offset_top_range, deltas_horizontal_range, deltas_vertical_range
        ):
            offsets_left.append(ol)
            offsets_top.append(ot)
            deltas_horizontal.append(dh)
            deltas_vertical.append(dv)

        # Note: using p_map here leads to multiprocessing issues when running from a Jupyter notebook
        overall_stds = []
        for ol, ot, dh, dv in zip(offsets_left, offsets_top, deltas_horizontal, deltas_vertical):
            overall_stds.append(self.compute_mask_score(ol, ot, dh, dv))

        opt_idx = np.argmin(overall_stds)
        mask_id = f"mask_{len(self.mask_params.keys())}"
        self.mask_params[mask_id] = {}
        self.mask_params[mask_id]["offset_left"] = offsets_left[opt_idx] + int(self.safety_margin / 2)
        self.mask_params[mask_id]["offset_top"] = offsets_top[opt_idx] + int(self.safety_margin / 2)
        self.mask_params[mask_id]["square_size"] = self.square_size
        self.mask_params[mask_id]["square_dist_horizontal"] = (
            self.square_dist_horizontal + self.safety_margin + deltas_horizontal[opt_idx]
        )
        self.mask_params[mask_id]["square_dist_vertical"] = (
            self.square_dist_vertical + self.safety_margin + deltas_vertical[opt_idx]
        )

        return self.create_mask(self.mask_params[mask_id])

    def create_automask(self) -> torch.Tensor:
        """
        Automated annotation mask generation.

        Returns: Annotation mask of the image which is optimal according to the score.
        """
        if self.cc_board == "cc_passport":
            deltas_horizontal = np.arange(-1, 1, 1)
            deltas_vertical = np.arange(-1, 2, 1)
            # optimize left part of mask
            # max offset top set to make sure the mask will not exceed the image boundaries
            ot_max = (
                self.img_height
                - 6 * (self.square_size + self.safety_margin)
                - 5 * (self.square_dist_vertical + np.max(deltas_vertical))
            )
            offsets_left = np.arange(0, 50, 2)
            offsets_top = np.arange(0, ot_max, 2)
            left_mask = self.automask_helper(offsets_left, offsets_top, deltas_horizontal, deltas_vertical)

            # optimize right part of mask
            # max offset left and top set to make sure the mask will not exceed the image boundaries
            ol_min = np.max(np.where(left_mask == 24)) + 100
            ol_max = (
                self.img_width
                - 4 * (self.square_size + self.safety_margin)
                - 3 * (self.square_dist_horizontal + np.max(deltas_horizontal))
            )
            assert ol_min < ol_max, "Right part of mask seems not to fit entirely into image frame"
            offsets_left = np.arange(ol_min, ol_max, 2)
            right_mask = self.automask_helper(offsets_left, offsets_top, deltas_horizontal, deltas_vertical)
            right_mask[right_mask > 0] = right_mask[right_mask > 0] + 24

            automask = left_mask + right_mask

        else:
            deltas_horizontal = np.arange(-2, 2, 2)
            deltas_vertical = np.arange(-2, 2, 2)
            # max offset left and top set to make sure the mask will not exceed the image boundaries
            ol_max = (
                self.img_width
                - 6 * (self.square_size + self.safety_margin)
                - 5 * (self.square_dist_horizontal + np.max(deltas_horizontal))
            )
            ot_max = (
                self.img_height
                - 4 * (self.square_size + self.safety_margin)
                - 3 * (self.square_dist_vertical + np.max(deltas_vertical))
            )
            offsets_left = np.arange(0, ol_max, 2)
            offsets_top = np.arange(0, ot_max, 2)
            automask = self.automask_helper(offsets_left, offsets_top, deltas_horizontal, deltas_vertical)

        return automask

    def save_mask(self, mask: torch.Tensor) -> None:
        """
        Saves a given annotation mask.

        Args:
            mask: Annotation mask which was generated either automatically or custom.
        """
        savepath = self.img_dir() / "annotations" / f"{self.img_dir.timestamp}#squares#automask#{self.cc_board}.png"
        savepath.parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray(mask.numpy()).save(savepath, optimize=True)
