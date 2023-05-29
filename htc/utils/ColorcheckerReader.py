# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch
from PIL import Image

from htc.cpp import colorchecker_automask, colorchecker_automask_search_area
from htc.tivita.DataPath import DataPath
from htc.utils.import_extra import requires_extra
from htc.utils.LabelMapping import LabelMapping

try:
    from deskew import determine_skew
    from skimage.color import rgb2gray
    from skimage.transform import rotate

    _missing_library = ""
except ImportError:
    _missing_library = "deskew and skimage"


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
    def __init__(
        self,
        img_dir: DataPath,
        cc_board: str,
        flipped: bool = False,
        rot_angle: float = None,
        square_size: int = None,
        safety_margin: int = None,
        square_dist_horizontal: int = None,
        square_dist_vertical: int = None,
    ):
        """
        The purpose of this class is to generate annotation masks for hyperspectral colorchecker images.

        There is two annotation options:
        - `ColorcheckerReader.create_automask()`: Automatically generates an annotation mask.
        - `ColorcheckerReader.create_mask()`: Generates a custom annotation mask.

        >>> from htc.settings import settings
        >>> from htc.utils.ColorcheckerReader import ColorcheckerReader
        >>> from htc.tivita.DataPath import DataPath
        >>> cc_board = "cc_classic"
        >>> img_dir = settings.data_dirs.studies / "2022_12_25_colorchecker_MIC1_TivitaHalogen/2022_12_25_20_45_29"
        >>> img_dir.exists()
        True
        >>> img_dir = DataPath(img_dir)
        >>> cc_reader = ColorcheckerReader(img_dir, cc_board)
        >>> automask = cc_reader.create_automask()
        >>> custom_mask_params = dict(square_size=70, square_dist_horizontal=34, square_dist_vertical=30, offset_top=60, offset_left=24)
        >>> custom_mask = cc_reader.create_mask({"mask_0": custom_mask_params})

        Args:
            img_dir: DataPath to the image directory (timestamp folder).
            cc_board: String describing the type of colorchecker board which is either the[colorchecker classic](https://www.xrite.com/de/categories/calibration-profiling/colorchecker-classic), referred to as "cc_classic" or a combination of the [colorchecker classic mini](https://www.xrite.com/categories/calibration-profiling/colorchecker-classic-family/colorchecker-classic-mini) and the video color chips of the [colorchecker passport video](https://www.xrite.com/categories/calibration-profiling/colorchecker-passport-video), referred to as "cc_passport".
            rot_angle: The rotation of the colorchecker board is corrected. The rotation angle is automatically determined by default, but if the determined rotation angle is not satisfying, a custom rotation angle rot_angle (in degrees) can be input.
            flipped: If the colorchecker board is upside-down, set flipped to True. The orientation of the mask will then be corrected.
            square_size: The size of the squares of the colorchecker mask. Provide a custom value if the default is not satisfying.
            safety_margin: The safety margin is the number of pixels added to the square size. TProvide a custom value if the default is not satisfying.
            square_dist_horizontal: The horizontal distance between the squares of the colorchecker mask. Provide a custom value if the default is not satisfying.
            square_dist_vertical: The vertical distance between the squares of the colorchecker mask. Provide a custom value if the default is not satisfying.
        """
        self.img_dir = img_dir
        rgb_img = self.img_dir.read_rgb_reconstructed()

        if rot_angle is None:
            self.rot_angle = determine_skew(rgb2gray(rgb_img))
        else:
            self.rot_angle = rot_angle

        assert (
            self.rot_angle is not None
        ), "Rotation angle could not be determined as None was returned, doublechecking of the HSI cube is needed!"
        assert (
            -30 <= self.rot_angle <= 30
        ), f"Rotation angle of {self.rot_angle} is not applied, doublechecking is needed!"

        if flipped:
            self.rot_angle = 180 + self.rot_angle

        self.rot_rgb = torch.from_numpy(rotate(rgb_img, self.rot_angle, resize=False, mode="reflect")).float()
        self.img_height, self.img_width = self.rot_rgb.shape[0:2]

        assert cc_board in [
            "cc_passport",
            "cc_classic",
        ], f"cc_board should be either cc_passport or cc_classic, but {cc_board} was given"
        self.cc_board = cc_board

        if self.cc_board == "cc_passport":
            self.square_size = 26 if square_size is None else square_size
            self.safety_margin = 12 if safety_margin is None else safety_margin
            self.square_dist_horizontal = 36 if square_dist_horizontal is None else square_dist_horizontal
            self.square_dist_vertical = 35 if square_dist_vertical is None else square_dist_vertical

        if self.cc_board == "cc_classic":
            self.square_size = 64 if square_size is None else square_size
            self.safety_margin = 15 if safety_margin is None else safety_margin
            self.square_dist_horizontal = 40 if square_dist_horizontal is None else square_dist_horizontal
            self.square_dist_vertical = 36 if square_dist_vertical is None else square_dist_vertical

        # The safety margin ensures that a solution more close to the center of the color chips is found
        self.square_dist_horizontal -= self.safety_margin
        self.square_dist_vertical -= self.safety_margin

        self.mask_params = {}

    def create_mask(
        self,
        mask_params: dict[str, dict[str, int]],
    ) -> torch.Tensor:
        """
        Generates an annotation mask from a given set of mask parameters. This is for example stored in `self.mas_params` after running `create_automask()`.

        Args:
            mask_params: Specifies the 5 mask parameters `offset_left` and `offset_top` (x- and y-coordinate of the origin of the colorchecker mask),  `square_size` (edge length of the square annotation for an individual color chip), `square_dist_horizontal` and`square_dist_vertical` (distance between neighboring square annotations along the x- and y-axis) (cf. htc/utils/colorchecker_mask_sketch.svg) for each mask, e.g. `{"mask_0": {"offset_left": 1}}`. There should be only "mask_0" for the colorchecker classic and "mask_0" and "mask_1" for the colorchecker passport.

        Returns: The custom annotation mask of the image.
        """

        def mask_from_params(params: dict[str, int]) -> torch.Tensor:
            mask = torch.zeros(self.img_height, self.img_width, dtype=torch.int32)

            if self.cc_board == "cc_passport":
                for row in range(4):
                    for col in range(6):
                        idx = row * 6 + (5 - col) + 1
                        left_min = params["offset_left"] + row * (
                            params["square_size"] + params["square_dist_horizontal"]
                        )
                        top_min = params["offset_top"] + col * (params["square_size"] + params["square_dist_vertical"])
                        mask[top_min : top_min + params["square_size"], left_min : left_min + params["square_size"]] = (
                            idx
                        )

            else:
                for row in range(4):
                    for col in range(6):
                        idx = row * 6 + col + 1
                        top_min = params["offset_top"] + row * (params["square_size"] + params["square_dist_vertical"])
                        left_min = params["offset_left"] + col * (
                            params["square_size"] + params["square_dist_horizontal"]
                        )

                        mask[top_min : top_min + params["square_size"], left_min : left_min + params["square_size"]] = (
                            idx
                        )

            if self.rot_angle is not None:
                mask = rotate(mask.numpy(), -self.rot_angle, resize=False, mode="constant", cval=0, order=0)
                mask = torch.from_numpy(mask)

            return mask

        if self.cc_board == "cc_passport":
            assert len(mask_params) == 2, f"Two masks are supported for cc_passport, but {len(mask_params)} were given"
            left_mask = mask_from_params(mask_params["mask_0"])
            right_mask = mask_from_params(mask_params["mask_1"])
            right_mask[right_mask > 0] = right_mask[right_mask > 0] + 24

            return left_mask + right_mask
        else:
            assert (
                len(mask_params) == 1
            ), f"Only one mask is supported for cc_classic, but {len(mask_params)} were given"
            return mask_from_params(mask_params["mask_0"])

    def create_automask(self) -> torch.Tensor:
        """
        Automatically create an annotation mask for the colorchecker image. Only works for "regular" images without distortions (viewed from the top) and only if all colorchecker chips are visible.

        Returns: Annotation mask of the image which is optimal according to the deviation score. The estimated paramters are stored in this class in the `self.mas_params` attribute.
        """
        self.mask_params = colorchecker_automask(
            self.rot_rgb,
            cc_board=self.cc_board,
            square_size=self.square_size,
            safety_margin=self.safety_margin,
            square_dist_horizontal=self.square_dist_horizontal,
            square_dist_vertical=self.square_dist_vertical,
        )
        return self.create_mask(self.mask_params)

    def automask_search_area(self) -> torch.Tensor:
        """
        Returns the search area for the automatic mask generation. This is the area where squares for the color chips are placed during optimization.

        Returns: A count map denoting the number of times each pixel was selected as the origin (top left corner) of a color chip area.
        """
        return colorchecker_automask_search_area(
            self.rot_rgb,
            cc_board=self.cc_board,
            square_size=self.square_size,
            safety_margin=self.safety_margin,
            square_dist_horizontal=self.square_dist_horizontal,
            square_dist_vertical=self.square_dist_vertical,
        )

    def save_mask(self, mask: torch.Tensor) -> None:
        """
        Saves a given annotation mask.

        Args:
            mask: Annotation mask which was generated either automatically or custom.
        """
        savepath = self.img_dir() / "annotations" / f"{self.img_dir.timestamp}#squares#automask#{self.cc_board}.png"
        savepath.parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray(mask.numpy()).save(savepath, optimize=True)
