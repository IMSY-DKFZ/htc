# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from typing import Union

import numpy as np
import torch

from htc.cpp import automatic_numpy_conversion
from htc.settings import settings
from htc.utils.import_extra import requires_extra

try:
    from skimage.segmentation import slic

    _missing_library_skimage = ""
except ImportError:
    _missing_library_skimage = "skimage"

try:
    import cv2
    from fast_slic.avx2 import SlicAvx2

    _missing_library_fast_slic = ""
except ImportError:
    _missing_library_fast_slic = "fast-slic and opencv-python"


class SLICWrapper:
    def __init__(
        self, n_segments: int, compactness: int, slic_zero: bool = True, sigma: int = 3, implementation: str = "skimage"
    ):
        """
        Wrapper for the fast_slic and skimage.slic functions. It ensures Gaussian blurring before superpixels are calculated.

        If you want to use the fast_slic implementation, you have to install the corresponding package first:
        ```bash
        pip install fast_slic
        ```
        Unfortunately, the [fast_slic package](https://github.com/Algy/fast-slic) is neither documented nor maintained anymore. At the moment (2023-04-14), it still works but there may be issues in the future which we cannot resolve. This is also the reason why we do not add `fast_slic` to the extra requirements so that we reduce the risk of breaking our test pipeline.

        Please also refer to the [documentation of skimage](https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic) for a detailed description of the parameters.

        Args:
            n_segments: The (approximate) number of superpixels in the output image.
            compactness: See skimage. The compactness is not the same between skimage and fast_slic. The latter usually requires higher values to achieve similar (less noisy) results.
            slic_zero: See skimage.
            sigma: See skimage.
            implementation: The implementation to be used for computing the superpixels (skimage or fast_slic).
        """
        self.n_segments = n_segments
        self.compactness = compactness
        self.slic_zero = slic_zero
        self.sigma = sigma
        self.implementation = implementation

        if self.implementation == "fast_slic" and self.compactness < 40:
            settings.log.warning(
                "It is recommended to use a compactness value of 40 or higher when using fast_slic for superpixels"
                " computations, to ensure less noisy superpixels."
            )

        # Truncate the filter at this many standard deviations. This is the default parameter in the skimage.slic gaussian filter
        self.truncate = 4.0

        # Formula for ksize parameter: ksize = 2 * (sigma * truncate) + 1
        # where sigma and truncate are parameters of skimage. With skimage default parameters of sigma = 3, truncate = 4, ksize = 25
        self.ksize = int(2 * (self.sigma * self.truncate) + 1)

    @automatic_numpy_conversion
    def apply_slic(self, img: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Apply the slic function on an image after applying Gaussian blurring

        Args:
            img: A torch tensor containing the image (HWC).

        Returns: The superpixel segmentation mask (HW) with a unique label for each superpixel.
        """
        assert img.ndim == 3, "The image must have three dimensions: (H, W, C)"

        if self.implementation == "skimage":
            spxs = self._apply_skimage_slic(img)
        elif self.implementation == "fast_slic":
            assert img.shape[-1] == 3, "fast_slic can only handle RGB images with shape: (H, W, 3)"
            spxs = self._apply_fast_slic(img)
        else:
            raise ValueError(
                f"Unknown implementation choice: {self.implementation}, Possible choices: {{skimage, fast_slic}}"
            )

        spxs = torch.from_numpy(spxs)
        spxs = spxs.type(torch.int64)

        return spxs

    @requires_extra(_missing_library_skimage)
    def _apply_skimage_slic(self, img: torch.Tensor):
        return slic(
            img.float().numpy(),
            start_label=0,
            n_segments=self.n_segments,
            compactness=self.compactness,
            slic_zero=self.slic_zero,
            sigma=self.sigma,
        )

    @requires_extra(_missing_library_fast_slic)
    def _apply_fast_slic(self, img: torch.Tensor):
        img = (img * 255).type(torch.uint8)
        img = cv2.GaussianBlur(img.cpu().numpy(), (self.ksize, self.ksize), sigmaX=self.sigma)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        fast_slic = SlicAvx2(num_components=self.n_segments, compactness=self.compactness)
        return fast_slic.iterate(img)
