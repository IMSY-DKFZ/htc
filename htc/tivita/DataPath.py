# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import importlib
import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Union

import numpy as np
import pandas as pd
from PIL import Image
from typing_extensions import Self

from htc.settings import settings
from htc.tivita.DatasetSettings import DatasetSettings
from htc.utils.blosc_compression import decompress_file
from htc.utils.Config import Config
from htc.utils.JSONSchemaMeta import JSONSchemaMeta
from htc.utils.LabelMapping import LabelMapping

if TYPE_CHECKING:
    from htc.cameras.calibration.CalibrationSwap import CalibrationFiles


class DataPath:
    _local_meta_cache = None
    _network_meta_cache = None
    _meta_labels_cache = {}
    _data_paths_cache = {}

    def __init__(
        self,
        image_dir: Union[str, Path, None],
        data_dir: Union[str, Path] = None,
        intermediates_dir: Union[str, Path] = None,
        dataset_settings: DatasetSettings = None,
        annotation_name_default: Union[str, list[str]] = None,
    ):
        """
        Base class for working with Tivita data files with the main design goal to easily access all the individual files (e.g. HSI cube or RGB image) and extract data path attributes (e.g. subject_name).

        This class works with a variety of datasets (e.g. human and pig data) and even though subclasses exists (e.g. DataPathMultiorgan), they merely capture the dataset-specific access logic. Everything is designed to use only this class which has three entry points:
        - `DataPath.from_image_name()`: Creates one data path (technically one of the subclasses) based on its unique identifier as defined in `DataPath.image_name()`.
        - `DataPath.iterate()`: Iterates over all images in one folder generating path objects for all images.
        - `DataPath()`: Create a data path just based on the path to the image directory. In this case, no information about the attributes will be available, e.g. the `subject_name` attribute will not be available.

        To use this class, please make sure to set the htc framework up correctly as described in the README. For the following examples, make sure that you set the environment variable `PATH_Tivita_multiorgan_semantic`, e.g. via `export PATH_Tivita_multiorgan_semantic=~/htc/2021_02_05_Tivita_multiorgan_semantic`. If you provide the path manually (i.e. not using the settings), make sure that it points to the data subdirectory of the dataset.

        >>> from htc.settings import settings
        >>> path = next(DataPath.iterate(settings.data_dirs.semantic))
        >>> path.timestamp
        '2019_12_14_12_00_16'
        >>> path.subject_name
        'P041'
        >>> path = DataPath.from_image_name("P041#2019_12_14_12_00_16")
        >>> path.timestamp
        '2019_12_14_12_00_16'
        >>> path.subject_name
        'P041'
        >>> path = DataPath(settings.data_dirs.semantic / "subjects/P041/2019_12_14_12_00_16")
        >>> path.timestamp
        '2019_12_14_12_00_16'

        You can basically read all the available information about the image with this class, for example information about the camera (this comes from the Tivita *_meta.log files):
        >>> path.meta("Camera_CamID")
        '0102-00085'

        Another example is reading the corresponding annotations (if available). Please note that there might be more than one annotation per image:
        >>> path = DataPath.from_image_name("P091#2021_04_24_12_02_50")
        >>> path.meta("annotation_name")
        ['polygon#annotator1', 'polygon#annotator2', 'polygon#annotator3']
        >>> seg2 = path.read_segmentation('polygon#annotator2')
        >>> seg2.shape
        (480, 640)

        It is possible to select the annotation already as part of the image name to make it the default:
        >>> path = DataPath.from_image_name("P091#2021_04_24_12_02_50@polygon#annotator2")
        >>> np.all(seg2 == path.read_segmentation())
        True

        Datasets can also define a default annotation name which, if available, will be used if no name is given:
        >>> path.dataset_settings["annotation_name_default"]
        'polygon#annotator1'

        If you want to know more, read_segmentation() has all the details.

        Args:
            image_dir: Path (or string) to the image directory (timestamp folder).
            data_dir: Path (or string) to the data directory of the dataset (it should contain a dataset_settings.json file).
            intermediates_dir: Path (or string) to the intermediates directory of the dataset.
            dataset_settings: Reference to the settings of the dataset. If None and no settings could be found in the image directory, the parents of the image directory are searched. If available, the closest dataset_settings.json is used. Otherwise, the data path gets an empty dataset settings assigned.
            annotation_name_default: Default annotation_name(s) which will be used when reading the segmentation with read_segmentation() with no arguments.
        """
        if isinstance(image_dir, str):
            image_dir = Path(image_dir)
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        if isinstance(intermediates_dir, str):
            intermediates_dir = Path(intermediates_dir)

        self.image_dir = image_dir
        self.data_dir = data_dir
        self.intermediates_dir = intermediates_dir
        self._dataset_settings = dataset_settings
        if self.image_dir is not None:
            self.timestamp = self.image_dir.name
        self.annotation_name_default = annotation_name_default

        if (data_dir is None or intermediates_dir is None) and self.image_dir is not None:
            # Check whether the image directory is from a known location so that we can infer data/intermediates
            entry = settings.datasets.find_entry(self.image_dir)
            if entry is not None:
                if data_dir is None:
                    self.data_dir = entry["path_data"]
                if intermediates_dir is None:
                    self.intermediates_dir = entry["path_intermediates"]

    def __call__(self, *args, **kwargs) -> Path:
        """
        Converts the data path to a pathlib.Path object pointing to the folder with the image data.

        >>> path = DataPath.from_image_name('P043#2019_12_20_12_38_35')
        >>> str(path())  # doctest: +ELLIPSIS
        '/.../subjects/P043/2019_12_20_12_38_35'

        Returns: The full path to the image folder.
        """
        return self.image_dir

    def __str__(self) -> str:
        return str(self.image_dir)

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(self.image_name_annotations())

    def __eq__(self, other: Self) -> bool:
        return self.image_name_annotations() == other.image_name_annotations()

    def __lt__(self, other: Self) -> bool:
        return self.image_name_annotations() < other.image_name_annotations()

    @property
    def dataset_settings(self) -> DatasetSettings:
        if self._dataset_settings is None:
            if self.image_dir is not None and (path := self.image_dir / "dataset_settings.json").exists():
                self._dataset_settings = DatasetSettings(path)
            else:
                self._dataset_settings = DatasetSettings(path_or_data={})
                parent_paths = list(self.image_dir.parents)
                for p in parent_paths:
                    if (path := p / "dataset_settings.json").exists():
                        self._dataset_settings = DatasetSettings(path)
                        break

        return self._dataset_settings

    def cube_path(self) -> Path:
        """
        Path to the HSI data cube (*.dat file) for this image.

        >>> path = DataPath.from_image_name('P043#2019_12_20_12_38_35')
        >>> path.cube_path().name
        '2019_12_20_12_38_35_SpecCube.dat'

        Returns: Path to the hsi cube.
        """
        return self() / f"{self.timestamp}_SpecCube.dat"

    def read_cube(self, *reading_args, **reading_kwargs) -> np.ndarray:
        """
        Read the Tivita HSI cube (see read_tivita_hsi()).

        Args:
            reading_args: Positional arguments to be passed to read_tivita_hsi function.
            reading_kwargs: Keyword arguments to be passed to read_tivita_hsi function.

        Returns: HSI data cube.
        """
        from htc.tivita.hsi import read_tivita_hsi

        cube_path = self.cube_path()
        return read_tivita_hsi(cube_path, *reading_args, **reading_kwargs)

    def read_cube_raw(self, calibration_original: Union["CalibrationFiles", None] = None) -> np.ndarray:
        """
        Compute the raw Tivita HSI cube by multiplying the stored HSI cube with the corresponding white calibration matrix and adding the corresponding dark current measurement.

        Args:
            calibration_original: Calibration files to use. If None, the closest calibration files are searched based on the timestamp.

        Returns: HSI data cube.
        """
        cube = self.read_cube()

        # find matching white calibration file
        if calibration_original is None:
            from htc.cameras.calibration.CalibrationSwap import CalibrationSwap

            t = CalibrationSwap()
            calibration_original = t.original_calibration_files(self)
        return cube * calibration_original.white_image.numpy() + calibration_original.dark_image.numpy()

    def compute_oversaturation_mask(self, threshold: int = 1000) -> np.ndarray:
        """
        Calculates the oversaturation mask for the image. All pixels with at least one channel with a count above a given threshold are considered oversaturated.

        Args:
            threshold: Threshold to consider a camera count being oversaturated. The maximum count of the camera is 1023, therefore a value of 1000 is chosen as default to account for noise and slight miscalibrations (e.g. due to the corresponding calibration files not being available).

        Returns: Oversaturation mask of the image.
        """
        cube_raw = self.read_cube_raw()
        return np.any(cube_raw > threshold, axis=-1)

    def is_cube_valid(self) -> bool:
        """
        Checks whether the HSI cube is valid, i.e. not broken. Unfortunately, the Tivita camera may produce broken images due to unknown reasons. Here, we basically check whether we can read the cube and whether it contains invalid values (zero, negative pixels, infinite numbers).

        >>> path = DataPath.from_image_name('P043#2019_12_20_12_38_35')
        >>> path.is_cube_valid()
        True

        Returns: True if all checks pass. If False, then the image should be excluded from the analysis as the spectra may be completely wrong. R.I.P.
        """
        is_valid = True

        try:
            cube = self.read_cube()

            if cube.dtype != np.float32:
                settings.log.error(f"The cube {self} does not have float32 data type ({cube.dtype = })")
                is_valid = False

            if cube.shape != self.dataset_settings["shape"]:
                settings.log.error(f"The cube {self} does not have the correct shape ({cube.shape = })")
                is_valid = False

            infinite_values = ~np.isfinite(cube)
            if np.any(infinite_values):
                settings.log.error(f"The cube {self} contains invalid values (nan/inf) ({np.sum(infinite_values) = })")
                is_valid = False

            zero_pixels = np.all(cube == 0, axis=-1)
            if np.any(zero_pixels):
                settings.log.error(f"The cube {self} has {np.sum(zero_pixels)} zero pixels")
                is_valid = False
            else:
                if np.any(cube == 0):
                    settings.log.warning(
                        f"The cube {self} has {np.sum(cube == 0)} zero values (the cube is still used)"
                    )

            if np.all(cube < 0):
                settings.log.error(f"The cube {self} contains only negative values")
                is_valid = False
            else:
                negative_pixels = np.all(cube < 0, axis=-1)
                if np.any(negative_pixels):
                    settings.log.warning(
                        f"The cube {self} contains {np.sum(negative_pixels)} negative pixels (the cube is still used)"
                    )
        except Exception as e:
            settings.log.error(f"Cannot read the cube {self}: {e}")
            is_valid = False

        return is_valid

    def rgb_path_reconstructed(self) -> Path:
        """
        Path to the Tivita RGB image file.

        >>> path = DataPath.from_image_name('P043#2019_12_20_12_38_35')
        >>> path.rgb_path_reconstructed().name
        '2019_12_20_12_38_35_RGB-Image.png'

        Returns: Path to the RGB file from Tivita (including the black header).
        """
        # Attention: For Tivita first generation cameras, '*_RGB-Image.png' stores the reconstructed RGB image. For the second generation, '*_RGB-Image.png' also exists, but stores the RGB image from the RGB sensor!
        rgb_path = self() / f"{self.timestamp}_HSI-RGB-Image.png"
        if not rgb_path.exists():
            rgb_path = self() / f"{self.timestamp}_RGB-Image.png"
        return rgb_path

    def read_rgb_reconstructed(self, *reading_args, **reading_kwargs) -> np.ndarray:
        """
        Read the Tivita RGB image (see read_tivita_rgb()) which was reconstructed from the HSI data.

        Args:
            reading_args: Positional arguments to be passed to read_tivita_hsi function.
            reading_kwargs: Keyword arguments to pass to read_tivita_rgb function.

        Returns: RGB image.
        """
        from htc.tivita.rgb import read_tivita_rgb

        rgb_path = self.rgb_path_reconstructed()
        return read_tivita_rgb(rgb_path, *reading_args, **reading_kwargs)

    def rgb_path_sensor(self) -> Path:
        """
        Path to the image file of the RGB sensor, in case it exists (currently only available for the Tivita Surgery 2.0 camera).

        Returns: Path to the RGB file from the RGB sensor (with or without the black header).
        """
        rgb_path_reconstructed = (
            self() / f"{self.timestamp}_HSI-RGB-Image.png"
        )  # checking indirectly whether the image was taken with a Tivita Surgery 2.0 camera
        assert rgb_path_reconstructed.exists(), "RGB sensor data is only available for the Tivita Surgery 2.0 camera"
        rgb_path = self() / f"{self.timestamp}_RGB-Capture.png"
        if not rgb_path.exists():
            rgb_path = self() / f"{self.timestamp}_RGB-Image.png"
        return rgb_path

    def read_rgb_sensor(self, *reading_args, **reading_kwargs) -> np.ndarray:
        """
        Read the RGB image from the RGB sensor.

        Args:
            reading_args: Positional arguments to be passed to read_tivita_rgb function.
            reading_kwargs: Keyword arguments to pass to read_tivita_rgb function.

        Returns: RGB image.
        """
        from htc.tivita.rgb import read_tivita_rgb

        rgb_path = self.rgb_path_sensor()
        return read_tivita_rgb(rgb_path, *reading_args, **reading_kwargs)

    def segmentation_path(self) -> Union[Path, None]:
        """
        Path to the file which stores the segmentation image(s). These are not the raw annotations but the processed images, i.e. numpy array with the same shape as the image and annotations for all labels merged in one file.

        >>> path = DataPath.from_image_name('P043#2019_12_20_12_38_35')
        >>> path.segmentation_path().name
        'P043#2019_12_20_12_38_35.blosc'

        Returns: Path to the file containing the segmentation mask or None if it could not be created.
        """
        if self.intermediates_dir is None:
            return None
        else:
            return self.intermediates_dir / "segmentations" / f"{self.image_name()}.blosc"

    def read_segmentation(
        self, annotation_name: Union[str, list[str]] = None
    ) -> Union[np.ndarray, dict[str, np.ndarray], None]:
        """
        Read the segmentation as numpy array(s).

        >>> path = DataPath.from_image_name('P043#2019_12_20_12_38_35')
        >>> seg = path.read_segmentation()
        >>> seg.shape
        (480, 640)

        If you want to know the meaning of the label indices, you can use the dataset settings:
        >>> mapping = LabelMapping.from_path(path)
        >>> for l in np.unique(seg):
        ...     print(f"{l} = {mapping.index_to_name(l)}")
        4 = blue_cloth
        13 = heart
        18 = lung

        Args:
            annotation_name: Unique name of the annotation(s) for cases where multiple annotations exist (e.g. inter-rater variability). If None, will use the default set to this path or the default set in the dataset settings (in that order and only if available). If annotation_name is a string, it can also be in the form name1&name which will be treated identical to ['name1', 'name2']. If 'all', the original annotation file with all available annotations is returned.

        Returns: The segmentation mask of the image or None if no annotation is available. Is a dict if multiple annotations are requested.
        """
        path = self.segmentation_path()
        if path is not None and path.exists():
            data = decompress_file(path)
            if len(data) == 0:
                return None

            if annotation_name is None:
                # Maybe we have a default
                if self.annotation_name_default is not None:
                    annotation_name = self.annotation_name_default
                elif name := self.dataset_settings.get("annotation_name_default"):
                    annotation_name = name

            if type(annotation_name) == str:
                annotation_name = annotation_name.split("&")
                if len(annotation_name) == 1:
                    annotation_name = annotation_name[0]

            if annotation_name is None:
                assert len(data) == 1, (
                    "annotation_name can only be None if there is exactly one annotation in the dataset but the"
                    f" segmentation for the image {self.image_name()} contains the following annotations: {data.keys()}"
                )
                return next(iter(data.values()))
            elif annotation_name == "all":
                return data
            elif type(annotation_name) == list:
                return {name: data[name] for name in annotation_name}
            elif annotation_name in data:
                return data[annotation_name]
            else:
                # We may have annotations but not for the requested annotation_name
                return None
        else:
            return None

    def read_colorchecker_mask(
        self, return_spectra: bool = False, normalization: int = None
    ) -> Union[dict[str, Union[np.ndarray, pd.DataFrame, LabelMapping]], None]:
        """
        Reads a precomputed colorchecker mask.

        The corresponding image folder must have a annotations/*cc_classic.png file which can be precomputed with the ColorcheckerMaskCreation.ipynb notebook. The output contains information commonly needed when working with colorchecker images.

        Classic colorchecker image with 24 color chips
        >>> path = DataPath(settings.data_dirs.studies / "2021_03_30_straylight/Tivita/colorchecker/2021_03_30_13_54_53")
        >>> cc_mask = path.read_colorchecker_mask()
        >>> cc_mask["mask"].shape
        (480, 640)
        >>> list(cc_mask["median_table"].columns)
        ['label_index', 'label_name', 'label_color', 'row', 'col', 'median_spectrum', 'std_spectrum', 'median_normalized_spectrum', 'std_normalized_spectrum', 'median_sto2', 'std_sto2', 'median_nir', 'std_nir', 'median_twi', 'std_twi', 'median_ohi', 'std_ohi', 'median_thi', 'std_thi', 'median_tli', 'std_tli']
        >>> len(cc_mask["label_mapping"])
        24
        >>> cc_mask["label_mapping"].index_to_name(1)
        'dark_skin'

        Args:
            return_spectra: If True, the raw spectra of the color chips will be added to the key "spectra" in the output dictionary. The array has the shape (n_rows, n_cols, n_samples, n_channels) to easily access the raw spectra of the individual color chips.
            normalization: The normalization which will be applied to the raw spectra array. This parameter has no effect on the median table.

        Returns: None if no mask could be found or a dictionary with information about the colorchecker image:
            - mask: Array of shape (height, width) with the label index for each pixel.
            - median_table: Table with median spectra (unnormalized and L1-normalized) for each color chip.
            - label_mapping: The label mapping object to interpret the values of the mask array.
        """
        mask_dir = self.image_dir / "annotations"
        mask_paths = list(mask_dir.glob(f"{self.timestamp}#squares#automask#*.png"))  # searching for automasks
        if len(mask_paths) == 0:
            mask_paths = list(mask_dir.glob(f"{self.timestamp}#polygon#*.nrrd"))  # searching for MITK masks
        assert len(mask_paths) <= 1, f"Too many colorchecker masks available for {self.image_dir}"

        if len(mask_paths) == 0:
            settings.log.warning(
                f"Colorchecker mask cannot be found for {self.image_dir}. Please refer to"
                " ColorcheckerMaskCreation.ipynb or use MITK to generate the corresponding colorchecker mask!"
            )
            return None

        else:
            mask_path = mask_paths[0]

            from htc.utils.ColorcheckerReader import ColorcheckerReader

            if mask_path.suffix == ".png":
                mask = np.array(Image.open(mask_path))

                cc_board = mask_path.name.split("#")[-1].removesuffix(".png")
                assert cc_board in ["cc_classic", "cc_passport"], f"Unknown colorchecker board {cc_board} given!"
                label_mapping_mask = (
                    ColorcheckerReader.label_mapping_classic
                    if cc_board == "cc_classic"
                    else ColorcheckerReader.label_mapping_passport
                )

            else:
                from htc.utils.mitk.mitk_masks import nrrd_mask

                mask_dict = nrrd_mask(mask_path)
                mask = mask_dict["mask"]
                label_mapping_mitk = mask_dict["label_mapping"]

                cc_board = mask_path.name.split("#")[-1].removesuffix(".nrrd")
                assert cc_board in ["cc_classic", "cc_passport"], f"Unknown colorchecker board {cc_board} given!"
                label_mapping_mask = (
                    ColorcheckerReader.label_mapping_classic
                    if cc_board == "cc_classic"
                    else ColorcheckerReader.label_mapping_passport
                )
                # remap the MITK mask to the default format
                label_dict = label_mapping_mask.mapping_name_index | {"unlabeled": 0}
                label_mapping_tmp = LabelMapping(
                    label_dict,
                    zero_is_invalid=True,
                    label_colors=label_mapping_mask.label_colors,
                )
                label_mapping_tmp.map_tensor(mask, label_mapping_mitk)

            cube_norm = self.read_cube(normalization=1)
            cube = self.read_cube()

            # compute parameter images from cube
            sto2_img = self.compute_sto2(cube)
            nir_img = self.compute_nir(cube)
            twi_img = self.compute_twi(cube)
            ohi_img = self.compute_ohi(cube)
            thi_img = self.compute_thi(cube)
            tli_img = self.compute_tli(cube)

            table_rows = []
            for i in np.unique(mask[mask != 0]):
                spectra_norm = cube_norm[mask == i, :]
                spectra = cube[mask == i, :]

                selected_sto2 = sto2_img[mask == i]
                selected_nir = nir_img[mask == i]
                selected_twi = twi_img[mask == i]
                selected_ohi = ohi_img[mask == i]
                selected_thi = thi_img[mask == i]
                selected_tli = tli_img[mask == i]

                table_rows.append({
                    "label_index": i,
                    "label_name": label_mapping_mask.index_to_name(i),
                    "label_color": label_mapping_mask.index_to_color(i),
                    "row": (i - 1) // 6,
                    "col": (i - 1) % 6,
                    "median_spectrum": np.median(spectra, axis=0),
                    "std_spectrum": np.std(spectra, axis=0),
                    "median_normalized_spectrum": np.median(spectra_norm, axis=0),
                    "std_normalized_spectrum": np.std(spectra_norm, axis=0),
                    "median_sto2": np.median(selected_sto2.data),
                    "std_sto2": np.std(selected_sto2.data),
                    "median_nir": np.median(selected_nir.data),
                    "std_nir": np.std(selected_nir.data),
                    "median_twi": np.median(selected_twi.data),
                    "std_twi": np.std(selected_twi.data),
                    "median_ohi": np.median(selected_ohi.data),
                    "std_ohi": np.std(selected_ohi.data),
                    "median_thi": np.median(selected_thi.data),
                    "std_thi": np.std(selected_thi.data),
                    "median_tli": np.median(selected_tli.data),
                    "std_tli": np.std(selected_tli.data),
                })
            table = pd.DataFrame(table_rows)

            res = {"mask": mask, "median_table": table, "label_mapping": label_mapping_mask}

            if return_spectra:
                # Add additional arrays with the raw spectral data for each color chip
                _, counts = np.unique(mask[label_mapping_mask.is_index_valid(mask)], return_counts=True)
                assert len(np.unique(counts)) == 1
                n_rows = table["row"].max() + 1
                n_cols = table["col"].max() + 1
                n_samples = counts[0]
                n_channels = cube.shape[-1]

                spectra = np.empty((n_rows, n_cols, n_samples, n_channels), dtype=cube.dtype)
                for i, row in table.iterrows():
                    spectra[row["row"], row["col"]] = cube[mask == row["label_index"], :]

                if normalization is not None:
                    with np.errstate(invalid="ignore"):
                        spectra = spectra / np.linalg.norm(spectra, ord=normalization, axis=-1, keepdims=True)
                        spectra = np.nan_to_num(spectra, copy=False)

                res["spectra"] = spectra

            return res

    def annotated_labels(self, annotation_name: Union[str, list[str]] = None) -> list[str]:
        """
        Gives a list of all label names which are part of the segmentation mask (based on the corresponding *.blosc file).

        >>> path = DataPath.from_image_name('P070#2020_07_25_00_29_02')
        >>> path.annotated_labels()
        ['anorganic_artifact', 'fat_subcutaneous', 'foil', 'heart', 'lung', 'metal', 'muscle', 'organic_artifact', 'skin', 'unsure']

        The annotated labels may be a subset of the image labels, if they exist:
        >>> path.meta("image_labels")
        ['anorganic_artifact', 'fat_subcutaneous', 'foil', 'heart', 'lung', 'metal', 'muscle', 'organic_artifact', 'skin', 'unsure']
        >>> set(path.meta("image_labels")).issuperset(path.annotated_labels())
        True

        The image labels are weak labels and only indicate that a certain label is present in the image but they convey no information about whether the label is also annotated.

        Args:
            annotation_name: Name of the annotation(s) passed on to read_segmentation(). If it refers to more than one annotation, the (sorted) unique set of all labels will be returned.

        Returns: Sorted list of valid label names. If no valid labels are available, an empty list is returned.
        """
        names = []

        label_mask = self.read_segmentation(annotation_name)

        if label_mask is not None:
            if type(label_mask) == dict:
                # label names across all annotations
                label_indices = [pd.unique(mask.flatten()) for mask in label_mask.values()]
                if len(label_indices) > 1:
                    label_indices = np.concatenate(label_indices)
                    label_indices = pd.unique(label_indices)
                else:
                    label_indices = label_indices[0]
            else:
                label_indices = pd.unique(label_mask.flatten())

            mapping = LabelMapping.from_path(self)
            label_indices = [mapping.index_to_name(l) for l in label_indices if mapping.is_index_valid(l)]
            names = sorted(label_indices)

        return names

    def _code_from_official(self, name: str) -> Union[ModuleType, None]:
        """
        Loads a function from the official code provided by Diaspective Vision.

        Args:
            name: Name of the function to load.

        Returns: The function or None if the code is not available.
        """
        options = ["htc.tivita.functions_official", "functions_official"]
        module = None
        for option in options:
            try:
                module = importlib.import_module(option)
                break
            except ImportError:
                pass

        if module is not None:
            return getattr(module, name, None)
        else:
            return None

    def _load_precomputed_parameters(self) -> Union[Union[np.ndarray, int], dict[Any, Union[np.ndarray, int]]]:
        """
        Loads the precomputed parameter images from the corresponding *.blosc file. A `ValueError` is thrown if the *.blosc file does not exist.
        """
        params_path = self.intermediates_dir / "preprocessing" / "parameter_images" / f"{self.image_name()}.blosc"
        if not params_path.exists():
            raise ValueError(
                f"Could not compute the parameter images for {self.image_name()}. Neither are precomputed values"
                f" available at {params_path} nor is the official code from Diaspective Vision importable. Please note"
                " that the code for computing the parameter images is intellectual property of Diaspective Vision and"
                " you must obtain it directly from the company if you want to use it. If you have the code, please"
                " provide it in a script with the name `functions_official` and make sure it can be imported (e.g."
                " `from functions_official import calc_sto2` should work)."
            )

        return decompress_file(params_path)

    def compute_sto2(self, cube: np.ndarray = None) -> np.ndarray:
        """
        Computes the Tissue oxygen saturation (StO2) for the image.

        >>> path = DataPath.from_image_name('P043#2019_12_20_12_38_35')
        >>> sto2 = path.compute_sto2()
        >>> sto2.shape
        (480, 640)

        The result is a masked array since automatic background detection (provided by Tivita) is applied to the cube. However, you can always access the raw data
        >>> sto2.data.shape
        (480, 640)

        or the background mask separately if necessary. In the background mask, True indicates that the corresponding pixel is part of the background.
        >>> sto2.mask.shape
        (480, 640)
        >>> np.unique(sto2.mask)
        array([False,  True])

        Note: Parameter images can only be computed if the corresponding code is available in a file called `functions_official` (e.g. `from functions_official import calc_sto2` should work). This is not the case per default since the code is intellectual property of Diaspective Vision. If you need the code, please contact the company directly. If the code cannot be found, the precomputed parameter images will be loaded instead. If they are not available, a `ValueError` will be raised.

        Args:
            cube: If not None, will use this cube instead of loading it.

        Returns: The StO2 parameter image (as numpy masked array) with values in the range [0;1].
        """
        try:
            from htc.tivita.functions_official import calc_sto2, calc_sto2_2_helper, detect_background

            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                cube = self.read_cube() if cube is None else cube
                if self.meta("Camera_CamID") is None or self.meta("Camera_CamID") in [
                    "0102-00057",
                    "0102-00085",
                    "0102-00098",
                    "0202-00113",
                    "0202-00118",
                ]:
                    sto2_img = calc_sto2(cube)  # Halogen formula should be used
                else:
                    sto2_img = calc_sto2_2_helper(cube)  # LED formula should be used
                param = np.nan_to_num(np.rot90(sto2_img, k=-1), copy=False)
                background = np.rot90(detect_background(cube), k=-1)

                return np.ma.MaskedArray(param, background == 0, fill_value=0)
        except ImportError:
            params = self._load_precomputed_parameters()
            return np.ma.MaskedArray(params["StO2"], params["background"], fill_value=0)

    def compute_nir(self, cube: np.ndarray = None) -> np.ndarray:
        """
        Computes the NIR Perfusion Index for the image.

        >>> path = DataPath.from_image_name('P043#2019_12_20_12_38_35')
        >>> nir = path.compute_nir()
        >>> nir.shape
        (480, 640)

        See compute_sto2() for information about the automatic background detection.

        Args:
            cube: If not None, will use this cube instead of loading it.

        Returns: The NIR parameter image (as numpy masked array) with values in the range [0;1].
        """
        calc_nir = self._code_from_official("calc_nir")
        detect_background = self._code_from_official("detect_background")

        if calc_nir is not None and detect_background is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                cube = self.read_cube() if cube is None else cube
                param = np.nan_to_num(np.rot90(calc_nir(cube), k=-1), copy=False)
                background = np.rot90(detect_background(cube), k=-1)

                return np.ma.MaskedArray(param, background == 0, fill_value=0)
        else:
            params = self._load_precomputed_parameters()
            return np.ma.MaskedArray(params["NIR"], params["background"], fill_value=0)

    def compute_twi(self, cube: np.ndarray = None) -> np.ndarray:
        """
        Computes the Tissue Water Index (TWI) for the image.

        >>> path = DataPath.from_image_name('P043#2019_12_20_12_38_35')
        >>> twi = path.compute_twi()
        >>> twi.shape
        (480, 640)

        Args:
            cube: If not None, will use this cube instead of loading it.

        Returns: The TWI parameter image (as numpy masked array) with values in the range [0;1].
        """
        calc_twi = self._code_from_official("calc_twi")
        detect_background = self._code_from_official("detect_background")

        if calc_twi is not None and detect_background is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                cube = self.read_cube() if cube is None else cube
                param = np.nan_to_num(np.rot90(calc_twi(cube), k=-1), copy=False)
                background = np.rot90(detect_background(cube), k=-1)

                return np.ma.MaskedArray(param, background == 0, fill_value=0)
        else:
            params = self._load_precomputed_parameters()
            return np.ma.MaskedArray(params["TWI"], params["background"], fill_value=0)

    def compute_ohi(self, cube: np.ndarray = None) -> np.ndarray:
        """
        Computes the Organ Hemoglobin Index (OHI) for the image.

        >>> path = DataPath.from_image_name('P043#2019_12_20_12_38_35')
        >>> ohi = path.compute_ohi()
        >>> ohi.shape
        (480, 640)

        Args:
            cube: If not None, will use this cube instead of loading it.

        Returns: The OHI parameter image (as numpy masked array) with values in the range [0;1].
        """
        calc_ohi = self._code_from_official("calc_ohi")
        detect_background = self._code_from_official("detect_background")

        if calc_ohi is not None and detect_background is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                cube = self.read_cube() if cube is None else cube
                param = np.nan_to_num(np.rot90(calc_ohi(cube), k=-1), copy=False)
                background = np.rot90(detect_background(cube), k=-1)

                return np.ma.MaskedArray(param, background == 0, fill_value=0)
        else:
            params = self._load_precomputed_parameters()
            return np.ma.MaskedArray(params["OHI"], params["background"], fill_value=0)

    def compute_tli(self, cube: np.ndarray = None) -> np.ndarray:
        """
        Computes the Tissue Lipid Index (TLI) for the image.

        >>> path = DataPath.from_image_name('P043#2019_12_20_12_38_35')
        >>> tli = path.compute_tli()
        >>> tli.shape
        (480, 640)

        Args:
            cube: If not None, will use this cube instead of loading it.

        Returns: The TLI parameter image (as numpy masked array) with values in the range [0;1].
        """
        calc_tli = self._code_from_official("calc_tli")
        detect_background = self._code_from_official("detect_background")

        if calc_tli is not None and detect_background is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                cube = self.read_cube() if cube is None else cube
                param = np.nan_to_num(np.rot90(calc_tli(cube), k=-1), copy=False)
                background = np.rot90(detect_background(cube), k=-1)

                return np.ma.MaskedArray(param, background == 0, fill_value=0)
        else:
            params = self._load_precomputed_parameters()
            return np.ma.MaskedArray(params["TLI"], params["background"], fill_value=0)

    def compute_thi(self, cube: np.ndarray = None) -> np.ndarray:
        """
        Computes the Tissue Hemoglobin Index (THI) for the image.

        >>> path = DataPath.from_image_name('P043#2019_12_20_12_38_35')
        >>> thi = path.compute_thi()
        >>> thi.shape
        (480, 640)

        Args:
            cube: If not None, will use this cube instead of loading it.

        Returns: The THI parameter image (as numpy masked array) with values in the range [0;1].
        """
        calc_ohi = self._code_from_official("calc_ohi")
        detect_background = self._code_from_official("detect_background")

        if calc_ohi is not None and detect_background is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                cube = self.read_cube() if cube is None else cube
                param = np.clip(calc_ohi(cube) * 2, 0.000001, 1)
                param = np.nan_to_num(np.rot90(param, k=-1), copy=False)
                background = np.rot90(detect_background(cube), k=-1)

                return np.ma.MaskedArray(param, background == 0, fill_value=0)
        else:
            params = self._load_precomputed_parameters()
            return np.ma.MaskedArray(params["THI"], params["background"], fill_value=0)

    def camera_meta_path(self) -> Path:
        """
        Path to the camera meta data file (*_meta.log).

        >>> path = DataPath.from_image_name('P043#2019_12_20_12_38_35')
        >>> path.camera_meta_path().name
        '2019_12_20_12_38_35_meta.log'

        Note: Not every image has a meta file so the returned path may point to a non-existent file.

        Returns: Path to the meta.log file (may not exist).
        """
        return self() / f"{self.timestamp}_meta.log"

    def read_camera_meta(self) -> Union[dict, None]:
        """
        Read the camera meta data (see read_meta_file()).

        Note: Only use this function if you want to read all camera meta information once. Otherwise, you can also use the meta() method to read meta information about the camera. The camera meta is always cached in this class after reading.

        Returns: Dictionary with the metadata or None if the *_meta.log file is not available.
        """
        from htc.tivita.metadata import read_meta_file

        camera_meta_path = self.camera_meta_path()
        if not camera_meta_path.exists():
            return None
        else:
            return read_meta_file(camera_meta_path)

    def patient_meta_path(self) -> Union[Path, None]:
        """
        Path to the patient meta data file (*.xml).

        >>> path = DataPath(settings.data_dirs.studies / "2022_09_29_Surgery2_baseline" / "2022_09_29_17_04_13")
        >>> path.patient_meta_path().name
        'calibration_white.xml'

        Note: Not every image has a patient meta file and if such a file does not exist, None will be returned.

        Returns: Path to the *.xml file (may be None if non-existent).
        """
        pat_meta_path = sorted(self().rglob("*.xml"))
        assert len(pat_meta_path) <= 1, f"Too many .xml files found: {pat_meta_path}"
        if len(pat_meta_path) == 1:
            return pat_meta_path[0]
        else:
            return None

    def read_patient_meta(self) -> Union[dict, None]:
        """
        Read the patient meta data (see read_meta_patient()).

        Note: Only use this function if you want to read all patient meta information once. Otherwise, you can also use the meta() method to read meta information about the patient. The patient meta is always cached in this class after reading.

        Returns: Dictionary with the metadata or None if the *.xml file is not available.
        """
        from htc.tivita.metadata import read_meta_patient

        patient_meta_path = self.patient_meta_path()
        if patient_meta_path is not None:
            return read_meta_patient(patient_meta_path)
        else:
            return None

    def annotation_meta_path(self) -> Path:
        """
        Path to a file with additional meta data (meta labels) for this image (custom, non-Tivita extension).

        >>> path = DataPath.from_image_name('P093#2021_04_28_08_49_29')
        >>> path.annotation_meta_path().name
        '2021_04_28_08_49_29_meta.json'

        Returns: Path to the file which contains additional meta labels (e.g. information about the situs or the angle). The file may not exist.
        """
        return self() / "annotations" / f"{self.timestamp}_meta.json"

    def read_annotation_meta(self) -> Union[dict, None]:
        """
        Read additional meta labels if available.

        Note: Only use this function if you want to read all meta labels at once. Otherwise, you can also use the meta() method to read meta labels. The meta labels are always cached in this class.

        >>> path = DataPath.from_image_name('P093#2021_04_28_08_49_29')
        >>> meta_labels = path.read_annotation_meta()
        >>> meta_labels['image_labels']
        ['stomach', 'gallbladder', 'peritoneum']

        Returns: Dictionary with the meta labels or None if not available.
        """
        if self.image_name() not in DataPath._meta_labels_cache:
            if self.annotation_meta_path().exists():
                with self.annotation_meta_path().open() as f:
                    meta_labels = json.load(f)
            else:
                meta_labels = None

            DataPath._meta_labels_cache[self.image_name()] = meta_labels

        return DataPath._meta_labels_cache[self.image_name()]

    def annotation_meta_schema(self) -> Union[JSONSchemaMeta, None]:
        """
        Read the associated JSON schema for the annotation meta labels if available (`meta.schema` file in the `data` directory). This is useful to retrieve additional information about the meta labels (e.g. description, type, unit, etc.).

        >>> path = DataPath.from_image_name('P087#2021_04_16_10_12_31')
        >>> schema = path.annotation_meta_schema()
        >>> schema["label_meta/kidney/angle"].meta("unit")
        'degree'

        Returns: The schema object or None if not available.
        """
        if self.data_dir is not None and (schema_path := self.data_dir / "meta.schema").exists():
            return JSONSchemaMeta(schema_path)
        else:
            return None

    def meta(self, key: str = None, default: Any = None) -> Any:
        """
        Read meta information about the image (e.g. camera_name). This makes use of the meta table in intermediates/tables/*meta.feather if it exists instead of extracting the information from the meta files directly (faster). The information is always on the image level (and e.g. not on the label level).

        >>> path = DataPath.from_image_name('P043#2019_12_20_12_38_35')
        >>> path.meta("Camera_CamID")
        '0102-00085'

        There are two additional meta information which can be read via this function. The camera name, which is an extension to `Camera_CamID` and can account for sensor or other hardware changes:
        >>> path.meta("camera_name")
        '0102-00085_correct-1'

        And `annotation_name` gives an overview over all available annotations for this image:
        >>> path.meta("annotation_name")
        ['semantic#primary']

        It is also possible to read annotation meta information with this function:
        >>> path = DataPath.from_image_name("P091#2021_04_24_12_02_50")
        >>> path.meta("label_meta/omentum/situs")
        2

        Args:
            key: Name of the value you want to read. If None, all available camera meta information for this image will be returned (camera and annotation meta).
            default: Default value which will be returned if the key was not found.

        Returns: The requested value or None if it could not be found.
        """
        from htc.tivita.metadata import generate_metadata_table

        meta = DataPath._find_image(self.image_name())
        if meta is None:
            if key == "annotation_name":
                names = self.read_segmentation("all")
                meta = {"annotation_name": list(names.keys()) if names is not None else None}
            else:
                meta = dict(generate_metadata_table([self]).iloc[0])
        assert (
            meta is not None and type(meta) == dict
        ), "There must always be at least some meta information (e.g. image_name)"

        if key is None:
            # User requests all available information
            value = meta
            if (meta_labels := self.read_annotation_meta()) is not None:
                value |= meta_labels
            if "annotation_name" not in value:
                names = self.read_segmentation("all")
                value["annotation_name"] = list(names.keys()) if names is not None else None
        else:
            if key in meta:
                value = meta[key]
            else:
                # User may requests the additional meta labels from the JSON file
                meta_labels = self.read_annotation_meta()
                if meta_labels is None:
                    value = default
                else:
                    # The hierarchical meta labels can be read as layer/value
                    meta_labels = Config(meta_labels)
                    value = meta_labels.get(key, default)

        if type(value) == np.ndarray and value.ndim == 1:
            value = value.tolist()

        return value

    def build_path(self, base_folder: Path) -> Path:
        """
        Constructs a path to the experiment folder based on the properties of the data path:

        >>> path = DataPath.from_image_name('P043#2019_12_20_12_38_35')
        >>> base_folder = Path('/test')
        >>> str(path.build_path(base_folder))
        '/test/P043/2019_12_20_12_38_35'

        This is useful if you want to find the same data on a different location.

        Args:
            base_folder: The base folder on which to append the other parts of the data path.

        Returns: The complete path up to the level of the experiment folder.
        """
        pass

    def image_name(self) -> str:
        """
        Unique identifier for this image. This is important in many scenarios to quickly access or identify images. The name is usually provided by the dataset and resembles the dataset structure (see subclasses for examples).

        If the subclass does not overwrite this method, the image name will be composed of the camera id and the timestamp:
        >>> path = DataPath(settings.data_dirs.studies / "2021_12_16_Surgery2_first_experiments/2021_12_16_13_50_04")
        >>> path.image_name()
        '0615-00023#2021_12_16_13_50_04'

        Returns: Unique identifier for the data sample.
        """
        meta = self.read_camera_meta()
        if meta is not None and "Camera_CamID" in meta:
            name = f"{meta['Camera_CamID']}#{self.timestamp}"
        else:
            name = f"unknown#{self.timestamp}"

        return name

    def image_name_parts(self) -> list[str]:
        """
        Names of the image_name parts (e.g. folder names from the root to the image).

        >>> path = DataPath.from_image_name('P043#2019_12_20_12_38_35')
        >>> path.image_name_parts()
        ['subject_name', 'timestamp']

        Returns: The parts which make up the image_name of the data path. This is e.g. useful to expand 'image_name' columns in data frames.
        """
        return ["Camera_CamID", "timestamp"]

    def image_name_typed(self) -> dict[str, Any]:
        """
        The image_name of the path together with its parts:

        >>> path = DataPath.from_image_name('P043#2019_12_20_12_38_35')
        >>> path.image_name_typed()
        {'subject_name': 'P043', 'timestamp': '2019_12_20_12_38_35'}

        Returns: Dictionary with the image_name parts (values) and its common names (keys).
        """
        return dict(zip(self.image_name_parts(), self.image_name().split("#")))

    def image_name_annotations(self) -> str:
        """
        Unique name of the image including the associated or default annotation name(s).

        Note: A data path object may only be unique with its annotation name if more than one annotation type is used. Two data paths can have the same image name but different annotation names. This implication is important when using the image name as a unique identifier (e.g. in dictionaries or tables).

        >>> path = next(DataPath.iterate(settings.data_dirs.semantic))
        >>> path.image_name_annotations()
        'P041#2019_12_14_12_00_16@semantic#primary'

        Returns: Image name with annotation information appended.
        """
        name = self.image_name()

        annotations = self.annotation_names()
        if len(annotations) > 0:
            name += "@" + "&".join(annotations)

        return name

    def datetime(self) -> datetime:
        return datetime.strptime(self.timestamp, "%Y_%m_%d_%H_%M_%S")

    def annotation_names(self) -> list[str]:
        """
        Returns the names of all associated annotations for this image.

        The annotation name may be explicitly set when constructing the data path from the image name:
        >>> path = DataPath.from_image_name("P091#2021_04_24_12_02_50@polygon#annotator1&polygon#annotator2")
        >>> path.annotation_names()
        ['polygon#annotator1', 'polygon#annotator2']

        If no annotation name is explicitly set, the default annotation name from the dataset settings is used:
        >>> path = DataPath.from_image_name("P091#2021_04_24_12_02_50")
        >>> path.annotation_names()
        ['polygon#annotator1']

        The list of all possible annotation names can be retrieved via the `meta()` method:
        >>> path.meta("annotation_name")
        ['polygon#annotator1', 'polygon#annotator2', 'polygon#annotator3']

        Returns: List of annotation names. If no annotation names are associated with this image (neither directly nor via the dataset settings), an empty list is returned.
        """
        if self.annotation_name_default is not None:
            # Explicit default set to this class
            if type(self.annotation_name_default) == str:
                assert "&" not in self.annotation_name_default, "Multiple annotations should already be resolved."
                return [self.annotation_name_default]
            else:
                return self.annotation_name_default
        else:
            name = self.dataset_settings.get("annotation_name_default")
            if name is not None:
                # Default via dataset settings
                if type(name) == str:
                    name = name.split("&")
                return name
            else:
                return []

    @staticmethod
    def _build_cache(local: bool) -> dict[str, Any]:
        # We use a dict for the cache because it is much faster than a dataframe
        cache = {}

        for env_key in settings.datasets.env_keys():
            if not env_key.upper().startswith("PATH_TIVITA"):
                continue

            entry = settings.datasets.get(env_key, local_only=local)
            if entry is None:
                continue

            if (local and entry["location"] == "local") or (not local and entry["location"] == "network"):
                table_path = list((entry["path_intermediates"] / "tables").glob("*@meta.feather"))
                if len(table_path) > 0:
                    assert len(table_path) == 1, f"More than one meta table found for {entry}"
                    table_path = table_path[0]

                    dsettings = DatasetSettings(entry["path_data"] / "dataset_settings.json")
                    df = pd.read_feather(table_path)
                    df["dsettings"] = dsettings
                    df["dataset_env_name"] = env_key
                    df["data_dir"] = entry["path_data"]
                    df["intermediates_dir"] = entry["path_intermediates"]

                    # Append the metadata for the current dataset to the global cache
                    df.set_index("image_name", inplace=True)
                    cache |= df.to_dict("index")

        return cache

    @staticmethod
    def _local_cache() -> dict[str, Any]:
        if DataPath._local_meta_cache is None:
            DataPath._local_meta_cache = DataPath._build_cache(local=True)

        return DataPath._local_meta_cache

    @staticmethod
    def _network_cache() -> dict[str, Any]:
        if DataPath._network_meta_cache is None:
            DataPath._network_meta_cache = DataPath._build_cache(local=False)

        return DataPath._network_meta_cache

    @staticmethod
    def _find_image(image_name: str) -> dict[str, Any]:
        cache = DataPath._local_cache()
        if image_name not in cache:
            # Avoid building the network cache if possible
            cache = DataPath._network_cache()
            if image_name not in cache:
                return None

        return cache[image_name]

    @staticmethod
    def image_name_exists(image_name: str) -> bool:
        """
        Checks whether the image name can be found.

        >>> DataPath.image_name_exists('P043#2019_12_20_12_38_35')
        True

        Args:
            image_name: Unique identifier of the path in the same format as for `from_image_name()`.

        Returns: True if the image name can be found, False otherwise.
        """
        if "@" in image_name:
            image_name = image_name.split("@")[0]

        if image_name.startswith("ref"):
            from htc.tivita.DataPathReference import DataPathReference

            return DataPathReference.image_name_exists(image_name)
        else:
            match = DataPath._find_image(image_name)
            return match is not None

    @staticmethod
    def from_image_name(image_name: str) -> Self:
        """
        Constructs a data path based on its unique identifier.

        This function can only be used if the corresponding dataset has a *meta.feather table with an overview of all paths of the dataset. This table can be created via the run_meta_table script.

        Note: Data path objects created via this method are cached and exists globally during program execution. If the same `image_name` is used again, then the reference to the same object as before is returned.

        Args:
            image_name: Unique identifier of the path. Usually in the form subject#timestamp but you can also extend it to define the default annotations to read, for example subject#timestamp@name1&name2.

        Returns: The data path object.
        """
        # The same path but with different annotation names should also be separate path objects since each object has its own annotation_name_default
        cache_name = image_name

        if image_name not in DataPath._data_paths_cache:
            if "@" in image_name:
                image_name, annotation_name = image_name.split("@")
                annotation_name = annotation_name.split("&")
                if len(annotation_name) == 1:
                    annotation_name = annotation_name[0]
            else:
                annotation_name = None

            if image_name.startswith("ref"):
                from htc.tivita.DataPathReference import DataPathReference

                DataPath._data_paths_cache[cache_name] = DataPathReference.from_image_name(image_name, annotation_name)
            else:
                match = DataPath._find_image(image_name)
                assert match is not None, (
                    f"Could not find the path for the image {image_name} ({len(DataPath._local_cache()) = },"
                    f" {len(DataPath._network_cache()) = })"
                )

                DataPathClass = match["dsettings"].data_path_class()
                if DataPathClass is None:
                    raise ValueError(
                        f"No known DataPath class for the dataset {match['dataset_env_name']}. Please make sure that"
                        " you have a dataset_settings.json file in your dataset which has a key data_path_class which"
                        " refers to a valid data path class (e.g. htc.tivita.DataPathMultiorgan>DataPathMultiorgan)"
                    )

                DataPath._data_paths_cache[cache_name] = DataPathClass(
                    match["data_dir"] / match["path"],
                    match["data_dir"],
                    match["intermediates_dir"],
                    match["dsettings"],
                    annotation_name,
                )

        return DataPath._data_paths_cache[cache_name]

    @staticmethod
    def iterate(
        data_dir: Path,
        filters: Union[list[Callable[[Self], bool]], None] = None,
        annotation_name: Union[str, list[str]] = None,
    ) -> Iterator[Self]:
        """
        Helper function to iterate over the folder structure of a dataset (e.g. subjects folder), yielding one image at a time.

        >>> paths = list(DataPath.iterate(settings.data_dirs.semantic))
        >>> len(paths)
        506

        Only images from one pig:
        >>> filter_pig = lambda p: p.subject_name == 'P041'
        >>> paths = list(DataPath.iterate(settings.data_dirs.semantic, filters=[filter_pig]))
        >>> len(paths)
        4

        For the default datasets, the path provided is expected to point to the `data` subfolder in the respective dataset directory.
        >>> settings.data_dirs.semantic.name
        'data'

        Args:
            data_dir: The path where the data is stored. The data folder should contain a dataset_settings.json file.
            filters: List of filters which can be used to alter the set of images returned by this function. Every filter receives a DataPath instance and the instance is only yielded when all filter return True for this path.
            annotation_name: Include only paths with this annotation name and use it as default in read_segmentation(). Must either be a lists of annotation names or as string in the form name1&name2 (which will automatically be converted to ['name1', 'name2']). If None, no default annotation name will be set and no images will be filtered by annotation name.

        Returns: Generator with all path objects.
        """
        if filters is None:
            filters = []

        if type(annotation_name) == str:
            annotation_name = annotation_name.split("&")

        if annotation_name is not None:
            annotation_name = set(annotation_name)
            # We keep the path if it has at least one of the requested annotations
            filters.append(
                lambda p: p.meta("annotation_name") is not None
                and len(set(p.meta("annotation_name")).intersection(annotation_name)) > 0
            )

        dsettings = DatasetSettings(data_dir / "dataset_settings.json")
        DataPathClass = dsettings.data_path_class()
        if DataPathClass is None:
            # The user may provide a subpath to the dataset, e.g. DataPath.iterate(settings.data_dirs.sepsis_ICU / "calibrations")
            # In this case, we still want to use the DataPathSepsisICU class which is specified in the dataset_settings.json file in the root data directory
            dataset_entry = settings.datasets.find_entry(data_dir)
            if dataset_entry is not None and data_dir.is_relative_to(dataset_entry["path_data"]):
                # Try all subdirectories beginning from the most deep up to the data dir root (=dataset_entry["path_data"])
                parts = list(data_dir.relative_to(dataset_entry["path_data"]).parts)
                while True:
                    dsettings = DatasetSettings(dataset_entry["path_data"] / Path(*parts) / "dataset_settings.json")
                    DataPathClass = dsettings.data_path_class()
                    if DataPathClass is not None:
                        break

                    if len(parts) == 0:
                        break
                    else:
                        parts.pop()

        if DataPathClass is None:
            if not (data_dir / "dataset_settings.json").exists() and (data_dir / "data").exists():
                settings.log.warning(
                    f"No dataset_settings.json file found in the data directory {data_dir} but the subdirectory data"
                    " exists in this directory. For the default datasets, please point data_dir to the data"
                    " subdirectory of the dataset, e.g. ~/htc/2021_02_05_Tivita_multiorgan_semantic/data"
                    " (=settings.data_dirs.semantic)"
                )

            from htc.tivita.DataPathTivita import DataPathTivita

            DataPathClass = DataPathTivita

        if annotation_name is not None:
            annotation_name = list(annotation_name)
            annotation_name_subclass = annotation_name[0] if len(annotation_name) == 1 else annotation_name
        else:
            annotation_name_subclass = None

        yield from DataPathClass.iterate(data_dir, filters, annotation_name_subclass)
