# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import logging
import os
from pathlib import Path
from typing import Union

from appdirs import user_config_dir
from dotenv import load_dotenv
from rich.console import Console
from rich.highlighter import ReprHighlighter
from rich.logging import RichHandler
from rich.theme import Theme

from htc.utils.DatasetDir import DatasetDir
from htc.utils.DuplicateFilter import DuplicateFilter
from htc.utils.MultiPath import MultiPath


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # Apply level-specific color
        levelname_prev = record.levelname
        record.levelname = f"[logging.level.{record.levelname.lower()}]{record.levelname}[/]"
        format_str = logging.Formatter.format(self, record)
        record.levelname = levelname_prev

        return format_str


class Settings:
    """
    This class holds the general settings relevant for the `htc` framework. It controls the logging or gives access to common path variables (e.g. results directory). As a user, you can control the behavior via environment variables.

    List of environment variables used by the htc framework:
    - `PATH_Tivita`: Any variable that starts with this name will be considered a data directory and will be accessible via the `settings.data_dirs` property. These variables will also be mounted automatically to the Docker container as read-only directory (useful for testing). Example: `PATH_Tivita_HeiPorSPECTRAL="/mnt/nvme_4tb/HeiPorSPECTRAL"` and then access it via `settings.data_dirs["HeiPorSPECTRAL"]`.

    It is also possible to set a shortcut for the dataset, e.g. `PATH_Tivita_my_dataset=~/htc/Tivita_my_dataset:shortcut=my_shortcut` and then access it via `settings.data_dirs.my_shortcut`.
    - `PATH_HTC_RESULTS`: Path to the directory where you want to store your results (e.g. training results). The path is accessible via the `settings.results_dir` property. Example: `PATH_HTC_RESULTS="~/htc/results"`
    - `PATH_HTC_RESULTS_`: If you are working only on one project, then setting `PATH_HTC_RESULTS` is enough. For multiple projects, however, it is a good idea to separate the result files per project to keep them organized (e.g. training results on the semantic pig dataset should go to a different directory than the training results for the tissue atlas). For this reason, it is possible to set more than one results directory and everything that starts with `PATH_HTC_RESULTS_` will be automatically registered in the framework. That is, they are all accessible via `settings.results_dir` so you can read any file from any of your results directories. Please note that you still need to set a main results directory via `PATH_HTC_RESULTS` to specify the path where new files will be written to.

    Let's consider an example where we are working mainly on the semantic segmentation project but also want to read files from the tissue atlas:

    Create a test environment with our two results directories and some test files
    >>> from htc.settings import settings
    >>> tmp_dir = getfixture('tmp_path')
    >>> results_semantic = tmp_dir / "results_semantic"
    >>> results_semantic.mkdir(parents=True, exist_ok=True)
    >>> n_bytes = (results_semantic / "testfile_semantic.txt").write_text("test")
    >>> results_atlas = tmp_dir / "results_atlas"
    >>> results_atlas.mkdir(parents=True, exist_ok=True)
    >>> n_bytes = (results_atlas / "testfile_atlas.txt").write_text("test")

    Mock our environment to use our temporary results directories. We also set results_semantic in `PATH_HTC_RESULTS` because we want new files to be written to this directory
    >>> monkeypatch = getfixture('monkeypatch')
    >>> for k in os.environ.keys():  # Make sure no other results directory is set
    ...     if k.startswith("PATH_HTC_RESULTS"):
    ...         monkeypatch.setenv(k, "")
    >>> monkeypatch.setenv("HTC_ADD_NETWORK_ALTERNATIVES", "false")
    >>> monkeypatch.setenv('PATH_HTC_RESULTS', str(results_semantic))
    >>> monkeypatch.setenv('PATH_HTC_RESULTS_SEMANTIC', str(results_semantic))
    >>> monkeypatch.setenv('PATH_HTC_RESULTS_ATLAS', str(results_atlas))
    >>> monkeypatch.setattr(settings, "_results_dir", None)  # May already exists from other doctests

    We can access our results directories via our settings
    >>> settings.results_dir  # doctest: +ELLIPSIS
    Class: MultiPath
    Root location: .../results_semantic (exists=True)
    Best location (considering needle .../results_semantic): .../results_semantic (exists=True)
    All locations:
    .../results_semantic (exists=True)
    .../results_atlas (exists=True)

    Now we can access files from both results directories
    >>> (settings.results_dir / "testfile_semantic.txt").exists()
    True
    >>> (settings.results_dir / "testfile_atlas.txt").exists()
    True

    If we create a new file, it will be created in our semantic results directory
    >>> (settings.results_dir / "new_testfile.txt").write_text("test")
    >>> files = [str(f) for f in sorted(settings.results_dir.rglob("*"))]
    >>> len(files)
    3
    >>> files  # doctest: +ELLIPSIS
    ['.../results_atlas/testfile_atlas.txt', '.../results_semantic/new_testfile.txt', '.../results_semantic/testfile_semantic.txt']

    The above behavior is useful for general scripts like `htc training` which can be used in any project. If you have a script or notebook which can only be used for one project, then it is also possible to access the specific results directory for this project explicitly
    >>> from htc.tissue_atlas.settings_atlas import settings_atlas
    >>> n_bytes = (settings_atlas.results_dir / "new_testfile_atlas.txt").write_text("test")
    >>> [str(f) for f in sorted(settings_atlas.results_dir.rglob("*"))]  # doctest: +ELLIPSIS
    ['.../results_atlas/new_testfile_atlas.txt', '.../results_atlas/testfile_atlas.txt']

    - `PATH_E130_Projekte` or `PATH_HTC_NETWORK`: Path to the network drive of your department. The value will be accessible via `settings.data_dirs.network_dir`. Example: `PATH_E130_Projekte="/mnt/E130-Projekte"`
    - `HTC_DOCKER_MOUNTS`: Additional mount locations for our Docker container. This is for example useful if part of the data lies on a separate disk and you link to this location in your dataset. With this environment variable, you can make the symbolic link also work in the Docker container. The syntax is `path_local1:path_docker1;path_local2:path_docker2`. Example: `HTC_DOCKER_MOUNTS="/mnt/nvme_4tb/2021_02_05_Tivita_multiorgan_masks/intermediates:/mnt/nvme_4tb/2021_02_05_Tivita_multiorgan_masks/intermediates"`
    - `PATH_HTC_DOCKER_RESULTS`: If you compute something in our Docker container, results will only be stored in the container and deleted as soon as the container exits (since the container is only intended for testing). Let this variable point to a directory of your choice to keep your Docker results. Example: `PATH_HTC_DOCKER_RESULTS="/my/results/folder"`
    - `HTC_ADD_NETWORK_ALTERNATIVES`: If set to the string `true`, will include results and intermediate directories on the network drive (default `false`). This is usually only required for testing. Example: `HTC_ADD_NETWORK_ALTERNATIVES="true"`
    - `HTC_ENV_OVERRIDE`: Whether environment variables defined in the .env file or in your user settings override existing variables (default `true`). Set this to `false` if you want that variables defined elsewhere (e.g. before the command: `ENV_NAME htc command`) have precedence. Example: `HTC_ENV_OVERRIDE="false"`
    - `HTC_MODEL_COMPARISON_TIMESTAMP`: Variable is read in settings_seg and can be used to overwrite the default comparison timestamp (e.g. used for reproducibility of the MIA2021 paper). Example: `HTC_MODEL_COMPARISON_TIMESTAMP="2022-02-03_22-58-44"`
    - `HTC_CUDA_MEM_FRACTION`: Used in run_training.py to limit the GPU memory to a fraction of the available GPU memory (e.g. to simulate GPUs with less memory). Example: `HTC_CUDA_MEM_FRACTION="0.5"`
    - `DKFZ_USERID`: Name of your AD account (DKFZ internal). This is useful for the communication with our cluster. Example: `DKFZ_USERID="a267c"`
    - `CLUSTER_FOLDER`: Path on the cluster for your division and user (DKFZ internal). The default is E130 and your user ID so there is usually no need to set this variable. Example: `CLUSTER_FOLDER="OE0176/a267c"`
    - 'SHARED_FOLDER': Path on the cluster for your division. The datasets used by the tissue classification group are stored at this path. Example: `SHARED_FOLDER="OE0176/shared_htc"`

    None of the variables are mandatory.
    """

    def __init__(self):
        # Logging setup
        highlighter = ReprHighlighter()
        highlighter.highlights.append(r"(?P<image_name>\w+#\w+(?:#\w+)*)")
        formatter = ColoredFormatter(r"[%(levelname)s][[italic]%(name)s[/]] %(message)s")

        try:
            from IPython import get_ipython

            self.is_interactive = get_ipython().__class__.__name__ in [
                "ZMQInteractiveShell",
                "TerminalInteractiveShell",
            ]
        except ModuleNotFoundError:
            self.is_interactive = False

        # Notebook settings
        if self.is_interactive:
            import random
            import uuid

            # This loads the Plotly library via CDN instead of embedding it in every notebook (reduces the file size)
            os.environ["PLOTLY_RENDERER"] = "plotly_mimetype+notebook_connected"

            # Plotly adds a random uuid to access the div for the plot. The following code ensures reproducible div ids
            # https://github.com/plotly/plotly.py/issues/3393
            uuid4_rnd = random.Random(0)

            def uuid4_seeded():
                """Generate a random UUID using random.Random."""
                return uuid.UUID(bytes=uuid4_rnd.randbytes(16), version=4)

            uuid.uuid4 = uuid4_seeded

        handler = RichHandler(
            markup=True,
            show_time=False,
            show_level=False,
            enable_link_path=False,  # File links don't work in notebooks and cause problems with nbval
            console=Console(
                width=(
                    120 if self.is_interactive else None
                ),  # Increase the default width in notebooks (unfortunately, rich cannot detect the browser width automatically: https://github.com/Textualize/rich/issues/504)
                theme=Theme(
                    {
                        "var": "dim",
                        "repr.image_name": "cyan",
                        "repr.str": "bright_black",
                        "repr.number": "grey69",
                        "logging.level.debug": "bright_blue",
                        "logging.level.info": "green4",
                        "logging.level.warning": "yellow",
                    }
                ),
            ),
            highlighter=highlighter,
        )
        handler.setFormatter(formatter)

        # Same formatting for all logs
        logging.basicConfig(handlers=[handler])
        logging.captureWarnings(True)
        logging.getLogger("py.warnings").addFilter(DuplicateFilter())

        self.log = logging.getLogger("htc")
        self.log.setLevel(logging.INFO)

        self.log_once = logging.getLogger("htc.no_duplicates")
        self.log_once.addFilter(DuplicateFilter())

        logging.getLogger("challenger-pydocker").setLevel(logging.INFO)

        if self.add_network_alternatives:
            # This info message is not relevant for testing
            self.log_once.addFilter(
                lambda record: "Falling back to the network drive (this may be slow)" not in record.msg
            )

        # Directories relative to this file
        self.htc_package_dir = Path(__file__).parent
        self.models_dir = self.htc_package_dir / "models"
        self.src_dir = self.htc_package_dir.parent
        if not (self.src_dir / ".gitignore").exists():
            # The source directory is only available when the package is installed in editable mode
            # Otherwise (when installed as package), src_dir is not meaningful (it will point to the site-packages folder)
            # This is mainly relevant for the CLI where we don't want to glob across all installed packages
            # To check how the package was installed, we look for a common file in the source directory
            self.src_dir = self.htc_package_dir

        # Read the .env file from the repository root and/or the user settings (if available)
        # Per default, environment variables are overridden by the .env file and the user settings, but this can be disabled (e.g. to allow monkeypatching during testing)
        env_override = os.getenv("HTC_ENV_OVERRIDE", "true") == "true"
        self.dotenv_path = self.src_dir / ".env"
        if self.dotenv_path.exists():
            load_dotenv(dotenv_path=self.dotenv_path, override=env_override)
        self.user_settings_path = Path(user_config_dir("htc")) / "variables.env"
        if self.user_settings_path.exists():
            load_dotenv(dotenv_path=self.user_settings_path, override=env_override)

        # Cluster settings
        self.dkfz_userid = os.getenv("DKFZ_USERID")
        self.cluster_folder = os.getenv("CLUSTER_FOLDER", f"OE0176/{self.dkfz_userid}")
        self.shared_folder = os.getenv("SHARED_FOLDER", "OE0176/shared_htc")

        # General
        self.default_seed = 1337
        self.label_index_thresh = 100

        # Colors for common labels
        # (label, color) mapping as generated by color_organs() in htc.utils.colors
        self.label_colors = {
            "stomach": "#A657C7",
            "small_bowel": "#ADBEE6",
            "small_intestine": "#ADBEE6",
            "colon": "#5F8217",
            "liver": "#D0856D",
            "gallbladder": "#000000",
            "pancreas": "#3733F0",
            "kidney": "#54B5D4",
            "spleen": "#60F07A",
            "bladder": "#67E619",
            "omentum": "#B9D912",
            "fat": "#E1E9AA",
            "subcutaneous_fat": "#e0536b",
            "lung": "#5267C7",
            "heart": "#0DA07C",
            "cartilage": "#0F92DB",
            "bone": "#F8B08C",
            "skin": "#2BF0F3",
            "muscle": "#CC170F",
            "peritoneum": "#98CC66",
            "aorta": "#F4D352",
            "major_vein": "#CCCCCC",
            "major_vein": "#CCCCCC",
            "veins": "#4E2A7E",
            "kidney_with_Gerotas_fascia": "#F43E4C",
            "kidney_with_fascia": "#F43E4C",
            "lymph_nodes": "#24A90A",
            "blue_cloth": "#B38919",
            "white_compress": "#F09B2D",
            "abdominal_linen": "#5E10B7",
            "silicone_gloves_white": "#AE0A4B",
            "silicone_gloves_blue": "#7C7CF4",
            "silicone_gloves_light_blue": "#B34737",
            "metal": "#E486C8",
            "bile_fluid": "#131192",
            "ignore": "#40BF4A",
            "syringe": "#E13BF1",
            "glove": "#1D639A",
            "foil": "#0DE322",
            "metal_parts": "#E916AC",
            "organic_artifact": "#932F7F",
            "anorganic_artifact": "#1AE08D",
            "colon_peritoneum": "#822317",
            "unsure": "#1A7F44",
            "tube": "#7AF5C0",
            "background": "#9E11E4",
            "diaphragm": "#c8ee68",
            "arteries": "#89af31",
            "ovary": "#7440f2",
            "ureter": "#a7e89b",
            "blood": "#de0d6f",
            "lymph_fluid": "#bb7af5",
            "urine": "#e1518d",
            "cauterization": "#ffbb00",
            "lig_teres_hep": "#d451e0",
            "fat_subcutaneous": "#e0536b",
            "fat_visceral": "#43c456",
            "meso": "#4363c4",
            "esophagus": "#4743c4",
            "unclear_organic": "#C49505",
            "stapler": "#C7DE12",
            "ligasure": "#1277DE",
            "monopolar": "#5B3478",
            "Exterior": "#00000000",  # Unlabeled parts in MITK
        }

        self.known_envs = (
            "PATH_Tivita",
            "PATH_TIVITA",
            "PATH_HTC_RESULTS",
            "PATH_HTC_RESULTS_",
            "PATH_E130_Projekte",
            "PATH_HTC_NETWORK",
            "HTC_DOCKER_MOUNTS",
            "PATH_HTC_DOCKER_RESULTS",
            "HTC_ADD_NETWORK_ALTERNATIVES",
            "HTC_ENV_OVERRIDE",
            "HTC_MODEL_COMPARISON_TIMESTAMP",
            "HTC_CUDA_MEM_FRACTION",
            "DKFZ_USERID",
            "CLUSTER_FOLDER",
            "SHARED_FOLDER",
        )

        self._data_dirs = None
        self._intermediates_dir = None
        self._results_dir = None

    @property
    def add_network_alternatives(self) -> bool:
        # Some tests (e.g. notebooks) require result or intermediate files and it is notoriously hard to share them between users
        # As a workaround, we store them on the network drives but this can be slow so this is only done during testing
        return os.getenv("HTC_ADD_NETWORK_ALTERNATIVES", "false") == "true"

    @property
    def data_dirs(self) -> DatasetDir:
        if self._data_dirs is None:
            if path_env := os.getenv("PATH_E130_Projekte", False):
                network_dir = Path(path_env)
            elif path_env := os.getenv("PATH_HTC_NETWORK", False):
                network_dir = Path(path_env)
            else:
                network_dir = None

            self._data_dirs = DatasetDir(network_dir=network_dir)

            # Automatically add all additional datasets which start with PATH_Tivita
            for env_name in os.environ.keys():
                if not env_name.upper().startswith("PATH_TIVITA"):
                    continue
                if env_name in self._data_dirs:
                    continue

                _, options = DatasetDir.parse_path_options(os.environ[env_name])
                shortcut = options.get("shortcut", None)

                # Per default, the dataset is accessible via three names. For example, for PATH_Tivita_HeiPorSPECTRAL=/my/dataset_folder_name:
                # - PATH_Tivita_HeiPorSPECTRAL
                # - dataset_folder_name
                # - atlas_pigs
                self._data_dirs.add_dir(
                    env_name,
                    shortcut=shortcut,
                    additional_names=[
                        env_name.removeprefix("PATH_Tivita_"),
                        env_name.upper().removeprefix("PATH_TIVITA_"),
                    ],
                )

        return self._data_dirs

    @property
    def intermediates_dir(self) -> Union[MultiPath, None]:
        if self._intermediates_dir is None:
            # Combine all intermediates from all data dirs into one variable
            dirs = []
            for name in self.data_dirs.dataset_names:
                found_entry = self.data_dirs.get(name, local_only=not self.add_network_alternatives, return_entry=True)
                if found_entry is None:
                    continue

                # found_dir points to the directory for the dataset which usually contains subfolders for data and intermediates
                found_dir = found_entry["path_intermediates"]
                if found_dir.exists():
                    try:
                        # Unfortunately, there is no direct way to reliable test whether the user has permissions (see issue #13)
                        list(found_dir.iterdir())  # Check whether the directory is accessible

                        dirs.append(found_dir)
                    except PermissionError:
                        self.log.info(
                            f"The data directory {found_dir} is available but not accessible (permission denied). The"
                            " directory will be skipped"
                        )
                        pass

            if len(dirs) == 0:
                self._intermediates_dir = False
                self.log.warning(
                    "Could not find an intermediates directory, probably because no data directory was found"
                )
            else:
                self._intermediates_dir = MultiPath(dirs[0])
                for d in dirs[1:]:
                    self._intermediates_dir.add_alternative(d)

        return None if not self._intermediates_dir else self._intermediates_dir

    @property
    def results_dir(self) -> Union[MultiPath, None]:
        if self._results_dir is None:
            if results_dir_path := os.getenv("PATH_HTC_RESULTS", False):
                self._results_dir = MultiPath(results_dir_path)

                # We always write to the main location
                self._results_dir.set_default_location(str(self._results_dir.find_best_location()))

                # Add additional result_dirs if available
                for k in os.environ.keys():
                    if k.startswith("PATH_HTC_RESULTS_"):
                        path = os.getenv(k)
                        if path and path != results_dir_path:  # No duplicate paths
                            self._results_dir.add_alternative(path)

                if self.add_network_alternatives:
                    local_location_names = [l.name for l in self._results_dir.possible_locations()]
                    for d in sorted(self.data_dirs.network_project.glob("results*")):
                        if d.name not in local_location_names:  # Do not add it if it already exists locally
                            self._results_dir.add_alternative(d)
            else:
                self._results_dir = False
                self.log.warning(
                    "Could not find the environment variable PATH_HTC_RESULTS so that a results directory will not be"
                    " available (scripts which use settings.results_dir will crash)"
                )

        return None if not self._results_dir else self._results_dir

    @property
    def training_dir(self) -> Union[MultiPath, None]:
        return self.results_dir / "training" if self.results_dir is not None else None


settings = Settings()
