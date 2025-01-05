<div align="center">
<a href="https://e130-hyperspectal-tissue-classification.s3.dkfz.de/figures/htc_logo.pdf"><img src="https://e130-hyperspectal-tissue-classification.s3.dkfz.de/figures/htc_logo.svg" alt="Logo" width="600" /></a>

[![Python](https://img.shields.io/pypi/pyversions/imsy-htc.svg)](https://pypi.org/project/imsy-htc)
[![PyPI version](https://badge.fury.io/py/imsy-htc.svg)](https://pypi.org/project/imsy-htc)
[![Tests](https://github.com/IMSY-DKFZ/htc/actions/workflows/tests.yml/badge.svg)](https://github.com/IMSY-DKFZ/htc/actions/workflows/tests.yml)

</div>

# Hyperspectral Tissue Classification

This package is a framework for automated tissue classification and segmentation on medical hyperspectral imaging (HSI) data. It contains:

-   The implementation of deep learning models to solve supervised classification and segmentation problems for a variety of different input spatial granularities (pixels, superpixels, patches and entire images, cf. figure below) and modalities (RGB data, raw and processed HSI data) from our paper [“Robust deep learning-based semantic organ segmentation in hyperspectral images”](https://doi.org/10.1016/j.media.2022.102488). It is based on [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/).
-   Corresponding pretrained models.
-   A pipeline to efficiently load and process HSI data, to aggregate deep learning results and to validate and visualize findings.
-   Presentation of several solutions to speed up the data loading process (see [Pytorch Conference 2023 poster details](./README.md#-dealing-with-io-bottlenecks-in-high-throughput-model-training) below).

<div align="center">
<a href="https://e130-hyperspectal-tissue-classification.s3.dkfz.de/figures/MIA_model_overview.pdf"><img src="https://e130-hyperspectal-tissue-classification.s3.dkfz.de/figures/MIA_model_overview.svg" alt="Overview of deep learning models in the htc framework, here shown for HSI input." /></a>
</div>

This framework is designed to work on HSI data from the [Tivita](https://diaspective-vision.com/en/) cameras but you can adapt it to different HSI datasets as well. Potential applications include:

-   Use our data loading and processing pipeline to easily access image and meta data for any work utilizing Tivita datasets.
-   This repository is tightly coupled to work with the public [HeiPorSPECTRAL](https://heiporspectral.org/) dataset. If you already downloaded the data, you only need to perform the setup steps and then you can directly use the `htc` framework to work on the data (cf. [our tutorials](./README.md#tutorials)).
-   Train your own networks and benefit from a pipeline offering e.g. efficient data loading, correct hierarchical aggregation of results and a set of helpful visualizations.
-   Apply deep learning models for different spatial granularities and modalities on your own semantically annotated dataset.
-   Use our pretrained models to initialize the weights for your own training.
-   Use our pretrained models to generate predictions for your own data.

If you use the `htc` framework, please consider citing the [corresponding papers](./README.md#papers). You can also cite this repository directly via:

<details closed>
<summary>Cite via BibTeX</summary>

```bibtex
@software{sellner_htc_2023,
  author    = {Sellner, Jan and Seidlitz, Silvia},
  publisher = {Zenodo},
  url       = {https://github.com/IMSY-DKFZ/htc},
  date      = {2024-10-23},
  doi       = {10.5281/zenodo.6577614},
  title     = {Hyperspectral Tissue Classification},
  version   = {v0.0.17},
}
```

</details>

## Setup

### Package Installation

This package can be installed via pip:

```bash
pip install imsy-htc
```

This installs all the required dependencies defined in [`requirements.txt`](./requirements.txt). The requirements include [PyTorch](https://pytorch.org/), so you may want to install it manually before installing the package in case you have specific needs (e.g. CUDA version).

> &#x26a0;&#xfe0f; This framework was developed and tested using the Ubuntu 20.04+ Linux distribution. Despite we do provide wheels for Windows and macOS as well, they are not tested.

> &#x26a0;&#xfe0f; Network training and inference was conducted using an RTX 3090 GPU with 24 GiB of memory. It should also work with GPUs which have less memory but you may have to adjust some settings (e.g. the batch size).

<details close>
<summary>PyTorch Compatibility</summary>

We cannot provide wheels for all PyTorch versions. Hence, a version of `imsy-htc` may not work with all versions of PyTorch due to changes in the ABI. In the following table, we list the PyTorch versions which are compatible with the respective `imsy-htc` version.

| `imsy-htc` | `torch` |
| ---------- | ------- |
| 0.0.9      | 1.13    |
| 0.0.10     | 1.13    |
| 0.0.11     | 2.0     |
| 0.0.12     | 2.0     |
| 0.0.13     | 2.1     |
| 0.0.14     | 2.1     |
| 0.0.15     | 2.2     |
| 0.0.15     | 2.3     |
| 0.0.16     | 2.4     |
| 0.0.17     | 2.5     |

However, we do not make explicit version constraints in the dependencies of the `imsy-htc` package because a future version of PyTorch may still work and we don't want to break the installation if it is not necessary.

> 💡 Please note that it is always possible to build the `imsy-htc` package with your installed PyTorch version yourself (cf. Developer Installation).

</details>

<details close>
<summary>Optional Dependencies (<code>imsy-htc[extra]</code>)</summary>

Some requirements are considered optional (e.g. if they are only needed by certain scripts) and you will get an error message if they are needed but unavailable. You can install them via

```bash
pip install --extra-index-url https://read_package:CnzBrgDfKMWS4cxf-r31@git.dkfz.de/api/v4/projects/15/packages/pypi/simple imsy-htc[extra]
```

or by adding the following lines to your `requirements.txt`

```
--extra-index-url https://read_package:CnzBrgDfKMWS4cxf-r31@git.dkfz.de/api/v4/projects/15/packages/pypi/simple
imsy-htc[extra]
```

This installs the optional dependencies defined in [`requirements-extra.txt`](./requirements-extra.txt), including for example our Python wrapper for the [challengeR toolkit](https://github.com/wiesenfa/challengeR).

</details>

<details close>
<summary>Docker</summary>

We also provide a Docker setup for testing. As a prerequisite:

-   Clone this repository
-   Install [Docker](https://docs.docker.com/get-docker/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
-   Install the required dependencies to run the Docker startup script:

```bash
pip install python-dotenv
```

Make sure that your environment variables are available and then bash into the container

```bash
export PATH_Tivita_HeiPorSPECTRAL="/path/to/the/dataset"
python run_docker.py bash
```

You can now run any commands you like. All datasets you provided via an environment variable that starts with `PATH_Tivita` will be accessible in your container (you can also check the generated `docker-compose.override.yml` file for details). Please note that the Docker container is meant for small testing only and not for development. This is also reflected by the fact that per default all results are stored inside the container and hence will also be deleted after exiting the container. If you want to keep your results, let the environment variable `PATH_HTC_DOCKER_RESULTS` point to the directory where you want to store the results.

</details>

<details close>
<summary>Developer Installation</summary>

If you want to make changes to the package code (which is highly welcome 😉), we recommend to install the `htc` package in editable mode in a separate conda environment:

```bash
# Set up the conda environment
conda create --yes --name htc python=3.12
conda activate htc

# Install the htc package and its requirements
pip install -r requirements-dev.txt
pip install --no-use-pep517 -e .
```

Before committing any files, please run the static code checks locally:

```bash
git add .
pre-commit run --all-files
```

</details>

### Environment Variables

This framework can be configured via environment variables. Most importantly, we need to know where your data is located (e.g. `PATH_Tivita_HeiPorSPECTRAL`) and where results should be stored (e.g. `PATH_HTC_RESULTS`). For a full list of possible environment variables, please have a look at the documentation of the [`Settings`](./htc/settings.py) class.

> 💡 If you set an environment variable for a dataset path, it is important that the variable name matches the folder name (e.g. the variable name `PATH_Tivita_HeiPorSPECTRAL` matches the dataset path `my/path/HeiPorSPECTRAL` with its folder name `HeiPorSPECTRAL`, whereas the variable name `PATH_Tivita_some_other_name` does not match). Furthermore, the dataset path needs to point to a directory which contains a `data` and an `intermediates` subfolder.

There are several options to set the environment variables. For example:

-   You can specify a variable as part of your bash startup script `~/.bashrc` or before running each command:
    ```bash
    PATH_HTC_RESULTS="~/htc/results" htc training --model image --config "models/image/configs/default"
    ```
    However, this might get cumbersome or might not give you the flexibility you need.
-   Recommended if you cloned this repository (in contrast to simply installing it via pip): You can create a `.env` file in the repository root and add your variables, for example:

    ```bash
    export PATH_Tivita_HeiPorSPECTRAL=/mnt/nvme_4tb/HeiPorSPECTRAL
    export PATH_HTC_RESULTS=~/htc/results

    # You can also add your own datasets via (the environment variable name must start with PATH_Tivita)
    # export PATH_Tivita_my_dataset=~/htc/Tivita_my_dataset:shortcut=my_shortcut
    # You can then access it via settings.data_dirs.my_shortcut
    ```

-   Recommended if you installed the package via pip: You can create user settings for this application. The location is OS-specific. For Linux the location might be at `~/.config/htc/variables.env`. Please run `htc info` upon package installation to retrieve the exact location on your system. The content of the file is of the same format as of the `.env` above.

After setting your environment variables, it is recommended to run `htc info` to check that your variables are correctly registered in the framework.

## Tutorials

A series of [tutorials](./tutorials) can help you get started on the `htc` framework by guiding you through different usage scenarios.

> 💡 The tutorials make use of our public HSI dataset [HeiPorSPECTRAL](https://heiporspectral.org/). If you want to directly run them, please download the dataset first and make it accessible via the environment variable `PATH_Tivita_HeiPorSPECTRAL` as described above.

-   As a start, we recommend to take a look at this [general notebook](./tutorials/General.ipynb) which showcases the basic functionalities of the `htc` framework. Namely, it demonstrates the usage of the `DataPath` class which is the entry point to load and process HSI data. For example, you will learn how to read HSI cubes, segmentation masks and meta data. Among others, you can use this information to calculate the median spectrum of an organ.
-   You want to perform some spectral analysis? Then we have a notebook on [loading and working with median spectra](./tutorials/MedianSpectra.ipynb) for you.
-   If you want to use our framework with your own dataset, it might be necessary to write a custom `DataPath` class so that you can load and process your images and annotations. We [collected some tips](./tutorials/CustomDataPath.md) on how this can be achieved.
-   You have some HSI data at hand and want to use one of our pretrained models to generate predictions? Then our [prediction notebook](./tutorials/CreatingPredictions.ipynb) has got you covered.
-   You want to use our pretrained models to initialize the weights for your own training? See the section about [pretrained models](./README.md#pretrained-models) below for details.
-   You want to use our framework to train a network? The [network training notebook](./tutorials/network_training/NetworkTraining.ipynb) will show you how to achieve this on the example of a heart and lung segmentation network.
-   If you are interested in our technical validation (e.g. because you want to compare your colorchecker images with ours) and need to create a mask to detect the different colorchecker fields, you might find our automatic [colorchecker mask creation pipeline](./htc/utils/ColorcheckerMaskCreation.ipynb) useful.

We do not have a separate documentation website for our framework yet. However, most of the functions and classes are documented so feel free to explore the source code or use your favorite IDE to display the documentation. If something does not become clear from the documentation, feel free to open an issue!

## Pretrained Models

This framework gives you access to a variety of pretrained segmentation and classification models. The models will be automatically downloaded, provided you specify the model type (e.g. `image`) and the run folder (e.g. `2022-02-03_22-58-44_generated_default_model_comparison`). It can then be used for example to [create predictions](./tutorials/CreatingPredictions.ipynb) on some data or as a baseline for your own training (see example below).

The following table lists all the models you can get:
| model type | modality | class | run folder |
| ----------- | ----------- | ----------- | ----------- |
| image | hsi | [`ModelImage`](./htc/models/image/ModelImage.py) | `2024-09-11_00-11-38_projected_rat2pig_nested-*-2` (outer folds: [0](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_rat2pig_nested-0-2.zip), [1](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_rat2pig_nested-1-2.zip), [2](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_rat2pig_nested-2-2.zip)) |
| image | hsi | [`ModelImage`](./htc/models/image/ModelImage.py) | `2024-09-11_00-11-38_projected_rat2human_nested-*-2` (outer folds: [0](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_rat2human_nested-0-2.zip), [1](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_rat2human_nested-1-2.zip), [2](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_rat2human_nested-2-2.zip)) |
| image | hsi | [`ModelImage`](./htc/models/image/ModelImage.py) | `2024-09-11_00-11-38_projected_pig2rat_nested-*-2` (outer folds: [0](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_pig2rat_nested-0-2.zip), [1](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_pig2rat_nested-1-2.zip), [2](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_pig2rat_nested-2-2.zip)) |
| image | hsi | [`ModelImage`](./htc/models/image/ModelImage.py) | `2024-09-11_00-11-38_projected_pig2human_nested-*-2` (outer folds: [0](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_pig2human_nested-0-2.zip), [1](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_pig2human_nested-1-2.zip), [2](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_projected_pig2human_nested-2-2.zip)) |
| image | hsi | [`ModelImage`](./htc/models/image/ModelImage.py) | `2024-09-11_00-11-38_joint_pig-p+rat-p2human_nested-*-2` (outer folds: [0](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_joint_pig-p+rat-p2human_nested-0-2.zip), [1](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_joint_pig-p+rat-p2human_nested-1-2.zip), [2](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_joint_pig-p+rat-p2human_nested-2-2.zip)) |
| image | hsi | [`ModelImage`](./htc/models/image/ModelImage.py) | `2024-09-11_00-11-38_baseline_rat_nested-*-2` (outer folds: [0](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_baseline_rat_nested-0-2.zip), [1](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_baseline_rat_nested-1-2.zip), [2](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_baseline_rat_nested-2-2.zip)) |
| image | hsi | [`ModelImage`](./htc/models/image/ModelImage.py) | `2024-09-11_00-11-38_baseline_pig_nested-*-2` (outer folds: [0](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_baseline_pig_nested-0-2.zip), [1](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_baseline_pig_nested-1-2.zip), [2](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_baseline_pig_nested-2-2.zip)) |
| image | hsi | [`ModelImage`](./htc/models/image/ModelImage.py) | `2024-09-11_00-11-38_baseline_human_nested-*-2` (outer folds: [0](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_baseline_human_nested-0-2.zip), [1](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_baseline_human_nested-1-2.zip), [2](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2024-09-11_00-11-38_baseline_human_nested-2-2.zip)) |
| image | hsi | [`ModelImage`](./htc/models/image/ModelImage.py) | [`2023-02-08_14-48-02_organ_transplantation_0.8`](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2023-02-08_14-48-02_organ_transplantation_0.8.zip) |
| image | rgb | [`ModelImage`](./htc/models/image/ModelImage.py) | [`2023-01-29_11-31-04_organ_transplantation_0.8_rgb`](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2023-01-29_11-31-04_organ_transplantation_0.8_rgb.zip) |
| image | hsi | [`ModelImage`](./htc/models/image/ModelImage.py) | [`2022-02-03_22-58-44_generated_default_model_comparison`](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2022-02-03_22-58-44_generated_default_model_comparison.zip) |
| image | param | [`ModelImage`](./htc/models/image/ModelImage.py) | [`2022-02-03_22-58-44_generated_default_parameters_model_comparison`](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip) |
| image | rgb | [`ModelImage`](./htc/models/image/ModelImage.py) | [`2022-02-03_22-58-44_generated_default_rgb_model_comparison`](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip) |
| patch | hsi | [`ModelImage`](./htc/models/image/ModelImage.py) | [`2022-02-03_22-58-44_generated_default_64_model_comparison`](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_64_model_comparison.zip) |
| patch | param | [`ModelImage`](./htc/models/image/ModelImage.py) | [`2022-02-03_22-58-44_generated_default_64_parameters_model_comparison`](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_64_parameters_model_comparison.zip) |
| patch | rgb | [`ModelImage`](./htc/models/image/ModelImage.py) | [`2022-02-03_22-58-44_generated_default_64_rgb_model_comparison`](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_64_rgb_model_comparison.zip) |
| patch | hsi | [`ModelImage`](./htc/models/image/ModelImage.py) | [`2022-02-03_22-58-44_generated_default_model_comparison`](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_model_comparison.zip) |
| patch | param | [`ModelImage`](./htc/models/image/ModelImage.py) | [`2022-02-03_22-58-44_generated_default_parameters_model_comparison`](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip) |
| patch | rgb | [`ModelImage`](./htc/models/image/ModelImage.py) | [`2022-02-03_22-58-44_generated_default_rgb_model_comparison`](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip) |
| superpixel_classification | hsi | [`ModelSuperpixelClassification`](./htc/models/superpixel_classification/ModelSuperpixelClassification.py) | [`2022-02-03_22-58-44_generated_default_model_comparison`](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/superpixel_classification@2022-02-03_22-58-44_generated_default_model_comparison.zip) |
| superpixel_classification | param | [`ModelSuperpixelClassification`](./htc/models/superpixel_classification/ModelSuperpixelClassification.py) | [`2022-02-03_22-58-44_generated_default_parameters_model_comparison`](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/superpixel_classification@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip) |
| superpixel_classification | rgb | [`ModelSuperpixelClassification`](./htc/models/superpixel_classification/ModelSuperpixelClassification.py) | [`2022-02-03_22-58-44_generated_default_rgb_model_comparison`](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/superpixel_classification@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip) |
| pixel | hsi | [`ModelPixel`](./htc/models/pixel/ModelPixel.py) | [`2022-02-03_22-58-44_generated_default_model_comparison`](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/pixel@2022-02-03_22-58-44_generated_default_model_comparison.zip) |
| pixel | param | [`ModelPixelRGB`](./htc/models/pixel/ModelPixelRGB.py) | [`2022-02-03_22-58-44_generated_default_parameters_model_comparison`](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/pixel@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip) |
| pixel | rgb | [`ModelPixelRGB`](./htc/models/pixel/ModelPixelRGB.py) | [`2022-02-03_22-58-44_generated_default_rgb_model_comparison`](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/pixel@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip) |

> 💡 The modality `param` refers to stacked tissue parameter images (named TPI in our paper [“Robust deep learning-based semantic organ segmentation in hyperspectral images”](https://doi.org/10.1016/j.media.2022.102488)). For the model type `patch`, pretrained models are available for the patch sizes 64 x 64 and 32 x 32 pixels. The modality and patch size is not specified when loading a model as it is already characterized by specifying a certain run folder.

> 💡 A wildcard `*` in the run folder name refers to a collection of models (e.g. from nested cross validation). You can use the name as noted in the table to retrieve all models from this collection as list of models or explicitly set the index to only retrieve one specific model from the collection. If you keep the wildcard for creating predictions (see below), all models will be loaded and the final prediction is an ensemble of the output from all individual networks (e.g. 15 networks with 3 outer and 5 inner folds).

After successful installation of the `htc` package, you can use any of the pretrained models listed in the table. There are several ways to use them but the general principle is that models are always specified via their `model` and `run_folder`.

### Option 1: Use the models in your own training pipeline

Every model class listed in the table has a static method [`pretrained_model()`](./htc/models/common/HTCModel.py) which you can use to create a model instance and initialize it with the pretrained weights. The model object will be an instance of `torch.nn.Module`. The function has examples for all the different model types but as a teaser consider the following example which loads the pretrained image HSI network:

```python
import torch
from htc import ModelImage, Normalization

run_folder = "2022-02-03_22-58-44_generated_default_model_comparison"  # HSI model
model = ModelImage.pretrained_model(model="image", run_folder=run_folder, n_channels=100, n_classes=19)
input_data = torch.randn(1, 100, 480, 640)  # NCHW
input_data = Normalization(channel_dim=1)(input_data)  # Model expects L1 normalized input
model(input_data).shape
# torch.Size([1, 19, 480, 640])
```

> 💡 Please note that when initializing the weights as in this example, the segmentation head is initialized randomly. Meaningful predictions on your own data can thus not be expected out of the box, but you will have to train the model on your data first.

### Option 2: Use the models to create predictions for your data

The models can be used to predict segmentation masks for your data. The segmentation models automatically sample from your input image according to the selected model spatial granularity (e.g. by creating patches) and the output is always a segmentation mask for an entire image. The set of output classes is determined by the training configuration, e.g. 18 organ classes + background for our semantic models. There are two alternatives for creating predictions:

1. The [`CreatingPredictions`](./tutorials/CreatingPredictions.ipynb) notebook shows how to create predictions for all images in a folder (via the `htc inference` command) and how to map the network output to meaningful label names.
2. If you want to compute predictions directly within your code for custom tensors, batches or paths, you can use the [`SinglePredictor`](./htc/model_processing/SinglePredictor.py) class.

### Option 3: Use the models to train a network with the `htc` package

If you are using the `htc` framework to [train your networks](./tutorials/network_training/NetworkTraining.ipynb), you only need to define the model in your configuration:

```json
{
    "model": {
        "pretrained_model": {
            "model": "image",
            "run_folder": "2022-02-03_22-58-44_generated_default_model_comparison"
        }
    }
}
```

This is very similar to option 1 but may be more convenient if you already train with the `htc` framework.

> 💡 We have a [JSON Schema file](./htc/utils/config.schema) which describes the structure of our config files including descriptions of the attributes.

### Prediction Tables

The above options will only make predictions for your data without evaluating those predictions. If you want to compare the predictions against some reference annotations and compute some metric values (e.g., DSC), then there are two possibilities:

1. Use the `htc tables --input-dir YOUR_INPUT_DIR --output-dir YOUR_OUTPUT_DIR --test --metrics DSC --model image --run-folder 2023-02-08_14-48-02_organ_transplantation_0.8` command to create predictions for a set of paths (cf. `htc tables --help` for more information on the arguments). This will save a validation or test table on disk.
2. Similar to the `SinglePredictor` class from before, there is also a [`SinglePredictionTable`](./htc/model_processing/SinglePredictionTable.py) class to compute a metrics table for your data within your own code (the validation or test table will be returned instead of saved to disk). This is useful if you only want to compute metrics for a custom set of paths.

In any case, it is required that reference segmentations are available for every image so that the output of the network can be compared against something.

## CLI

There is a common command line interface for many scripts in this repository. More precisely, every script which is prefixed with `run_NAME.py` can also be run via `htc NAME` from any directory. For more details, just type `htc`.

## Papers

This repository contains code to reproduce our publications listed below:

### 📝 [Xeno-learning: knowledge transfer across species in deep learning-based spectral image analysis](https://doi.org/10.48550/arXiv.2410.19789)

<div align="center">
<a href="https://e130-hyperspectal-tissue-classification.s3.dkfz.de/figures/species_motivation.pdf"><img src="https://e130-hyperspectal-tissue-classification.s3.dkfz.de/figures/species_motivation.png" alt="Logo" width="700" /></a>
</div>

This paper introduces a cross-species knowledge transfer paradigm termed <i>xeno-learning</i> to make use of what has been learned in one species in other species. Specifically, we showcase how human segmentation performance on malperfused tissues can be improved by leveraging perfusion knowledge obtained from animal data via a <q>physiology-based data augmentation</q> method. All trained networks are available as pretrained models (baseline networks and networks which included the new data augmentation method during training). Compared to previous papers, we switched to a nested cross-validation scheme with 3 outer folds so each training configuration is composed of three run folders on disk. However, you can still refer to them via the `run_folder` argument by using a wildcard (e.g., `2024-09-11_00-11-38_baseline_human_nested-*-2` to get the baseline networks `0`, `1` and `2` trained on human data). You can find all notebooks which generate the paper figures in [paper/XenoLearning2024](./paper/XenoLearning2024) accompanied by [reproducibility instructions](./paper/XenoLearning2024/reproducibility.md). The code for all experiments is located in the [htc_projects/species](./htc_projects/species/) folder.

> 📂 The dataset for this paper is not fully publicly available, but a subset of the data is available through the public [HeiPorSPECTRAL](https://heiporspectral.org/) dataset.

<details closed>
<summary>Cite via BibTeX</summary>

```bibtex
@misc{sellner_species_2024,
  author      = {Sellner, Jan and Studier-Fischer, Alexander and Qasim, Ahmad Bin and Seidlitz, Silvia and Schreck, Nicholas and Tizabi, Minu and Wiesenfarth, Manuel and Kopp-Schneider, Annette and Knödler, Samuel and Haney, Caelan Max and Salg, Gabriel and Özdemir, Berkin and Dietrich, Maximilian and Michel, Maurice Stephan and Nickel, Felix and Kowalewski, Karl-Friedrich and Maier-Hein, Lena},
  url         = {https://arxiv.org/abs/2410.19789},
  date        = {2024},
  doi         = {10.48550/arXiv.2410.19789},
  eprint      = {2410.19789},
  eprintclass = {cs.CV},
  eprinttype  = {arXiv},
  title       = {Xeno-learning: knowledge transfer across species in deep learning-based spectral image analysis},
}
```

</details>

### 📝 [Semantic segmentation of surgical hyperspectral images under geometric domain shifts](https://doi.org/10.1007/978-3-031-43996-4_59)

<div align="center">
<a href="https://e130-hyperspectal-tissue-classification.s3.dkfz.de/figures/MICCAI_abstract.pdf"><img src="https://e130-hyperspectal-tissue-classification.s3.dkfz.de/figures/MICCAI_abstract.png" alt="Logo" width="600" /></a>
</div>

This MICCAI2023 paper is the direct successor of our MIA2022 paper. We analyzed how well our networks perform under geometrical domain shifts which commonly occur in real-world open surgeries (e.g. situs occlusions). The effect is drastic (drop of Dice similarity coefficient by 45 %) but the good news is that performance on par with in-distribution data can be achieved with our simple, model-independent solution (augmentation method). You can find all the code in [htc_projects/context](./htc_projects/context) and paper figures as well as [reproducibility instructions](./paper/MICCAI2023/reproducibility.md) in [paper/MICCAI2023](./paper/MICCAI2023). Pretrained models are available for our organ transplantation networks with HSI and RGB modalities.

> 💡 If you are only interested in our data augmentation method, you can also head over to [Kornia](https://github.com/kornia/kornia) where this augmentation is implemented for generic use cases (including 2D and 3D data). You will find it under the name [`RandomTransplantation`](https://kornia.readthedocs.io/en/latest/augmentation.module.html#kornia.augmentation.RandomTransplantation).

> 📂 The dataset for this paper is not publicly available.

<details closed>
<summary>Cite via BibTeX</summary>

```bibtex
@inproceedings{sellner_context_2023,
  author    = {Sellner, Jan and Seidlitz, Silvia and Studier-Fischer, Alexander and Motta, Alessandro and Özdemir, Berkin and Müller-Stich, Beat Peter and Nickel, Felix and Maier-Hein, Lena},
  editor    = {Greenspan, Hayit and Madabhushi, Anant and Mousavi, Parvin and Salcudean, Septimiu and Duncan, James and Syeda-Mahmood, Tanveer and Taylor, Russell},
  location  = {Cham},
  publisher = {Springer Nature Switzerland},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023},
  date      = {2023},
  doi       = {10.1007/978-3-031-43996-4_59},
  isbn      = {978-3-031-43996-4},
  pages     = {618--627},
  title     = {Semantic Segmentation of Surgical Hyperspectral Images Under Geometric Domain Shifts},
}
```

</details>

### 📝 [Robust deep learning-based semantic organ segmentation in hyperspectral images](https://doi.org/10.1016/j.media.2022.102488)

<div align="center">
<a href="https://e130-hyperspectal-tissue-classification.s3.dkfz.de/figures/MIA_abstract.pdf"><img src="https://e130-hyperspectal-tissue-classification.s3.dkfz.de/figures/MIA_abstract.png" alt="Logo" width="800" /></a>
</div>

In this paper, we tackled fully automatic organ segmentation and compared deep learning models on different spatial granularities (e.g. patch vs. image) and modalities (e.g. HSI vs. RGB). Furthermore, we studied the required amount of training data and the generalization capabilities of our models across subjects. The pretrained networks are related to this paper. You can find the notebooks to generate the paper figures in [paper/MIA2022](./paper/MIA2022) (the folder also includes a [reproducibility document](./paper/MIA2022/reproducibility.md)) and the models in [htc/models](./htc/models). For each model, there are three configuration files, namely `default`, `default_rgb` and `default_parameters`, which correspond to the HSI, RGB and TPI modality, respectively. You can also download the [NSD thresholds](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/nsd_thresholds_semantic.csv) which we used for the NSD metric (cf. Fig. 12).

> 📂 The dataset for this paper is not publicly available.

<details closed>
<summary>Cite via BibTeX</summary>

```bibtex
@article{seidlitz_seg_2022,
  author       = {Seidlitz, Silvia and Sellner, Jan and Odenthal, Jan and Özdemir, Berkin and Studier-Fischer, Alexander and Knödler, Samuel and Ayala, Leonardo and Adler, Tim J. and Kenngott, Hannes G. and Tizabi, Minu and Wagner, Martin and Nickel, Felix and Müller-Stich, Beat P. and Maier-Hein, Lena},
  date         = {2022-08},
  doi          = {10.1016/j.media.2022.102488},
  issn         = {1361-8415},
  journaltitle = {Medical Image Analysis},
  keywords     = {Hyperspectral imaging,Surgical data science,Deep learning,Open surgery,Organ segmentation,Semantic scene segmentation},
  pages        = {102488},
  title        = {Robust deep learning-based semantic organ segmentation in hyperspectral images},
  volume       = {80},
}
```

</details>

### 📝 [Dealing with I/O bottlenecks in high-throughput model training](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/figures/PyTorchConference_Poster.pdf)

The poster was presented at the PyTorch Conference 2023 and presents several solutions to improve data loading for faster network training. This originated from our MICCAI2023 paper, where we load huge amount of data while using a relatively small network resulting in GPU idle times when the GPU has to wait for the CPU to deliver new data. This requested the need for fast data loading strategies so that the CPU delivers data in-time for the GPU. The solutions include (1) efficient data storage via [Blosc](https://www.blosc.org/) compression, (2) appropriate precision settings, (3) GPU instead of CPU augmentations using the [Kornia](https://kornia.readthedocs.io) library and (4) a fixed shared pinned memory buffer for efficient data transfer to the GPU. For the last part, you will find the relevant code to create the buffer in this repository as part of the [SharedMemoryDatasetMixin](./htc/models/common/SharedMemoryDatasetMixin.py) class (`_add_tensor_shared()` method).

You can find the code to generate the results figures of the poster in [paper/PyTorchConf2023](./paper/PyTorchConf2023) including [reproducibility instructions](./paper/PyTorchConf2023/reproducibility.md). The experiment code can be found in the project folder [htc_projects/benchmarking](./htc_projects/benchmarking).

> 📂 The dataset for this poster is not publicly available.

<details closed>
<summary>Cite via BibTeX</summary>

```bibtex
@misc{sellner_benchmarking_2023,
  author       = {Sellner, Jan and Seidlitz, Silvia and Maier-Hein, Lena},
  url          = {https://e130-hyperspectal-tissue-classification.s3.dkfz.de/figures/PyTorchConference_Poster.pdf},
  date         = {2023-10-16},
  howpublished = {Poster presented at the PyTorch Conference 2023, San Francisco, United States of America},
  title        = {Dealing with I/O bottlenecks in high-throughput model training},
}
```

</details>

### 📝 [HeiPorSPECTRAL - the Heidelberg Porcine HyperSPECTRAL Imaging Dataset of 20 Physiological Organs](https://doi.org/10.1038/s41597-023-02315-8)

This paper introduces the [HeiPorSPECTRAL](https://heiporspectral.org/) dataset containing 5756 hyperspectral images from 11 subjects. We are using these images in our tutorials. You can find the visualization notebook for the paper figures in [paper/NatureData2023](./paper/NatureData2023) (the folder also includes a [reproducibility document](./paper/NatureData2023/reproducibility.md)) and the remaining code in [htc_projects/atlas_open](./htc_projects/atlas_open).

If you want to learn more about the [HeiPorSPECTRAL](https://heiporspectral.org/) dataset (e.g. the underlying data structure) or you stumbled upon a file and want to know how to read it, you might find this [notebook with low-level details](./htc_projects/atlas_open/FileReference.ipynb) helpful.

> 📂 The dataset for this paper is publicly available.

<details closed>
<summary>Cite via BibTeX</summary>

```bibtex
@article{studierfischer_open_2023,
  author       = {Studier-Fischer, Alexander and Seidlitz, Silvia and Sellner, Jan and Bressan, Marc and Özdemir, Berkin and Ayala, Leonardo and Odenthal, Jan and Knoedler, Samuel and Kowalewski, Karl-Friedrich and Haney, Caelan Max and Salg, Gabriel and Dietrich, Maximilian and Kenngott, Hannes and Gockel, Ines and Hackert, Thilo and Müller-Stich, Beat Peter and Maier-Hein, Lena and Nickel, Felix},
  url          = {https://heiporspectral.org},
  date         = {2023-06-24},
  doi          = {10.1038/s41597-023-02315-8},
  issn         = {2052-4463},
  journaltitle = {Scientific Data},
  number       = {1},
  pages        = {414},
  title        = {HeiPorSPECTRAL - the Heidelberg Porcine HyperSPECTRAL Imaging Dataset of 20 Physiological Organs},
  volume       = {10},
}
```

</details>

### 📝 [Spectral organ fingerprints for machine learning-based intraoperative tissue classification with hyperspectral imaging in a porcine model](https://doi.org/10.1038/s41598-022-15040-w)

In this paper, we trained a classification model based on median spectra from HSI data. You can find the model code in [htc_projects/atlas](./htc_projects/atlas) and the confusion matrix figure of the paper in [paper/NatureReports2022](./paper/NatureReports2022) (including a reproducibility document).

> 📂 The dataset for this paper is not fully publicly available, but a subset of the data is available through the public [HeiPorSPECTRAL](https://heiporspectral.org/) dataset.

<details closed>
<summary>Cite via BibTeX</summary>

```bibtex
@article{studierfischer_atlas_2022,
  author       = {Studier-Fischer, Alexander and Seidlitz, Silvia and Sellner, Jan and Özdemir, Berkin and Wiesenfarth, Manuel and Ayala, Leonardo and Odenthal, Jan and Knödler, Samuel and Kowalewski, Karl Friedrich and Haney, Caelan Max and Camplisson, Isabella and Dietrich, Maximilian and Schmidt, Karsten and Salg, Gabriel Alexander and Kenngott, Hannes Götz and Adler, Tim Julian and Schreck, Nicholas and Kopp-Schneider, Annette and Maier-Hein, Klaus and Maier-Hein, Lena and Müller-Stich, Beat Peter and Nickel, Felix},
  date         = {2022-06-30},
  doi          = {10.1038/s41598-022-15040-w},
  issn         = {2045-2322},
  journaltitle = {Scientific Reports},
  number       = {1},
  pages        = {11028},
  title        = {Spectral organ fingerprints for machine learning-based intraoperative tissue classification with hyperspectral imaging in a porcine model},
  volume       = {12},
}
```

</details>

### 📝 [Künstliche Intelligenz und hyperspektrale Bildgebung zur bildgestützten Assistenz in der minimal-invasiven Chirurgie](https://doi.org/10.1007/s00104-022-01677-w)

This paper presents several applications of intraoperative HSI, including our organ [segmentation](https://doi.org/10.1016/j.media.2022.102488) and [classification](https://doi.org/10.1038/s41598-022-15040-w) work. You can find the code generating our figure for this paper at [paper/Chirurg2022](./paper/Chirurg2022).

> 📂 The sample image used here is contained in the dataset from our paper [“Robust deep learning-based semantic organ segmentation in hyperspectral images”](https://doi.org/10.1016/j.media.2022.102488) and hence not publicly available.

<details closed>
<summary>Cite via BibTeX</summary>

```bibtex
@article{chalopin_chirurgie_2022,
  author       = {Chalopin, Claire and Nickel, Felix and Pfahl, Annekatrin and Köhler, Hannes and Maktabi, Marianne and Thieme, René and Sucher, Robert and Jansen-Winkeln, Boris and Studier-Fischer, Alexander and Seidlitz, Silvia and Maier-Hein, Lena and Neumuth, Thomas and Melzer, Andreas and Müller-Stich, Beat Peter and Gockel, Ines},
  date         = {2022-10-01},
  doi          = {10.1007/s00104-022-01677-w},
  issn         = {2731-698X},
  journaltitle = {Die Chirurgie},
  number       = {10},
  pages        = {940--947},
  title        = {Künstliche Intelligenz und hyperspektrale Bildgebung zur bildgestützten Assistenz in der minimal-invasiven Chirurgie},
  volume       = {93},
}
```

</details>

## Funding

This project has received funding from the European Research Council (ERC) under the European Unions Horizon 2020 research and innovation programme (NEURAL SPICING, grant agreement No. 101002198) and was supported by the German Cancer Research Center (DKFZ) and the Helmholtz Association under the joint research school HIDSS4Health (Helmholtz Information and Data Science School for Health). It further received funding from the Surgical Oncology Program of the National Center for Tumor Diseases (NCT) Heidelberg.
