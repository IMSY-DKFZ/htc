# Reproducibility (DKFZ internal only)

This document will guide you through the process of reproducing the main results for our xeno-learning paper.

## Setup

Start by installing the [repository](https://git.dkfz.de/imsy/issi/htc) according to the [README](../../README.md).

This work makes use of many different datasets. Your `.env` file should contain at least the following datasets:

```bash
export DKFZ_USERID=j562r
export PATH_E130_Projekte=/mnt/E130-Projekte
export PATH_HTC_RESULTS=~/htc/results

# Datasets
export PATH_Tivita_multiorgan_kidney=~/htc/2023_04_22_Tivita_multiorgan_kidney
export PATH_Tivita_multiorgan_masks=~/htc/2021_02_05_Tivita_multiorgan_masks
export PATH_Tivita_multiorgan_semantic=~/htc/2021_02_05_Tivita_multiorgan_semantic
export PATH_Tivita_multiorgan_human=~/htc/2021_07_26_Tivita_multiorgan_human
export PATH_Tivita_multiorgan_rat=~/htc/2023_12_07_Tivita_multiorgan_rat
```

> If you are already using this repository, it is recommend to clone it to a new folder and use a fresh conda environment. Existing results folder should not be available for the reproduction.

You also need access to the cluster, i.e. `ssh $DKFZ_USERID@$WORKER_NODE` should work (cf. our [cluster documentation](../../htc/cluster/cluster_usage.md) for more details).

> Please use a `screen` environment for all of the following commands since they may take a while to complete.

## Sanity check

The figures for this paper are created via the notebooks in [`paper/XenoLearning2024`](../XenoLearning2024). Run the [paper/XenoLearning2024/DomainShiftPerformance.ipynb](../XenoLearning2024/DomainShiftPerformance.ipynb) notebook now and you should get an error about missing paths.

## Data specification files

The paper contains many different experiments comparing various combinations of our datasets and employs a nested cross-validation scheme. To represent these experiments, we need a lot of data specification files. Create them by running the following commands:

```bash
htc pig_semantic_nested_dataset
htc rat_semantic_dataset
htc human_physiological_dataset
htc human_physiological_dataset --additional-species pig-p rat-p
```

## Xeno-learning: in-species learning

The first step of our xeno-learning approach is to learn how perfusion shifts look like in the source species so that this knowledge can later be applied in the target species. Run the following command to learn how to transform physiological spectra to malperfused spectra separately for each species:

```bash
htc create_projections
```

Apply the projections on some example spectra (this data will be used in the PCA later)

```bash
htc transform_spectra
```

## Training

We can now train our baseline networks and the networks which utilize our physiology-based data augmentation. The data augmentation will use the previously created projection files. Run the following command to submit everything to the cluster:

```bash
htc species_models
```

After all jobs are finished, copy the results from the cluster and combine the results for the individual folds

```bash
htc move_results
htc table_generation
```

## Inference

The trained networks are stored in `$PATH_HTC_RESULTS/training/image` and all run directories will start with the same timestamp (e.g., `2024-09-11_00-11-38`). Use your new timestamp to run inference on the test datasets for all trained networks

```bash
htc physiological_scores --timestamp "<YOUR_TIMESTAMP>"
htc perfusion_scores --timestamp "<YOUR_TIMESTAMP>"
```

## Figures

You now have all ingredients together to create the final figures. Run the following commands to produce the figures which utilize the trained networks

```bash
HTC_MODEL_TIMESTAMP="<YOUR_TIMESTAMP>" jupyter nbconvert --to html --execute --stdout paper/XenoLearning2024/DomainShiftPerformance.ipynb > /dev/null
HTC_MODEL_TIMESTAMP="<YOUR_TIMESTAMP>" jupyter nbconvert --to html --execute --stdout paper/XenoLearning2024/PerfusionPerformance.ipynb > /dev/null
```

You will find the resulting figures in `$PATH_HTC_RESULTS/paper`. You can run the other notebooks in the same way.
