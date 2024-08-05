# Reproducibility (DKFZ internal only)

This document will guide you through the process of reproducing the main results for our [tissue atlas paper](https://doi.org/10.1038/s41598-022-15040-w), namely the confusion matrix of Figure 5 a.

## Setup

Start by installing the [repository](https://git.dkfz.de/imsy/issi/htc) according to the [README](../../README.md).

> The instructions of this document were tested on the `paper_tissue_atlas_v1` tag. However, for reproducing, we recommend to use the latest master instead as there are some general dependencies (e.g. dataset version, cluster access) which are not guaranteed to work on an old tag in the future. Further, the following hardware was used for reproducing:
>
> -   GPU: NVIDIA GeForce RTX 3090
> -   CPU: Intel(R) Xeon(R) W-2145 CPU @ 3.70GHz
> -   Storage space requirements for the dataset: at least 2.5 TiB

You need the masks datasets for this paper. Please copy it from the network drive to your computer, for example via: [≈ 12–16 hours]

```bash
rsync -a --delete --info=progress2 --exclude=".git" /mnt/E130-Projekte/Biophotonics/Data/2021_02_05_Tivita_multiorgan_masks/ ~/htc/2021_02_05_Tivita_multiorgan_masks/
```

Additionally, please create an empty results folder. In the end, your `.env` may look like:

```bash
export DKFZ_USERID=j562r
export PATH_E130_Projekte=/mnt/E130-Projekte
export PATH_Tivita_multiorgan_masks=~/htc/2021_02_05_Tivita_multiorgan_masks
export PATH_HTC_RESULTS=~/htc/results
```

> Please use a [`screen`](https://linuxize.com/post/how-to-use-linux-screen/) environment for all of the following commands since they may take a while to complete.

## Sanity check

The confusion matrix is created via the [ConfusionMatrix.ipynb](./ConfusionMatrix.ipynb) notebook. If you run it now, you should get an error about missing paths since the trained network is not available yet.

## Training

We will only re-train the best-performing network, i.e. we will not redo the complete grid search to reduce the required number of training runs. To start the training, you only need the following command: [≈ 4–6 hours]

```bash
htc training --model median_pixel --config tissue_atlas/median_pixel/configs/default.json --test
```

which takes the [default configuration](../../htc/tissue_atlas/median_pixel/configs/default.json) file and trains all folds subsequently.

After the training is complete, combine the results from the different folds including test fold ensembling [≈ 15 minutes]

```bash
htc tissue_atlas.test_table_generation
htc table_generation --notebook htc/tissue_atlas/ExperimentAnalysis.ipynb
```

The trained network is stored in `$PATH_HTC_RESULTS/training/median_pixel`. Every folder contains an `ExperimentAnalysis.html` notebook with some statistics of the trained run. Feel free to take a look.

The used network of the paper is defined in [settings_atlas.py](../../htc/tissue_atlas/settings_atlas.py). Please replace the variable `best_run` with the name of your training run folder (the folder name which starts with a timestamp).

## Figure

You can now create the confusion matrix from the paper with your newly created training run by executing the following notebook: [≈ 2 minutes]

```bash
jupyter nbconvert --to html --execute --stdout ~/htc/src/paper/NatureReports2022/ConfusionMatrix.ipynb > /dev/null
```

The results are stored in `$PATH_HTC_RESULTS/paper`. Due to non-determinism in our machine learning, the results cannot be expected to be exactly the same, but as long as the results are roughly similar to the paper, everything is good :-)
