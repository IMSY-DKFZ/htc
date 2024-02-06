# Reproducibility (DKFZ internal only)
This document will guide you through the process of reproducing the results for our [benchmarking poster](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/figures/PyTorchConference_Poster.pdf) presented at the PyTorch Conference 2023.

## Setup
Start by installing the [repository](https://git.dkfz.de/imsy/issi/htc) according to the [README](../../README.md).

> These instructions were tested on the `poster_benchmarking_v1` tag. However, for reproducing, we recommend to use the latest master instead as there are some general dependencies (e.g. dataset version, cluster access) which are not guaranteed to work on an old tag in the future.
Further, the following hardware was used for reproducing (different setups will yield different runtimes and hence different results):
>* GPU: NVIDIA GeForce RTX 4090
>* CPU: AMD Ryzen 9 7950X
>* SSD: Seagate FireCuda 530 4 TB
>* Storage space requirements for the dataset: at least 250 GiB

You need the semantic datasets for this poster. Please copy it from the network drive to your computer, for example via: [≈ 1–2 hours]
```bash
rsync -a --delete --info=progress2 --exclude=".git" /mnt/E130-Projekte/Biophotonics/Data/2021_02_05_Tivita_multiorgan_semantic/ ~/htc/2021_02_05_Tivita_multiorgan_semantic/
```

Additionally, please create an empty results folder. In the end, your `.env` may look like:
```bash
export PATH_E130_Projekte=/mnt/E130-Projekte
export PATH_Tivita_multiorgan_semantic=~/htc/2021_02_05_Tivita_multiorgan_semantic
export PATH_HTC_RESULTS=~/htc/results
```

> Please use a [`screen`](https://linuxize.com/post/how-to-use-linux-screen/) environment for all of the following commands since they may take a while to complete.

## Sanity check
The result figures are created via the [Benchmarking.ipynb](./Benchmarking.ipynb) notebook. If you run it now, you should get an error about missing paths since the trained networks are not available yet.

## Preprocessing
For this project, we need the raw32 data, i.e. compressed float32 data. This is not part of the semantic dataset so we need to create it first with the following command: [≈ 10 minutes]
```bash
htc raw16 --dataset-name 2021_02_05_Tivita_multiorgan_semantic --precision 32 --spec benchmarking/data/pigs_semantic-all_train-only.json
```

> The corresponding folder `$PATH_Tivita_multiorgan_semantic/intermediates/preprocessing/raw32` can be deleted if this reproducibility is finished since we usually don't need it.

## Benchmarking
> Please make sure that no other heavy task is running on the computer while the benchmark is running (e.g. by running the below command over night).

Run the benchmark with the following command: [≈ 4–6 hours]
```bash
htc benchmark --io-speed "unlimited" "1000mb"
```

This creates a training run for each loading strategy (original, blosc, fp16, gpu-aug and ring-buffer), repetition (0, 1, 2) and io speed limit (1000mb and unlimited). The networks are trained for several epochs using all the available semantic data and the time for each epoch is measured. The run folders do not store the trained networks (since they are not needed) but only the training logs and the time measurements.

The networks are stored in `$PATH_HTC_RESULTS/training/image` and they all share the same timestamp (filename prefix). Please note down this timestamp as you will need it in the next step (e.g. `2023-09-03_22-48-13`).

> The benchmarks run in a Docker container. Hence, all created files inside the container are owned by the root user. To change that, simply run: `sudo chown -R $USER:$USER $PATH_HTC_RESULTS` after the benchmarking finished.

## Figures
You can now create the result figures from the poster with your newly created training runs by executing the following notebook: [≈ 2 minutes]
```bash
HTC_BENCHMARKING_TIMESTAMP="<YOUR_TIMESTAMP>" jupyter nbconvert --to html --execute --stdout ~/htc/src/paper/PyTorchConf2023/Benchmarking.ipynb > /dev/null
```

The figures are stored in `$PATH_HTC_RESULTS/paper`. The results cannot be expected to be exactly the same as in the poster (and depend on the used hardware of course) but as long as they are similar, everything is good :-)
