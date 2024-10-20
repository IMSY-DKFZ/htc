# Reproducibility (DKFZ internal only)

This document will guide you through the process of reproducing the main results for our [context paper](https://arxiv.org/abs/2303.10972). That is, based on the pretrained networks from our [MIA paper](https://doi.org/10.1016/j.media.2022.102488), we will re-create Fig. 3 and Fig. 4.

## Setup

Start by installing the [repository](https://git.dkfz.de/imsy/issi/htc) according to the [README](../../README.md).

> These instructions were tested on the `paper_context_v1` tag. However, for reproducing, we recommend to use the latest master instead as there are some general dependencies (e.g. dataset version, cluster access) which are not guaranteed to work on an old tag in the future.

You also need the semantic, masks and unsorted datasets. Please copy them from `E130-Projekte/Biophotonics/Data` to your computer if not already done (e.g. `rsync -a --delete --info=progress2 --exclude=".git*" /mnt/E130-Projekte/Biophotonics/Data/2021_02_05_Tivita_multiorgan_semantic/ ~/htc/2021_02_05_Tivita_multiorgan_semantic/`). Additionally, please create an empty results folder. In the end, your `.env` may look like:

```bash
export DKFZ_USERID=j562r
export PATH_E130_Projekte=/mnt/E130-Projekte
export PATH_Tivita_multiorgan_semantic=~/htc/2021_02_05_Tivita_multiorgan_semantic
export PATH_Tivita_multiorgan_masks=~/htc/2021_02_05_Tivita_multiorgan_masks
export PATH_Tivita_unsorted_images=~/htc/2022_08_03_Tivita_unsorted_images
export PATH_HTC_RESULTS=~/htc/results
```

> If you are already using this repository, it is recommend to clone it to a new folder and use a fresh conda environment. Existing results folder should not be available for the reproduction.

You also need access to the cluster, i.e. `ssh $DKFZ_USERID@$WORKER_NODE` should work (cf. our [cluster documentation](../../htc/cluster/cluster_usage.md) for more details). You may want to run the (short) tests to ensure that everything works (cf. [README](../../README.md)).

> Please use a [`screen`](https://linuxize.com/post/how-to-use-linux-screen/) environment for all of the following commands since they may take a while to complete.

## Sanity check

The two main figures of the paper are created via the [TaskPerformanceComparison.ipynb](./TaskPerformanceComparison.ipynb) and [BootstrapRankingBubblePlots.ipynb](./BootstrapRankingBubblePlots.ipynb) notebooks. Please run them now and ensure that you get some error about missing paths since the trained networks should not be accessible yet.

## Prepare the baseline networks

The context paper builds upon our semantic organ segmentation work. We first need to prepare the results for the old networks which consists of two parts: (1) baseline performance on the glove dataset (a new split based on the old data) and (2) results of the baseline networks for the different datasets. For the first part, submit the corresponding jobs to the cluster [≈ 2–4 hours]

```bash
htc glove_baseline_runs
```

After all jobs succeeded (you will receive mails about that), copy the trained models from the cluster and combine the results from the different folds [≈ 5 minutes]

```bash
# Copy results form the cluster
htc move_results

# Combine results and generate figures
htc table_generation
```

The trained networks are stored in `$PATH_HTC_RESULTS/training/image`. The [settings_context.py](../../htc_projects/context/settings_context.py) file lists all the networks which are used for our paper. During the reproducibility, you need to change the run folder names with the updated names from the new training runs. For now, please update the `baseline` network names of the `glove_runs` and `glove_runs_rgb` properties since you just re-trained them.

> Please do not commit changes to the [settings_context.py](../../htc_projects/context/settings_context.py) file which are due to the run folder name changes. This is just for the reproducibility.

For the second part, we need to run some inference tasks on the existing MIA2022 networks as well as your freshly trained glove runs (HSI and RGB image models) [≈ 1 day]:

```bash
htc baseline_tables
```

The results will be stored in `$PATH_HTC_RESULTS/neighbour_analysis` with a subfolder for each transformation.

## Training

Let's start with the reproduction steps of the context transformations and re-train the networks using different augmentations. Submit the corresponding jobs (14 in total) to the cluster via: [≈ 10 hours]

```bash
htc transform_runs --config context/models/configs/context.json context/models/configs/context-glove.json --best --include-rgb
```

> We do not reproduce the grid search of the augmentation probability parameter $p$ to reduce the required number of training runs.

Similar to before, after all jobs succeeded, copy the trained models from the cluster and combine the results from the different folds [≈ 5 minutes]

```bash
htc move_results
htc table_generation --notebook "htc_projects/context/models/ExperimentAnalysis.ipynb"
```

The trained networks are stored in `$PATH_HTC_RESULTS/training/image`. Every folder contains an `ExperimentAnalysis.html` notebook (which is different to the previous training runs) with some statistics of the trained run. Feel free to take a look.

You have now re-trained all networks of our paper. Please update all remaining run folder names in [settings_context.py](../../htc_projects/context/settings_context.py) of the properties `best_transform_runs`, `best_transform_runs_rgb`, `glove_runs` and `glove_runs_rgb` (everything except the `baseline` runs because you already updated those properties in the previous step). In the end, you should have changed the following properties in the [settings_context.py](../../htc_projects/context/settings_context.py) file:

<details>
<summary>Show changes</summary>

```diff
src » git diff htc_projects/context/settings_context.py                                                                                  ~/htc/src
diff --git a/htc_projects/context/settings_context.py b/htc_projects/context/settings_context.py
index 4c207d7d..b478c9ad 100644
--- a/htc_projects/context/settings_context.py
+++ b/htc_projects/context/settings_context.py
@@ -287,40 +287,40 @@ class SettingContext:
     def best_transform_runs(self) -> dict[str, MultiPath]:
         # Best runs for each transformation (found via find_best_transform_run())
         return {
-            "organ_transplantation": settings.training_dir / "image/2023-02-08_14-48-02_organ_transplantation_0.8",
-            "cut_mix": settings.training_dir / "image/2023-02-08_17-08-57_cut_mix_1",
-            "jigsaw": settings.training_dir / "image/2023-02-16_21-17-59_jigsaw_0.8",
-            "random_erasing": settings.training_dir / "image/2023-02-08_12-06-44_random_erasing_0.4",
-            "hide_and_seek": settings.training_dir / "image/2023-02-16_15-34-51_hide_and_seek_1",
-            "elastic": settings.training_dir / "image/2023-02-08_09-40-59_elastic_0.6",
+            "organ_transplantation": settings.training_dir / "image/2023-07-08_15-55-10_context_organ_transplantation_0.8",
+            "cut_mix": settings.training_dir / "image/2023-07-08_15-55-10_context_cut_mix_1.0",
+            "jigsaw": settings.training_dir / "image/2023-07-08_15-55-10_context_jigsaw_0.8",
+            "random_erasing": settings.training_dir / "image/2023-07-08_15-55-10_context_random_erasing_0.4",
+            "hide_and_seek": settings.training_dir / "image/2023-07-08_15-55-10_context_hide_and_seek_1.0",
+            "elastic": settings.training_dir / "image/2023-07-08_15-55-10_context_elastic_0.6",
         }

     @property
     def best_transform_runs_rgb(self) -> dict[str, MultiPath]:
         return {
-            "organ_transplantation": settings.training_dir / "image/2023-01-29_11-31-04_organ_transplantation_0.8_rgb",
+            "organ_transplantation": settings.training_dir / "image/2023-07-08_15-55-10_context_organ_transplantation_rgb_0.8",
         }

     @property
     def glove_runs(self) -> dict[str, MultiPath]:
         return {
-            "baseline": settings.training_dir / "image/2023-02-21_23-14-44_glove_baseline",
+            "baseline": settings.training_dir / "image/2023-06-29_18-00-40_default_glove",
             "organ_transplantation": (
-                settings.training_dir / "image/2023-02-21_23-14-55_glove_organ_transplantation_0.8"
+                settings.training_dir / "image/2023-07-08_15-55-10_context-glove_organ_transplantation_0.8"
             ),
-            "cut_mix": settings.training_dir / "image/2023-02-23_19-07-27_glove_cut_mix_1.0",
-            "jigsaw": settings.training_dir / "image/2023-02-22_12-31-26_glove_jigsaw_0.8",
-            "elastic": settings.training_dir / "image/2023-02-22_12-31-26_glove_elastic_0.6",
-            "random_erasing": settings.training_dir / "image/2023-02-22_12-31-26_glove_random_erasing_0.4",
-            "hide_and_seek": settings.training_dir / "image/2023-02-22_12-31-26_glove_hide_and_seek_1.0",
+            "cut_mix": settings.training_dir / "image/2023-07-08_15-55-10_context-glove_cut_mix_1.0",
+            "jigsaw": settings.training_dir / "image/2023-07-08_15-55-10_context-glove_jigsaw_0.8",
+            "elastic": settings.training_dir / "image/2023-07-08_15-55-10_context-glove_elastic_0.6",
+            "random_erasing": settings.training_dir / "image/2023-07-08_15-55-10_context-glove_random_erasing_0.4",
+            "hide_and_seek": settings.training_dir / "image/2023-07-08_15-55-10_context-glove_hide_and_seek_1.0",
         }

     @property
     def glove_runs_rgb(self) -> dict[str, MultiPath]:
         return {
-            "baseline": settings.training_dir / "image/2023-02-24_12-07-15_glove_baseline_rgb",
+            "baseline": settings.training_dir / "image/2023-06-29_18-00-40_default_rgb_glove",
             "organ_transplantation": (
-                settings.training_dir / "image/2023-02-24_14-27-15_glove_organ_transplantation_0.8_rgb"
+                settings.training_dir / "image/2023-07-08_15-55-10_context-glove_organ_transplantation_rgb_0.8"
             ),
         }
```

</details>

## Test inference

During training, only results on the validation data were computed. To compute the test results for the context networks, please run [≈ 1 day]

```bash
htc context_test_tables
```

The results will be stored in the corresponding run directories (e.g. `test_table_isolation_0.pkl.xz`) in your training folder (`$PATH_HTC_RESULTS/training/image`).

## Figures

You now have all ingredients together to re-create the figures by executing the following two notebooks [≈ 5 minutes]

```bash
jupyter nbconvert --to html --execute --stdout ~/htc/src/paper/MICCAI2023/TaskPerformanceComparison.ipynb > /dev/null
jupyter nbconvert --to html --execute --stdout ~/htc/src/paper/MICCAI2023/BootstrapRankingBubblePlots.ipynb > /dev/null
```

The results are stored in `$PATH_HTC_RESULTS/paper`. Due to non-determinism in our machine learning, the results cannot be expected to be exactly the same, but as long as the results support the findings from our paper, everything is good :o)
