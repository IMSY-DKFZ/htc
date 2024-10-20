# Reproducibility (DKFZ internal only)

This document lists the necessary steps to reproduce the additional results from our extended context paper. Since this paper builds upon our previous [context paper](https://arxiv.org/abs/2303.10972) (MICCAI2023), only the additional steps are listed here and it is assumed that all results from the previous paper have already been computed. If this is not the case, head over to the [reproducibility instructions](../MICCAI2023/reproducibility.md) of the context paper first. It is also assumed that the same results directory (with the corresponding runs from the context paper) is used.

## Setup

Same as in the previous [reproducibility instructions](../MICCAI2023/reproducibility.md).

## New networks

We need to train new networks for the lower spatial granularities (< image) for the occlusion scenario (glove runs). You can submit the new runs on the cluster via [≈ 1 day]:

```bash
htc glove_baseline_runs --model patch superpixel_classification pixel
```

After all jobs succeeded (you will receive mails about that), copy the trained models from the cluster and combine the results from the different folds [≈ 5 minutes]

```bash
# Copy results form the cluster
htc move_results

# Combine results and generate figures
htc table_generation
```

As before, the trained networks are listed in the [settings_context.py](../../htc_projects/context/settings_context.py) file. The new runs are stored in the properties `glove_runs_granularities` and `glove_runs_granularities_rgb`. Adapt the timestamps according to your newly trained networks.

> Similar to before, please do not commit changes to the [settings_context.py](../../htc_projects/context/settings_context.py) file which are due to the run folder name changes. This is just for the reproducibility.

## Test inference

We need to compute the tables with the test results for all lower spatial granularities and datasets. This can be achieved with the following commands which only compute tables if they don't exist yet [≈ 1 week]:

```bash
# Results on the isolation_real and occlusion datasets
htc baseline_tables

# Results on the simulated datasets for the lower spatial granularities
htc all_experiments --test
htc aggregate_tables
```

## Figures

All the new and updated figures can now be re-created with the notebooks in [this folder](../MIA2024/). You can for example run them via

```bash
jupyter nbconvert --to html --execute --stdout ~/htc/src/paper/MIA2024/SpatialGranularityComparison.ipynb > /dev/null
```

and you will find the figures in `$PATH_HTC_RESULTS/paper_extended`. As always, don't expect results to be exactly identical due to non-determinism etc.
