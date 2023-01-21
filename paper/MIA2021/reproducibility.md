# Reproducibility (DKFZ internal only)
This document will guide you through the process of reproducing the main results for our [semantic organ segmentation paper](https://arxiv.org/abs/2111.05408). To reduce the number of required training runs, we are only reproducing the results for the spatial-spectral comparison (Fig. 5).

## (Optional) setup an OpenStack instance
> NB: You can also use your own computer if you cannot use an OpenStack instance.

If yo do not have access to OpenStack yet, you need to [request access](https://info.dkfz-heidelberg.de/itcf/UserPortal/Openstack/Antrag) first.
- Select `I apply for inclusion in an existing project`
- Project Name: `E130-GPU`

After access being granted, download the files from the `openstack` folder ([https://git.dkfz.de/imsy/issi/htc/-/tree/master/openstack](https://git.dkfz.de/imsy/issi/htc/-/tree/master/openstack)) and then set up the OpenStack instance (this will install some Python packages on your machine). Also make sure that you set up an SSH keypair with Phabricator
```bash
cd openstack
bash openstack.sh
```

If no errors occurred (you should see the message <span style="color:#327FBA">**Setup complete**</span>), then you have now a running instance with all the necessary tools installed.
```
You can now login to your instance (make sure you change the IP if it is different in your case)
```bash
ssh -i ~/.ssh/openstack_gpu ubuntu@10.133.25.126
```

From now on, please run every command on the OpenStack instance and not on your own computer. Please also use a [`screen`](https://linuxize.com/post/how-to-use-linux-screen/) environment for all commands since they may take a while to complete.

> NB: If you want access to the OpenStack files from you workstation, you can mount it via `sshfs`:
> ```bash
> mkdir ~/openstack
> sudo sshfs -o StrictHostKeyChecking=no,follow_symlinks,allow_other,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,cache_timeout=10,IdentityFile=/home/$USER/.ssh/openstack_gpu ubuntu@10.133.25.126:/ ~/openstack
> ```

> NB: In case you need to reboot the instance (usually not required), make sure to mount the network drives again:
> ```bash
> sudo mount /mnt/E130-Projekte
> ```

## Semantic paper repository
Start by installing the [repository](https://gitlab-test.dkfz.de/imsy/susi/htc) according to the [README](README.md). Make sure to check out the version of our semantic paper after the `git clone` step:
```bash
git checkout tags/paper_semantic_v3
```

## (Optional) run tests
If you like, you can run all the tests (some tests may be skipped) [≈ 1 hour]
```bash
htc tests --slow --parallel 4
```
However, the tests depend on files on the network drive and they may change so tests can fail in the future with the `paper_semantic_v3` tag. With the test `test_paper_files` you already reproduced all paper figures based on the trained models. We will now re-train the networks again to see whether we can still reproduce the main results.

## Start fresh
We want to train our networks again based on the raw data, so please delete the intermediate files
```bash
rmd ~/htc/2021_02_05_Tivita_multiorgan_semantic/intermediates
```

There are 20 pigs in total in the semantic dataset and the pigs `['P043', 'P046', 'P062', 'P068', 'P072']` are used as test set. Please move the corresponding pig folders (located in `~/htc/2021_02_05_Tivita_multiorgan_semantic/data/subjects`) somewhere else to a location only you know (but outside the repository). This ensures that the following training steps cannot accidentally access the test set.
```bash
for subject_name in P043 P046 P062 P068 P072; do mv ~/htc/2021_02_05_Tivita_multiorgan_semantic/data/subjects/$subject_name YOUR_SECRET_FOLDER/$subject_name; done
```

## Preprocessing
Create the preprocessed files by running the following scripts (this basically re-creates the intermediates) [≈ 10 minutes]
```bash
htc l1_normalization && htc median_spectra && htc parameter_images
```

## Training
Start the training runs with the following script. This will create and submit 75 cluster jobs. It is recommended that you have set up filters in your mailbox to ensure that mails from the cluster get sorted into their own folder. [≈ 1–2 days (depending on the cluster utilization)]
```bash
htc model_comparison
```

If all jobs are finished and succeeded successfully, copy the trained models from the cluster and combine the results from the different folds (some unimportant warnings may be raised) [≈ 20 minutes]
```bash
htc move_results
htc table_generation
```

All run folders are stored in `~/htc/results/training/(image|patch|superpixel_classification|pixel)` and there will be a `validation_table.pkl.xz` with all the validation results and an `ExperimentAnalysis.html` notebook with visualizations for each run. You also need the timestamp which was used for the runs later (e.g. `2022-02-03_22-58-44`). Every algorithm is prefixed with the same timestamp.

## Test inference
During training, we computed only validation results. It is now time to move the previously hidden test pigs back to the data folder and re-run the preprocessing steps from above [≈ 10 minutes]
```bash
for subject_name in P043 P046 P062 P068 P072; do mv YOUR_SECRET_FOLDER/$subject_name ~/htc/2021_02_05_Tivita_multiorgan_semantic/data/subjects/$subject_name; done
htc l1_normalization && htc median_spectra && htc parameter_images
```

For the NSD, we need to make the inter-rater results available (they are also shown in Fig. 5) [≈ 5 minutes]
```bash
htc nsd_thresholds
```

After this, it is finally time for the test predictions and validation [≈ 4 hour]
```bash
htc multiple --filter "<YOUR_TIMESTAMP>" --script "run_tables.py"
```

## Main results
It is now time to take a look at the final results. The main figures are produced by a notebook and you can generate a HTML version via [≈ 5 minutes]
```bash
HTC_MODEL_COMPARISON_TIMESTAMP="<YOUR_TIMESTAMP>" jupyter nbconvert --to html --execute --output-dir=~/htc ~/htc/src/paper/MIA2021/Benchmarking.ipynb
```
Fig. 5, Fig. 7 and Fig. 11 are directly shown in the notebook. Fig 6 is stored in `~/htc/results/paper/ranking_bootstrapped_test_dice_metric_image.pdf`. Due to non-determinism in our machine learning, the results cannot be expected to be exactly the same, but as long as the results are roughly similar to the paper, everything is good :-)
