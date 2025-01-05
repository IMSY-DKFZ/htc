# Tests

This folder contains all tests for our repository. We are using the [pytest](https://docs.pytest.org/en/stable/) test framework. In addition to the tests defined here, all notebooks in this repository are regularly tested by running each cell and ensuring that no error occurs.

> &#x26a0;&#xfe0f; Please note that many tests rely on access to internal datasets and hence can only be run if those datasets are available. For the public version of this repository, the tests serve as examples to use our functions.

## Running Tests

We are using Docker to run the tests in a fresh environment. This includes an isolated results directory so that your locally stored results are not altered by accident. What is more, the network drive, the data and network results directories and your local result directories are mounted as read-only so that you cannot accidentally modify them. To run the test suit locally, you may use

```bash
# Full test suit (may take 1 to 2 hours)
screen -S htc_tests -d -m script -q -c "htc docker htc tests --slow --notebooks --parallel 4" test_docker.log

# Quick tests (should be finished in around 15 minutes)
screen -S htc_tests -d -m script -q -c "htc docker htc tests --parallel 4" test_docker.log
```

This will run the tests in a separate screen environment and logs the output to the file `test_docker.log` (with color support) which you can later read again via

```bash
cat test_docker.log
```

If you want to run doctests for a single file/folder, you can use

```bash
py.test --doctest-modules PATH_TO_FILE.py
```

> ⚠️ Since the tests have access to your local datasets and results directories, they won't necessarily pass on other machines (including our GitLab runner) if the tests rely on changes you made to a dataset or a results directory. Hence, make sure that you sync changes you made back to the network drive. Per default, all `results_*` folders in `/mnt/E130-Projekte/Biophotonics/Projects/2021_02_05_hyperspectral_tissue_classification` will automatically be available during testing so make sure you regularly sync your local result directories to this folder on the network drive.

## Folder Structure

In general, the folder structure here mirrors the folder structure of the htc folder. Please put all project-related tests from htc_projects to the [projects](./projects/) folder (with a subfolder for the project name).
