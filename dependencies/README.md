# Dependencies

This folder contains all dependency files to get the htc project running.

The `requirements*.txt` list all the Python dependencies and are installed automatically if you followed the installation instructions. Despite the general dependencies (`requirements.txt`, 'requirements-extra.txt' and 'requirements-dev.txt') there may also be additional project-dependent dependencies which are not installed automatically but which may be required to run the project (and hence need to be installed manually.).

The Docker files define how to run the htc project in an isolated Docker container. From a user perspective, it is usually enough to run `htc docker bash` to get an isolated container (with mounted read-only results folder and datasets). The Docker files are mainly used for testing.
