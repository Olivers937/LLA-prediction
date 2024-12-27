# LLA-prediction documentation!

## Description

We use classic machine learning algorithm (modified) to predict if somebody is sick or not, by using pictures of his blood cells.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `az storage blob upload-batch -d` to recursively sync files in `data/` up to `container-name/data/`.
* `make sync_data_down` will use `az storage blob upload-batch -d` to recursively sync files from `container-name/data/` to `data/`.


