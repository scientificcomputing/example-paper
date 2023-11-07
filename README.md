# Supplementary code for the paper: Title of paper
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/scientificcomputing/example-paper/HEAD)

This repository contains supplementary code for the paper
> Finsberg, H., Dokken, J. 2022.
> Title of paper, Journal of ..., volume, page, url


## Abstract
Provide the abstract of the paper

## Getting started

We provide a pre-build Docker image which can be used to run the the code in this repository. First thing you need to do is to ensure that you have [docker installed](https://docs.docker.com/get-docker/).

To start an interactive docker container you can execute the following command

```bash
docker run --rm -it ghcr.io/scientificcomputing/example-paper:latest
```

### Pre-processing
Add steps for pre-processing, e.g

```
cd code
python3 pre-processing.py
```

### Running simulation
Add steps for running simulations, e.g

```
cd code
python3 run_all.py
```


### Postprocessing
Add steps for postprocessing / reproducing figures and tables in the paper, e.g

```
cd code
python3 postprocess.py
```

## Citation

```
@software{Lisa_My_Research_Software_2017,
  author = {Lisa, Mona and Bot, Hew},
  doi = {10.5281/zenodo.1234},
  month = {12},
  title = {{My Research Software}},
  url = {https://github.com/scientificcomputing/example-paper},
  version = {2.0.4},
  year = {2017}
}
```


## Having issues
If you have any troubles please file and issue in the GitHub repository.

## License
MIT
