# ConstraintMatch for Semi-constrained Clustering (SSCC)

Code accompanying the paper "[ConstraintMatch for Semi-constrained Clustering](https://arxiv.org/abs/2311.15395)", published at the 2023 International Joint Conference on Neural Networks (IJCNN). 

The codebase is licensed via the MIT license

## Setup

* create a `python=3.7` miniconda environment and activate that environment
* install the required packages via `pip install -r requirements.txt`
* install the local package `sscc` via `pip install -e .` from the main folder
* run the `get_pretrained_scan_weights.sh` script to retrieve all pretrained model weights from an anonymous GDrive
* test your install via `python run.py --config configs/intro.yaml` from the `experiments` level to
    * download the cifar10 dataset in a new `experiments/data/` folder and sample constraints accordingly over the folds
    * train ConstraintMatch on in this setting with pre-training (loading previously downloaded model weights) for 100 training steps
    * store the results via `mlflow` logging 

## Experiments 

We work with config files in this codebase (stored in `experiments/configs/`). These allow you to run different models in different settings on different datasets. See some examples below (all model runs executed from the `sscc/experiments` level): 

ConstraintMatch:

* `python run.py --config configs/constraintmatch_cifar10.yaml` for ConstraintMatch on Cifar10 with 10k constraints
* `python run.py --config configs/constraintmatch_stl10.yaml` for ConstraintMatch on STL-10 with 10k constraints
* `python run.py --config configs/constraintmatch_overcluster_cifar20.yaml` for ConstraintMatch on Cifar100-20 with 10k constraints in the overclustering setting with `nout=100` instead of the true `numclasses=20`

Constrained Baseline:

* `python run.py --config configs/constrained_cifar10.yaml` for the constrained baseline on Cifar10 with 10k constraints

Pseudo-Labeling Competitor:

* `python run.py --config configs/pseudolabel_cifar20.yaml` for the pseudo-labeling competitor only using pseudo-labels on Cifar20 with 10k constraints