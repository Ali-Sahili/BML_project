# BML Project
#### Implementation of the paper ["Fast and Scalable Estimation of Uncertainty using Bayesian Deep Learning"](https://arxiv.org/abs/1806.04854)

## Requirements
The experiments were performed using Python 3.8.5 with the following Python packages:
* [torch](https://pytorch.org/) == 0.4.0
* [torchvision](https://pypi.org/project/torchvision/0.1.8/) == 0.6.1
* [numpy](http://www.numpy.org/) == 1.14.1
* [pandas](https://pandas.pydata.org/) == 0.25.3
* [matplotlib](https://matplotlib.org/) == 3.1.2
* [tqdm](https://tqdm.github.io/) == 4.30.0

## Install

In the folder containing `setup.py`, run
```
pip install --user -e .
```
## Scripts explanation:
There are 4 python scripts:
* `run_experiments.py` - This is the main script to run experiments.
* `experiments.py` - This script implements experiment classes for Vadam and BBVI.
* `produce_gifs.py` - This script is for producing the animated GIFs. It loads files produced by `run_experiments.py`.
* `visualize_results.ipynb` - This is a jupyter notebook for visualizing the results of the experiment. It loads files produced by `run_experiments.py`.

## Testing the Code

## MNIST Experiments

## Aknownledgements
Original implementation by [Emtiyaz_khan](github.com/emtiyaz/vadam)
