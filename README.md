# BML Project
#### Implementation of the paper ["Fast and Scalable Estimation of Uncertainty using Bayesian Deep Learning"](https://arxiv.org/abs/1806.04854)

## Requirements
The experiment was performed using Python 3.5.2 with the following Python packages:
* [torch](https://pytorch.org/) == 0.4.0
* [numpy](http://www.numpy.org/) == 1.14.1

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
