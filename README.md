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

For Logistic regression, we use Matlab to perform experiments with its [online compiler](https://matlab.mathworks.com/).

## Install

In the folder containing `setup.py`, run
```
pip install --user -e .
```
## Scripts explanation:
<pre>
|Logistic_reg <br />
  |datasets <br />
    |preprocess.py - prepare data as a .csv file before processing. <br />
    |colon-cancer.mat - training data and labels of [colo-cancer]() dataset. <br />
    |fourclass.mat - training data and labels of [fourclass]() dataset. <br />
    |breast_cancer_scale.mat - training data and labels of [breast_cancer_scale]() dataset. <br />
    |usps_resampled.mat - training data and labels of [usps_resampled]() dataset. <br />
  |main.m - main script to run experiments (vadam, vogn and mf-exact) on different datasets. <br />
  |main_3d.m - main script to run experiments (vadam, vogn and mf-exact) on different datasets with an additional feature dimension. <br />
  |plot_results.m - script to plot results showing the posterior and the uncertainty of each method. <br />
  |lib - Folder containing some useful functions used to compute likelihoods, posterior distributions, uncertainty and so on. <br />
  |results - contains some of our results on two of the used datasets. <br />
|vadam <br />
  |models.py <br />
  |datasets.py <br />
  |optimizers.py <br />
  |metrics.py <br />
  |utils.py <br />
|VOGN <br />
  |results - Results of varying some parameters of VOGN <br />
  |experiments.py <br />
  |run_experiments.py <br />
|MNIST <br />
  |results - Results of varying some parameters of VOGN <br />
  |experiments.py <br />
  |run_experiments.py <br />
  |Visualizing.ipynb - This is a jupyter notebook for visualizing the results of the experiment. It loads files produced by `run_experiments.py`. <br />
<pre>

## Testing the Code

## MNIST Experiments
To run the MNIST experiments:
```
python MNIST/run_experiments.py
```
You can specify the hyperparameters to explore by changing the `grid` variable in the this script.
## Aknownledgements
Original implementation by [Emtiyaz_khan](github.com/emtiyaz/vadam)
