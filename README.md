# DiPPS: Differentially Private Propensity Scores for Bias Correction

## Introduction

This is the implementation of the differentially private data collection mechanism *DiPPS* proposed in the paper [Differentially Private Propensity Scores for Bias Correction](https://arxiv.org/abs/2210.02360) and the experiments to compare it with other methods.

## Prerequisites

* [PyTorch 1.7.1](https://pytorch.org/) `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`

* [numpy 1.19.2](https://numpy.org/) `conda instal -c anaconda numpy`

* [pandas 1.2.2](https://pandas.pydata.org/) `conda install -c anaconda pandas`

* [scikit-learn 0.23.2](https://scikit-learn.org/stable/install.html)  `conda install -c conda-forge scikit-learn`

* [R package 'transport'](https://cran.r-project.org/web/packages/transport/index.html) `wget https://cran.r-project.org/src/contrib/transport_0.12-2.tar.gz`


## Code structure
```
.
├── run.py                              # The main script to run the experiments to estimate mean/variance/median or other target functions on datasets
├── data/                               # The datasets we use for the experiments
├── preprocess/                         # The scripts to preprocess the original datasets and to obtain the data split into participant and non-participant data
├── config/                             # The configurations of the experiments, including the model parameters
├── exponential_mechanism.py            # The implementation of the *exponential mechanism*
├── invert_exponential_mechanism.py     # The implementation of the estimation of group statistics based on data collected under the *exponential mechanism*
├── hybrid_mechanism.py                	# The implementation of the *Hybrid mechanism*
├── laplace_mechanism.py                # The implementation of the *Laplace mechanism*
├── LCIW.py                     	# The implementation of *PS*, i.e., the propensity score based estimation method proposed in the paper. Combined with the exponential mechanism, it turns into DiPPS
├── estimation.py                    	# The summarization script to estimate the target variable based on all the methods
├── computeWassersteinDist.r            # The helper script to compute the Wasserstein distance. It is based on the R package 'transport', which is mentioned in *Prerequisites*
├── view_results.ipynb          	# Summarize statistics of experiments
└── 
...
```

## Run experiments

- `bash run.sh` The experiments can be configured via parameters in the run script itself, which are documented therein.

- The experiments in the paper are obtained with the 5 random seeds `[1111, 2222, 3333, 4444, 5555]`. One can set the seed in `run.sh` directly.
