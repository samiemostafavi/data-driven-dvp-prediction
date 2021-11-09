# Data-Driven End-to-End Delay Violation Probability Prediction with Extreme Value Mixture Models

## Description

### Latency Prediction (`latency_prediction` folder):

Implementation of various mixture density network methods to predict the end-to-end delay of a tandem queuing network. We use conditional density estimation with parametric neural networks.

* **Parametric neural network based predictors**
    * Mixture Density Network with Gaussian Mixture Model only (GMM)
    * Mixture Density Network with Gaussian and Extreme Value Mixture Model (EMM)

### Queuing Simulation (`queueing_simulation` folder):

Implementation of a 3 hop queuing network in MATLAB Simulink using SimEvents toolbox.

## Installation

To use the library, clone the GitHub repository and run 
```bash
pip install .
``` 
Note that the package only supports tensorflow versions between 1.4 and 1.7, and MATLAB r2020b.

## paper
This repository contains the implementation of the paper [here](https://arxiv.org/abs/1903.00954).


## Citing
If you use our extreme value theory-based predictor in your research, you can cite it as follows:

```
@article{rothfuss2019conditional,
  title={Conditional Density Estimation with Neural Networks: Best Practices and Benchmarks},
  author={Rothfuss, Jonas and Ferreira, Fabio and Walther, Simon and Ulrich, Maxim},
  journal={arXiv:1903.00954},
  year={2019}
}

```


