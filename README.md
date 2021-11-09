# Data-Driven End-to-End Delay Violation Probability Prediction with Extreme Value Mixture Models

## Description

### Latency Prediction (`latency_prediction` folder):

Implementation of two conditional density estimation methods with parametric neural networks in Python:

* Conventional Mixture Density Network with Gaussian Mixture Model (GMM)
* Novel Mixture Density Network with Gaussian and Extreme Value Mixture Model (EMM)

Based on the repository [here](https://github.com/freelunchtheorem/Conditional_Density_Estimation).

### Queuing Simulation (`queueing_simulation` folder):

Implementation of a 3 hop queuing network in MATLAB Simulink using SimEvents toolbox.

## Installation

To use the library, clone the GitHub repository and run 
```bash
pip install .
``` 
Note that the package only supports tensorflow version 1.7 and MATLAB r2020b.

## Paper
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


