# Data-Driven End-to-End Delay Violation Probability Prediction with Extreme Value Mixture Models

## Description

### Latency Prediction (`latency_prediction` folder):

Implementation of two conditional density estimation methods with parametric neural networks in Python:

* Conventional Mixture Density Network with Gaussian Mixture Model (GMM)
* Novel Mixture Density Network with Gaussian and Extreme Value Mixture Model (EMM)

To use the code, create a 3.6 Python virtual environment in `latency_prediction` folder and run:
```bash
pip install -r requirements.txt
```

This implementation is based on the repository [here](https://github.com/freelunchtheorem/Conditional_Density_Estimation).

### Queuing Simulation (`queueing_simulation` folder):

Implementation of a 3 hop tandem queuing network in MATLAB Simulink using SimEvents toolbox.

To use the code and design, open the `queueing_simulation/noaqm_threehop_parallel` folder using MATLAB r2020b.


## Paper
This repository contains the models, evaluation schemes, and numerics of the following paper: ***Data-Driven End-to-End Delay Violation Probability Prediction with Extreme Value Mixture Models*** published by The Sixth ACM/IEEE Symposium on Edge Computing (SEC) in San Jose, CA, December 14-17, 2021 [here](https://google.com).


## Citing
If you use our extreme value theory-based predictor in your research, you can cite it as follows:

```
@article{,
  title={},
  author={},
  journal={},
  year={}
}

```


