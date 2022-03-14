# Data-Driven End-to-End Delay Violation Probability Prediction with Extreme Value Mixture Models

In this work, we have implemented a novel conditional density estimation method that uses extreme value mixture model as the parametric density function of the mixture density network (MDN). We use this novel density estimator to predict the transient delay violation probability (DVP) of the packets traversing a tandem queueing network from the queues lengths.
Numerically, we showed in the paper that our proposed approach outperforms state-of-the-art Gaussian mixture model-based predictors by orders of magnitude, in particular for quantiles above 0.99.

This repository contains the implementation of the aformentioned system. First, we simulate a 3-hop tandem queuing system in MATLAB Simulink environment and record the end-to-end delays together with the observed queue lengths. Then, these records are used as the training or evaluation dataset for the latency predictor in Python using Tensorflow.


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
This repository contains the models, evaluation schemes, and numerics of the following paper: ***Data-Driven End-to-End Delay Violation Probability Prediction with Extreme Value Mixture Models*** published by The Sixth ACM/IEEE Symposium on Edge Computing (SEC) in San Jose, CA, December 14-17, 2021 [here](https://ieeexplore.ieee.org/document/9708928).


## Citing
If you use our extreme value theory-based predictor in your research, you can cite it as follows:

```

@INPROCEEDINGS{9708928,
  author={Mostafavi, Seyed Samie and Dán, György and Gross, James},
  booktitle={2021 IEEE/ACM Symposium on Edge Computing (SEC)}, 
  title={Data-Driven End-to-End Delay Violation Probability Prediction with Extreme Value Mixture Models}, 
  year={2021},
  volume={},
  number={},
  pages={416-422},
  doi={10.1145/3453142.3493506}
}

```


