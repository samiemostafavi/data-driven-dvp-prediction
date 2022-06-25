import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

def mixture_samples(
    cdf_bool_split_t,
    gpd_sample_t,
    bulk_sample_t,
    dtype: tf.DType,
):

    gpd_multiplexer = cdf_bool_split_t
    bulk_multiplexer = tf.logical_not(cdf_bool_split_t)

    gpd_multiplexer = tf.cast(gpd_multiplexer, dtype=dtype) # convert it to float (1.00 or 0.00) for multiplication
    bulk_multiplexer = tf.cast(bulk_multiplexer, dtype=dtype) # convert it to float (1.00 or 0.00) for multiplication

    multiplexed_gpd_sample = tf.multiply(gpd_sample_t,gpd_multiplexer)
    multiplexed_bulk_sample = tf.multiply(bulk_sample_t,bulk_multiplexer)

    return tf.reduce_sum(
        tf.stack([
            multiplexed_gpd_sample,
            multiplexed_bulk_sample,
        ]),
        axis=0,
    )

def heavytail_gamma_n(
    n,
    gamma_concentration,
    gamma_rate,
    gpd_concentration,
    seed : int = 0 ,
    threshold_qnt = 0.9,
    dtype = tf.float64,
):

    # making conditional distribution
    # gamma + gpd

    # sample = rnd;
    samples_t = tf.random.uniform(
        shape = [n],
        dtype = dtype,
        seed = seed,
    )

    threshold_qnt_t = tf.convert_to_tensor(threshold_qnt, dtype=dtype)
    
    gamma = tfp.distributions.Gamma(
        concentration=gamma_concentration,
        rate=gamma_rate,
    )

    threshold_act_t = gamma.quantile(threshold_qnt_t)

    # split the samples into bulk and tail according to the norm_factor (from X and Y)
    # gives a tensor, indicating which random_input are greater than norm_factor
    # greater than threshold is true, else false
    bool_split_t = tf.greater(samples_t, threshold_qnt_t) # this is in Boolean

    # gamma samples tensor
    gamma_samples_t = gamma.quantile(samples_t)
    
    # gpd samples tensor
    gpd_presamples_t = tf.divide(
        samples_t-threshold_qnt_t,
        tf.constant(1.00,dtype = dtype)-threshold_qnt_t,
    )

    gpd_scale = tf.multiply(
        tf.divide(
            tf.constant(1.00,dtype = dtype),
            gamma.prob(threshold_act_t),
        ),
        tf.square(
            tf.constant(1.00,dtype = dtype)-threshold_qnt_t
        ),
    )

    gpd = tfp.distributions.GeneralizedPareto(
        loc = 0.00,
        scale = gpd_scale,
        concentration = gpd_concentration,
    )
    gpd_samples_t = gpd.quantile(gpd_presamples_t)/(tf.constant(1.00,dtype = dtype)-threshold_qnt_t)+threshold_act_t

    # pass them through the mixture filter
    result = mixture_samples(
        cdf_bool_split_t = bool_split_t,
        gpd_sample_t = gpd_samples_t,
        bulk_sample_t = gamma_samples_t,
        dtype = dtype,
    )

    return result

class HeavyTail():
    def __init__ (self, 
        n = 1000000,
        gamma_concentration = 10,
        gamma_rate = 0.5,
        gpd_concentration = 0.2,
        threshold_qnt = 0.8,
        seed = 0,
        dtype = tf.float32,
    ):

        self._n = n
        self._counter = 0
        self._gamma_concentration = gamma_concentration
        self._gamma_rate = gamma_rate
        self._gpd_concentration = gpd_concentration
        self._threshold_qnt = threshold_qnt
        self._dtype = dtype
        self._seed = seed

        self._numbers = heavytail_gamma_n(
            n = n,
            gamma_concentration = gamma_concentration,
            gamma_rate = gamma_rate,
            gpd_concentration = gpd_concentration,
            threshold_qnt = threshold_qnt,
            dtype = dtype,
            seed = seed,
        ).numpy()

    def get_rnd_heavy_tail(self):
        res = self._numbers[self._counter]
        self._counter = self._counter + 1
        return res
