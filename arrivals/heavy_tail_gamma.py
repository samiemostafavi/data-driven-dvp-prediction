import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from dataclasses import dataclass
from typing import Dict
from qsimpy.random import RandomProcess


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

class HeavyTailGamma(RandomProcess):
    type: str = 'heavytailgamma'
    gamma_concentration : np.float64
    gamma_rate : np.float64
    gpd_concentration : np.float64
    threshold_qnt : np.float64

    def prepare_for_run(self):
        self._rng = np.random.default_rng(self.seed)

    def sample_n(self,
        n,
    ):

        # making conditional distribution
        # gamma + gpd

        # sample = rnd;
        samples_np = self._rng.uniform(
            low = 0.0,
            high = 1.0,
            size = n,
        )
        samples_t = tf.convert_to_tensor(value=samples_np, dtype=self.dtype)

        threshold_qnt_t = tf.convert_to_tensor(self.threshold_qnt, dtype=self.dtype)
        
        gamma = tfp.distributions.Gamma(
            concentration=self.gamma_concentration,
            rate=self.gamma_rate,
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
            tf.constant(1.00,dtype = self.dtype)-threshold_qnt_t,
        )

        gpd_scale = tf.multiply(
            tf.divide(
                tf.constant(1.00,dtype = self.dtype),
                gamma.prob(threshold_act_t),
            ),
            tf.square(
                tf.constant(1.00,dtype = self.dtype)-threshold_qnt_t
            ),
        )

        gpd = tfp.distributions.GeneralizedPareto(
            loc = 0.00,
            scale = gpd_scale,
            concentration = self.gpd_concentration,
        )
        gpd_samples_t = gpd.quantile(gpd_presamples_t)/(tf.constant(1.00,dtype = self.dtype)-threshold_qnt_t)+threshold_act_t

        # pass them through the mixture filter
        result = mixture_samples(
            cdf_bool_split_t = bool_split_t,
            gpd_sample_t = gpd_samples_t,
            bulk_sample_t = gamma_samples_t,
            dtype = self.dtype,
        )

        return result.numpy()

    def prob(self,
        y,
    ):
    
        # making conditional distribution
        # gamma + gpd
        # to give probability

        y_t = tf.convert_to_tensor(y, dtype=self.dtype)

        threshold_qnt_t = tf.convert_to_tensor(self.threshold_qnt, dtype=self.dtype)
        
        gamma = tfp.distributions.Gamma(
            concentration=self.gamma_concentration,
            rate=self.gamma_rate,
        )

        threshold_act_t = gamma.quantile(threshold_qnt_t)

        #print("threshold:")
        #print(threshold_act_t.numpy())

        # split the samples into bulk and tail according to the norm_factor (from X and Y)
        # gives a tensor, indicating which random_input are greater than norm_factor
        # greater than threshold is true, else false
        bool_split_t = tf.greater(y_t,threshold_act_t) # this is in Boolean
        
        #print("split_tensor:")
        #print(bool_split_t.numpy())

        # gamma probabilities tensor
        gamma_probs_t = gamma.prob(y_t)
        #print("gamma probs:")
        #print(gamma_probs_t.numpy())

        # make gpd distribution
        gpd_scale = tf.divide(
            tf.constant(1.00,dtype = self.dtype) - threshold_qnt_t,
            gamma.prob(threshold_act_t),
        )

        gpd = tfp.distributions.GeneralizedPareto(
            loc = threshold_act_t, #0.00
            scale = gpd_scale,
            concentration = self.gpd_concentration,
        )

        # gpd probabilities tensor
        gpd_probs_t = tf.multiply(
            gpd.prob(y_t),
            tf.constant(1.00,dtype = self.dtype)-threshold_qnt_t,
        )
        
        #print("gpd_probs:")
        #print(gpd_probs_t.numpy())
        # convert NaNs to zeros
        gpd_probs_t = tf.where(tf.math.is_nan(gpd_probs_t), tf.zeros_like(gpd_probs_t), gpd_probs_t)
        #print(gpd_probs_t.numpy())

        # pass them through the mixture filter
        result = mixture_samples(
            cdf_bool_split_t = bool_split_t,
            gpd_sample_t = gpd_probs_t,
            bulk_sample_t = gamma_probs_t,
            dtype = self.dtype,
        )

        # Keep NaNs from the input
        result = tf.where(tf.math.is_nan(y_t), tf.ones_like(y_t)*tf.constant(np.nan), result)

        return result
    
    def cdf(self,
        y,
    ):
    
        # making conditional distribution
        # gamma + gpd
        # to give probability

        y_t = tf.convert_to_tensor(y, dtype=self.dtype)

        threshold_qnt_t = tf.convert_to_tensor(self.threshold_qnt, dtype=self.dtype)
        
        gamma = tfp.distributions.Gamma(
            concentration=self.gamma_concentration,
            rate=self.gamma_rate,
        )

        threshold_act_t = gamma.quantile(threshold_qnt_t)

        #print("threshold:")
        #print(threshold_act_t.numpy())

        # split the samples into bulk and tail according to the norm_factor (from X and Y)
        # gives a tensor, indicating which random_input are greater than norm_factor
        # greater than threshold is true, else false
        bool_split_t = tf.greater(y_t,threshold_act_t) # this is in Boolean
        
        #print("split_tensor:")
        #print(bool_split_t.numpy())

        # gamma probabilities tensor
        gamma_probs_t = gamma.cdf(y_t)
        #print("gamma probs:")
        #print(gamma_probs_t.numpy())

        # make gpd distribution
        gpd_scale = tf.divide(
            tf.constant(1.00,dtype = self.dtype) - threshold_qnt_t,
            gamma.prob(threshold_act_t),
        )

        gpd = tfp.distributions.GeneralizedPareto(
            loc = threshold_act_t, #0.00
            scale = gpd_scale,
            concentration = self.gpd_concentration,
        )

        # gpd probabilities tensor
        gpd_probs_t = tf.multiply( 
            gpd.cdf(y_t),
            tf.constant(1.00,dtype = self.dtype) - threshold_qnt_t,
        ) + threshold_qnt_t
        
        #print("gpd_probs:")
        #print(gpd_probs_t.numpy())
        gpd_probs_t = tf.where(tf.math.is_nan(gpd_probs_t), tf.zeros_like(gpd_probs_t), gpd_probs_t)

        # pass them through the mixture filter
        result = mixture_samples(
            cdf_bool_split_t = bool_split_t,
            gpd_sample_t = gpd_probs_t,
            bulk_sample_t = gamma_probs_t,
            dtype = self.dtype,
        )

        # Keep NaNs from the input
        result = tf.where(tf.math.is_nan(y_t), tf.ones_like(y_t)*tf.constant(np.nan), result)

        return result
