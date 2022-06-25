import numpy as np
from typing import Callable

class Autoregressive():
    def __init__(self,
        level : int,
        phi : list,
        c : np.float64 = 0,
    ):
        self._level = level
        self._latent_var = list(np.zeros(self._level))
        self._phi = phi
        self._c = c
        
        
    def get_rnd_ar(self,
        rnd_fn : Callable,
    ):
        rnd = rnd_fn()
        result = rnd + np.squeeze(np.sum(np.array(self._latent_var)*np.array(self._phi))) + self._c

        self._latent_var.insert(0,result)
        self._latent_var.pop()
        return result