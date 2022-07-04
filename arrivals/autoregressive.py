import numpy as np
from typing import Callable
from qsimpy.random import RandomProcess
from pydantic import PrivateAttr

class Autoregressive(RandomProcess):
    type: str = 'autoregressive'
    level : int
    phi : list[np.float64]
    c : np.float64 = 0
    base_process : RandomProcess

    _latent_var : list[np.float64] = PrivateAttr()

    def __init__(self, **data):
        if isinstance(data['base_process'], RandomProcess):
            data['base_process'] = data['base_process'].dict()
        super().__init__(**data)

    def prepare_for_run(self):
        self._rng = np.random.default_rng(self.seed)
        self._latent_var = list(np.zeros(self.level))
        self.base_process.prepare_for_run()
        
    def sample(self):
        rnd = self.base_process.sample()
        result = rnd + np.squeeze(np.sum(np.array(self._latent_var)*np.array(self.phi))) + self.c

        self._latent_var.insert(0,result)
        self._latent_var.pop()
        return result

    def sample_n(self, n: int):
        rnd = self.base_process.sample_n(n)
        res = []
        for num in rnd:
            result = num + np.squeeze(np.sum(np.array(self._latent_var)*np.array(self.phi))) + self.c
            self._latent_var.insert(0,result)
            self._latent_var.pop()
            res.append(result)
        return np.array(res)