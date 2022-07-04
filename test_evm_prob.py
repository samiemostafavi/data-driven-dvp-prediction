from  arrivals import Autoregressive, HeavyTailGamma, RandomProcess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#marray = np.linspace(0.0, 40.0, num=100)
marray = [np.nan, 1,2,3,4,5,6,7,8,9,10, np.nan]


htg = HeavyTailGamma(
    seed = 12345,
    gamma_concentration = 5,
    gamma_rate = 0.5,
    gpd_concentration = 0.4,
    threshold_qnt = 0.8,
    dtype = 'float32'#np.float32,
)
htg.prepare_for_run()

print(htg.prob(marray))
print(htg.cdf(marray))
print(htg.sample_n(10))

mstr = htg.json()
del htg

print(mstr)
htg = RandomProcess.parse_raw(mstr)

ahtg = Autoregressive(
    seed = 12345,
    level = 1,
    phi=[0.1],
    c = 0,
    base_process=htg,
    dtype = 'float32',
)
ahtg.prepare_for_run()
print(ahtg.json())
print(ahtg.sample_n(10))

mstr = ahtg.json()
del ahtg
ahtg = RandomProcess.parse_raw(mstr)
ahtg.prepare_for_run()
print(ahtg.json())
print(ahtg.sample_n(10))

exit(0)

del htg
htg = RandomProcess.from_json('{"name": "HeavyTailGamma", "dtype": "float32", "gamma_concentration": 5, "gamma_rate": 0.5, "gpd_concentration": 0.4, "threshold_qnt": 0.8}')

print(htg.prob(marray))
print(htg.cdf(marray))
print(htg.sample_n(10,12345))

#plt.plot(ht)
#plt.savefig('books_read.png')