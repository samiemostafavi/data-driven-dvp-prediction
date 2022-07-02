from  arrivals import heavytail_gamma_prob,heavytail_gamma_cdf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#marray = np.linspace(0.0, 40.0, num=100)
marray = [np.nan, 1,2,3,4,5,6,7,8,9,10, np.nan]

ht = heavytail_gamma_cdf(
    y = marray,
    gamma_concentration = 5,
    gamma_rate = 0.5,
    gpd_concentration = 0.4,
    threshold_qnt = 0.8,
    dtype = np.float32,
)

print(ht)
#plt.plot(ht)
#plt.savefig('books_read.png')