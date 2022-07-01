from  arrivals import heavytail_gamma_prob
import numpy as np
import matplotlib.pyplot as plt

marray = np.linspace(0.0, 40.0, num=100)

ht = heavytail_gamma_prob(
    y = marray,
    gamma_concentration = 5,
    gamma_rate = 0.5,
    gpd_concentration = 0.4,
    threshold_qnt = 0.8,
    dtype = np.float32,
)

plt.plot(ht)
plt.savefig('books_read.png')