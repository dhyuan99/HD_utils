import timeit
from quantize import record_based, N_gram_based, distance_preserving
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore')

SETUP = '''
from quantize import record_based, N_gram_based, distance_preserving
import numpy as np
Q1 = record_based(real_dim=85, HD_dim=2048, q=100)
Q2 = N_gram_based(real_dim=85, HD_dim=2048, q=100)
Q3 = distance_preserving(real_dim=85, q=2048)
x = np.random.uniform(size=[85])
'''

print(f"record_based: {timeit.timeit(setup=SETUP, stmt='Q1.quantize(x)', number=10000) / 10} ms per vector.")
print(f"N_gram_based: {timeit.timeit(setup=SETUP, stmt='Q2.quantize(x)', number=10000) / 10} ms per vector.")
print(f"N_gram_based: {timeit.timeit(setup=SETUP, stmt='Q3.quantize(x)', number=10000) / 10} ms per vector.")

R = record_based(real_dim=1, HD_dim=2048, q=100)
G = N_gram_based(real_dim=1, HD_dim=2048, q=100)
D = distance_preserving(real_dim=1, q=2048)

R_results = np.zeros(shape=[11, 2048])
G_results = np.zeros(shape=[11, 2048])
D_results = np.zeros(shape=[11, 2048])

for i in range(10):
    R_results[i, :] = R.quantize(np.array([i / 10]))
    G_results[i, :] = G.quantize(np.array([i / 10]))
    D_results[i, :] = D.quantize(np.array([i / 10]))

R_dist = np.sum((np.expand_dims(R_results, axis=0) - np.expand_dims(R_results, axis=1)) ** 2, axis=2) / 2048
G_dist = np.sum((np.expand_dims(G_results, axis=0) - np.expand_dims(G_results, axis=1)) ** 2, axis=2) / 2048
D_dist = np.sum((np.expand_dims(D_results, axis=0) - np.expand_dims(D_results, axis=1)) ** 2, axis=2) / 2048


y, x = np.mgrid[slice(0, 1 + 0.1, 0.1),
                slice(0, 1 + 0.1, 0.1)]

plt.pcolor(x, y, R_dist, cmap='RdBu', vmin=0, vmax=0.7)
plt.colorbar()
plt.savefig('figs/Record_matrix.jpg')
plt.close()

plt.pcolor(x, y, G_dist, cmap='RdBu', vmin=0, vmax=0.7)
plt.colorbar()
plt.savefig('figs/N-Gram_matrix.jpg')
plt.close()

plt.pcolor(x, y, D_dist, cmap='RdBu', vmin=0, vmax=0.7)
plt.colorbar()
plt.savefig('figs/distance-preserving_matrix.jpg')
plt.close()