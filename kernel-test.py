# Author: Jake Vanderplas <jakevdp@cs.washington.edu>

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.utils.fixes import parse_version


# Generate data
np.random.seed(1)
N = 20
X = np.concatenate(
    (np.random.normal(0, 1, int(0.3 * N)), np.random.normal(5, 1, int(0.7 * N)))
)[:, np.newaxis]
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]


plt.figure(figsize=(15, 10))

# Initialise gaussian KDE
kde = KernelDensity(kernel="gaussian", bandwidth=0.75)
# Fit kde on X
kde.fit(X)
# Fetch kde samples along x-axis
log_dens = kde.score_samples(X_plot)

plt.plot(X_plot[:, 0], np.exp(log_dens))
plt.fill(X_plot[:, 0], np.exp(log_dens), fc="#AAAAFF")
plt.text(-3.5, 0.31, "Gaussian Kernel Density")

plt.plot(X[:, 0], np.full(X.shape[0], -0.01), "+k")
plt.xlim(-4, 9)
plt.ylim(-0.02, 0.34)

plt.xlabel("x")
plt.ylabel("Normalized Density")

plt.show()
