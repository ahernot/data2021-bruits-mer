from sklearn.preprocessing import StandardScaler
from scipy.io import wavfile

import numpy as np
import matplotlib.pyplot as plt


from preferences import *


class Preprocessing:

    def __init__ (self):
        pass




# Read file
filepath = RESOURCES_PATH + 'samples/2020_01_28_3A/2020_01_28_1.wav'  # use OS path objects
sample_rate, data = wavfile.read(filepath)

# Separate channels
data_0 = data[:, 0]
data_1 = data[:, 1]


def standardise (signal):
    scaler = StandardScaler()
    scaler.fit(signal.reshape(-1, 1))  # unlimited lines, one column
    signal_std = scaler.transform(signal.reshape(-1, 1)).flatten()
    return signal_std

# signal_std = standardise(data_0)
# plt.figure(figsize=(15, 10))
# plt.plot(np.arange(signal_std.shape[0]), signal_std)
# plt.show()










from sklearn.neighbors import KernelDensity

def smooth (signal):

    xvals = np.arange(signal.shape[0])

    # Initialise gaussian KDE
    kde = KernelDensity(kernel="gaussian", bandwidth=0.01)#0.75)  # bandwidth=1.0
    # Fit kde on X
    kde.fit(signal.reshape(-1, 1))
    # Fetch kde samples along x-axis
    log_dens = kde.score_samples(xvals.reshape(-1, 1))

    plt.plot(xvals, signal)
    plt.plot(xvals, np.exp(log_dens))
    # plt.fill(xvals, np.exp(log_dens), fc="#AAAAFF")
    plt.text(-3.5, 0.31, "Gaussian Kernel Density")

    # plt.plot(X[:, 0], np.full(X.shape[0], -0.01), "+k")
    # plt.xlim(-4, 9)
    # plt.ylim(-0.02, 0.34)

    plt.xlabel("x")
    plt.ylabel("Normalized Density")

    plt.show()

    signal_smooth = ''
    return signal_smooth

signal_std = standardise(data_0[:10000])
signal_smooth = smooth(signal_std)







# score_samples returns the log of the probability density


















# wavelet denoising
pass
