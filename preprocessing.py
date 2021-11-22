# for standardisation
from sklearn.preprocessing import StandardScaler
from scipy.io import wavfile

# for smoothing
from sklearn.neighbors import KernelDensity

import numpy as np
import matplotlib.pyplot as plt

from preferences import *


class Signal:

    def __init__ (self, data: np.ndarray, sample_rate: int):
        self.data = data
        self.data_line = self.data.reshape(-1, 1)
        self.sample_rate = sample_rate
        self.samples_nb = data.shape[0]        
        self.time = np.linspace(0, self.samples_nb / self.sample_rate, self.samples_nb)

        self.modifiers = list()

    def __repr__ (self):
        desc_list = [
            'Signal object',
            f' Signal length (samples): {self.samples_nb}',
            f' Sampling rate (Hz):      {self.sample_rate}',
            f' Modifiers:'
        ]
        desc_list += [f'\t{mod}' for mod in self.modifiers]
        return '\n'.join(desc_list)


    def standardise (self):
        data_reshaped = self.data.reshape(-1, 1)

        scaler = StandardScaler()
        scaler.fit(self.data_line)  # unlimited lines, one column
        data_std = scaler.transform(self.data_line).flatten()

        self.data = data_std
        self.modifiers .append('standardisation')


    def smooth_demo(self, window_len=11, window='hanning'):
        if self.data.ndim != 1: raise (ValueError, "smooth only accepts 1 dimension arrays.")
        if self.data.size < window_len: raise (ValueError, "Input vector needs to be bigger than window size.")
        if window_len<3: return self.data
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']: raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s = np.r_[ self.data[window_len-1:0:-1], self.data, self.data[-2:-window_len-1:-1] ]  # pad signal on both sides

        if window == 'flat': w = np.ones(window_len, 'd')  # moving average
        else: w = eval('np.' + window + '(window_len)')
        self.data = np.convolve(w / w.sum(), s, mode='valid')

        print(self.data.shape)


    def smooth (self):

        xvals = np.arange(self.samples_nb)
        xvals = self.time

        # Initialise gaussian KDE
        kde = KernelDensity(kernel="gaussian", bandwidth=0.01)#0.75)  # bandwidth=1.0
        # Fit kde on X
        kde.fit(self.data_line)
        # Fetch kde samples along x-axis
        log_dens = kde.score_samples(xvals.reshape(-1, 1))

        # plt.plot(xvals, signal)

        plt.figure(figsize=(15, 10))
        plt.plot(xvals, self.data)
        plt.plot(xvals, np.exp(log_dens))
        # plt.fill(xvals, np.exp(log_dens), fc="#AAAAFF")
        plt.text(-3.5, 0.31, "Gaussian Kernel Density")

        # plt.plot(X[:, 0], np.full(X.shape[0], -0.01), "+k")
        # plt.xlim(-4, 9)
        # plt.ylim(-0.02, 0.34)

        plt.xlabel("x")
        plt.ylabel("Normalized Density")
        plt.show()

        self.modifiers .append('kernel smoothing')



# Read file
filepath = RESOURCES_PATH + 'samples/2020_01_28_3A/2020_01_28_1.wav'  # use OS path objects
sample_rate, data = wavfile.read(filepath)

# Separate channels
data_0 = data[:, 0]
data_1 = data[:, 1]

# Process signal
signal = Signal (data_0[:10000], sample_rate)
signal.standardise()



# signal.smooth_demo(50)
# plt.plot(np.arange(signal.data.shape[0]), signal.data)

plt.figure(figsize=(15, 10))
# plt.plot(signal.time, signal.data)

plt.show()







# score_samples returns the log of the probability density






# wavelet denoising
pass
