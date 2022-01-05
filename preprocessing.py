# for standardisation
from sklearn.preprocessing import StandardScaler
from scipy.io import wavfile
from scipy import signal as sg

# for smoothing
from sklearn.neighbors import KernelDensity

import numpy as np
import matplotlib.pyplot as plt

import math

from preferences import *


class Signal:

    def __init__ (self, data: np.ndarray, sample_rate: int):
        self.data = data
        self.data_line = self.data.reshape(-1, 1)
        self.sample_rate = sample_rate
        self.samples_nb = data.shape[0]
        
        self.signal_len = self.samples_nb / sample_rate  # length of signal, in seconds
        self.time = np.linspace(0, self.samples_nb / self.sample_rate, self.samples_nb)

        self.plotted = False
        self.modifiers = list()

    def __repr__ (self):
        desc_list = [
            'Signal object',
            f' Signal length (samples): {self.samples_nb}',
            f' Signal length (seconds): {self.signal_len}',
            f' Sampling rate (Hz):      {self.sample_rate}',
            f' Modifiers:'
        ]
        desc_list += [f'\t{mod}' for mod in self.modifiers]
        return '\n'.join(desc_list)


    def standardise (self):
        scaler = StandardScaler()
        scaler.fit(self.data_line)  # unlimited lines, one column
        data_std = scaler.transform(self.data_line).flatten()

        self.data = data_std
        self.modifiers .append('standardisation')


    def smooth_window (self, window_len=11, window='hanning'):
        if self.data.ndim != 1: raise (ValueError, "smooth only accepts 1 dimension arrays.")
        if self.data.size < window_len: raise (ValueError, "Input vector needs to be bigger than window size.")
        if window_len<3: return self.data
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']: raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        data_padded = np.r_[ self.data[window_len-1:0:-1], self.data, self.data[-2:-window_len-1:-1] ]  # pad signal on both sides

        # Generate window
        if window == 'flat': w = np.ones(window_len, 'd')  # moving average
        else: w = eval('np.' + window + '(window_len)')

        # Apply window convolution
        self.data = np.convolve(w / w.sum(), data_padded, mode='valid')
        self.data = self.data[math.floor((window_len-1)/2):-1*math.ceil((window_len+1)/2)+1]  # remove excess from padding


    def smooth_test2 (self):
        # Create an order 3 lowpass butterworth filter
        #b, a = sg.butter(3, 0.05)
        b, a = sg.butter(3, 0.021)  # lower is smoother
        #b, a = sg.cheby1(3, 10, 100, 'hp', fs=1000)

        sig_filt = sg.lfilter(b, a, self.data)#, axis=- 1, zi=None)
        self.data = sig_filt


    def smooth (self):

        xvals = np.arange(self.samples_nb)
        xvals = self.time

        # Initialise gaussian KDE
        kde = KernelDensity(kernel="gaussian", bandwidth=0.01)#0.75)  # bandwidth=1.0
        # Fit kde on X
        kde.fit(self.data_line)
        # Fetch kde samples along x-axis
        log_dens = kde.score_samples(xvals.reshape(-1, 1))

        # score_samples returns the log of the probability density

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

# plt.figure(figsize=(15, 10))

# Process signal
data = data_0#[:10000]
signal = Signal (data, sample_rate) #[:10000]
signal.standardise()
# plt.plot(signal.time, signal.data, label='')  # plot

# Smooth signal
signal.smooth_window(50, window='bartlett')
# plt.plot(signal.time, signal.data, label='')  # plot

# Add axis labels
# plt.ylabel('amplitude')
# max_abs = max(abs(min(signal.data)), abs(max(signal.data)))
# plt.yticks(np.linspace(math.floor(-max_abs), math.ceil(max_abs), 15))
# plt.xlabel('time [s]')
# plt.xticks(np.linspace(0, signal.signal_len, 15))  # math.ceil(signal.signal_len), 15))
# plt.show()





# wavelet denoising
pass




# VIDEOGRAPH
from legacy.window_fft import VideoGraph

window_width = int(1e6)
window_step  = int(1e5)

video_graph = VideoGraph (
    signal=signal.data,
    sample_rate=sample_rate,
    window_width=window_width,
    window_step=window_step
)

print(video_graph)
print('\n')

video_graph.plot_rfft(freq_cutoff_min=1000, freq_cutoff_max=25000)
video_graph.save(plots=['rfft', 'signal'])
