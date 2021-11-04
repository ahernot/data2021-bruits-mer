import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft, fftfreq
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft

from scipy.signal import cwt, ricker
from scipy.io import wavfile
from scipy.stats.stats import ttest_1samp

from preferences import *


# Read the wav file
filepath = RESOURCES_PATH + 'samples/2020_01_28_3A/2020_01_28_1.wav'  # use OS path objects
sample_rate, data = wavfile.read(filepath)


# Plot part of the signal
RUN_1 = False
if RUN_1 == True:
    sample = data[:10000, :]
    xcoords = np.arange(sample.shape[0])

    plt.figure(figsize=(15, 10))
    plt.plot(xcoords, sample[:, :])
    plt.show()


# Plot fourier transform of the signal (to continue)
RUN_2 = False
if RUN_2 == True:
    sp = np.fft.fft (data[:, 0])  # spectrum
    freq = np.fft.fftfreq( data.shape[0] )  # frequencies
    plt.figure(figsize=(15, 10))
    plt.plot(freq, sp.real, freq, sp.imag)
    plt.show()


# Fourier transform
RUN_3 = False
if RUN_3 == True:
    N = data.shape[0]  # Number of samples

    yf = fft(data[:, 0])
    xf = fftfreq(N, 1 / sample_rate)

    plt.figure(figsize=(15, 10))
    plt.plot(xf, np.abs(yf))
    plt.show()


# Real Fourier transform
RUN_4 = True
if RUN_4 == True:
    N = data.shape[0]  # Number of samples

    # Note the extra 'r' at the front
    yf = rfft(data[:, 0])
    xf = rfftfreq(N, 1 / sample_rate)

    # plt.figure(figsize=(15, 10))
    # plt.plot(xf, np.abs(yf))
    # plt.show()

    # todo: compute maxima



    ### Filtre passe-bande basique
    # The maximum frequency is half the sample rate
    points_per_freq = len(xf) / (sample_rate / 2)

    target_freq = 3000
    target_idx = int(points_per_freq * target_freq)

    yf_filtered = np.copy(yf)
    yf_filtered[:target_idx - 4000] = 0
    yf_filtered[target_idx + 50000:] = 0

    plt.figure(figsize=(15, 10))
    plt.plot(xf, np.abs(yf_filtered))
    plt.show()


    ### Inverse real fourier transform
    new_sig = irfft(yf_filtered)

    plt.figure(figsize=(15, 10))
    plt.plot(new_sig)
    plt.show()








RUN_10 = False
if RUN_10 == True:
    widths = np.arange(1, 31)
    test_wavelet = cwt (data[:, 0], wavelet=ricker, widths=widths)
    # print(test_wavelet.shape)  # (30, 6398400)

    nplots = len(widths)
    fig, ax = plt.subplots(1, nplots, figsize=(100, 5), sharey=True)
    for width_id in range(nplots):

        wavelet_sample = test_wavelet[width_id, :]

        ax[width_id].plot (np.arange(wavelet_sample.shape[0]), wavelet_sample)
        ax[width_id].set_title (str(widths[width_id]))  # f'width: {widths[width_id]}')

    # plt.show()
    plt.savefig('fig.jpg', dpi=300, bbox_inches='tight')
