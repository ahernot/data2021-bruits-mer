import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from preferences import *


# Read the wav file
filepath = RESOURCES_PATH + 'samples/2020_01_28_3A/2020_01_28_1.wav'  # use OS path objects
samplerate, data = wavfile.read(filepath)


# Plot part of the signal
RUN_1 = False
if RUN_1 == True:
    sample = data[:10000, :]
    xcoords = np.arange(sample.shape[0])

    plt.figure(figsize=(15, 10))
    plt.plot(xcoords, sample[:, :])
    plt.show()


# Plot fourier transform of the signal (to continue)
RUN_2 = True
if RUN_2 == True:
    sp = np.fft.fft (data[:, 0])  # spectrum
    freq = np.fft.fftfreq( data.shape[0] )  # frequencies
    plt.figure(figsize=(15, 10))
    plt.plot(freq, sp.real, freq, sp.imag)
    plt.show()

