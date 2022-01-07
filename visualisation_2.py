import matplotlib.pyplot as plt
import numpy as np


def plot_sig_add (sig: np.ndarray, sampling_freq: int, title: str = None, xlabel: str = None, ylabel: str = None):
    plt.plot(np.arange(sig.shape[0]) / sampling_freq, sig)
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)

def plot_sig (sig: np.ndarray, sampling_freq: int, title: str = None):
    plt.figure(figsize=(15, 10))
    plot_sig_add(sig=sig, sampling_freq=sampling_freq, title=title, xlabel='Time [s]', ylabel='Amplitude')
    plt.show()

