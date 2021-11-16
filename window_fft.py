import scipy
from scipy.io import wavfile
from scipy.signal import cwt, ricker, spectrogram
from scipy.fft import rfft, rfftfreq

import matplotlib.pyplot as plt
import numpy as np

import cv2

import os
import shutil

from preferences import *




def plot_signal (signal: np.ndarray, filename: str = 'signal'):
    plt.figure(figsize=(15, 10))
    plt.plot(np.arange(signal.shape[0]), signal)
    plt.savefig(f'{OUTPUT_PATH}{filename}.png', bbox_inches='tight')

def plot_spectrogram (spectrogram: tuple, filename: str = 'spectrogram'):
    f, t, Sxx = spectrogram
    plt.figure(figsize=(15, 10))
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig(f'{OUTPUT_PATH}{filename}.png', bbox_inches='tight')


# Read file
filepath = RESOURCES_PATH + 'samples/2020_01_28_3A/2020_01_28_1.wav'  # use OS path objects
sample_rate, data = wavfile.read(filepath)
print(sample_rate)

# Separate channels
data_0 = data[:, 0]
data_1 = data[:, 1]


# sample = data_0[:10000]
# plot_signal(sample)
# spectrogram = spectrogram(sample)
# f, t, Sxx = spectrogram
# plot_spectrogram((f*sample_rate, t/sample_rate, Sxx))




### ADD SECONDS TICKMARKS


def sliding_rfft (signal: np.ndarray, sample_rate: int, folder: str = 'sliding_rfft/', window_width: int = 100, step: int = 1):

    figsize = (15, 10)
    vidsize = (figsize[0]*100, figsize[1]*100)
    fps = 10 #15

    savedir = OUTPUT_PATH + folder
    if os.path.exists(savedir): shutil.rmtree(savedir)
    os.makedirs(savedir)
    signal_width = signal.shape[0]

    # Initialise video
    # rfft_video_path = f'{savedir}rfft-video.avi'
    # rfft_video = cv2.VideoWriter(rfft_video_path,cv2.VideoWriter_fourcc(*'DIVX'), fps, vidsize)
    # signal_video_path = f'{savedir}signal-video.avi'
    # signal_video = cv2.VideoWriter(signal_video_path,cv2.VideoWriter_fourcc(*'DIVX'), fps, vidsize)
    # stack_video_path = f'{savedir}stack-video.mp4'
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # #cv2.VideoWriter_fourcc(*'DIVX')
    # stack_video = cv2.VideoWriter(stack_video_path, fourcc, fps, vidsize)


    window_position = 0
    position_id = 0
    while window_position < signal_width - window_width:

        # Print progress
        progress = round(window_position / signal_width * 100, 1)
        print(f'Progress (window position): {window_position} / {signal_width} ({progress}%)')

        # Generate signal window
        signal_window = signal[window_position: window_position+window_width]

        # Compute rfft
        yf = rfft(signal_window)
        xf = rfftfreq(signal_window.shape[0], 1 / sample_rate)

        # Save rfft
        plt.figure(figsize=figsize)
        plt.ylim((0, 10000000))
        plt.plot(xf, np.abs(yf))
        rfft_path = f'{savedir}rfft-{position_id}.png'
        plt.savefig (rfft_path)
        # plt.clf()
        plt.close()

        # Save signal
        plt.figure(figsize=figsize)
        plt.ylim((-10000, 10000))
        plt.plot(np.arange(window_position, window_position + window_width), signal_window)
        signal_path = f'{savedir}signal-{position_id}.png'
        plt.savefig(signal_path) # bbox_inches='tight'
        # plt.clf()
        plt.close()

        # Save stack
        rfft_image = cv2.imread(rfft_path)
        signal_image = cv2.imread(signal_path)
        stack_image = np.column_stack ((rfft_image, signal_image))
        stack_path = f'{savedir}stack-{position_id}.png'
        cv2.imwrite(stack_path, stack_image)

        cv2.destroyAllWindows()
        window_position += step
        position_id += 1

    
    # stack_video = cv2.VideoWriter(stack_video_path, 0, fps, vidsize)
    # for img in frame_list:
    #     stack_video.write(img)
    # cv2.destroyAllWindows()
    # stack_video.release()

    stack_video_path = f'{savedir}stack-video.mp4'
    os.system('cd  /Users/anatole/Documents/Data Sophia/data2021-bruits-mer')
    os.system(f'ffmpeg -r {fps} -i {savedir}stack-%01d.png -vcodec mpeg4 -y {stack_video_path}')









class VideoGraph:

    def __init__ (self, plots: dict, minmax: dict):
        self.__plots = plots
        self.__minmax = minmax

    @classmethod
    def sliding_rfft (cls, signal: np.ndarray, sample_rate: int, window_width: int = 100, step: int = 1):
        plots = {'signal': list(), 'rfft': list()}
        minmax = {'signal': [None, None], 'rfft': [None, None]}

        signal_length = signal.shape[0]
        window_position = 0
        while window_position < signal_length - window_width:

            # Print progress
            progress = round(window_position / signal_length * 100, 1)
            print(f'Progress (window position): {window_position} / {signal_length} ({progress}%)')

            # Generate signal window
            sample_ids = np.arange(window_position, window_position + window_width)
            signal_window = signal[window_position: window_position+window_width]

            # Save signal window & minmax
            plots['signal'] .append( (sample_ids, signal_window) )
            signal_min = min(signal_window)
            signal_max = max(signal_window)
            if signal_min < minmax['signal'][0]: minmax['signal'][0] = signal_min
            if signal_max > minmax['signal'][1]: minmax['signal'][1] = signal_max

            # Compute rfft
            xf = rfftfreq(signal_window.shape[0], 1 / sample_rate)
            yf = abs( rfft(signal_window) )

            # Save rfft & minmax
            plots['rfft'] .append( (xf, yf) )
            rfft_min = min(yf)
            rfft_max = max(yf)
            if rfft_min < minmax['rfft'][0]: minmax['rfft'][0] = rfft_min
            if rfft_max > minmax['rfft'][1]: minmax['rfft'][1] = rfft_max

            # Move window
            window_position += step

        return VideoGraph(plots=plots, minmax=minmax)


    def save (self, dirpath: str):
        for position_id in range (10):
            pass


    def save_imgs (self):
        pass





# RUN
# window_width = 100000
# step = 10000

window_width = int(1e6)
step = int(25e3)

sliding_rfft (
    signal = data_0,
    sample_rate=sample_rate,
    folder=f'silding_rfft-data_0-width_{window_width}-step_{step}/',
    window_width=window_width,
    step=step
)
