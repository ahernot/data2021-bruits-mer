import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import select

import scipy
from scipy.io import wavfile
from scipy.signal import cwt, ricker, spectrogram
from scipy.fft import rfft, rfftfreq

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

        print(yf.shape)

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







from typing import Callable

class VideoGraph:

    def __init__ (self, signal: np.ndarray, sample_rate: int, window_width: int = 100, window_step: int = 1):
        self.signal = signal
        self.signal_length = signal.shape[0]
        self.sample_rate = sample_rate

        self.window_width = window_width
        self.window_step = window_step
        self.window_nb = math.floor((self.signal_length - window_width) / window_step) + 1

        self.__plots = dict()
        self.__minmax = dict()
        self.__labels = dict()

        self.__plot_signal(plot_id='signal')

    def plot_names (self):
        return list(self.__plots.keys())


    def __plot_signal (self, plot_id: str = 'signal'):
        self.__plots  [plot_id] = list()
        self.__minmax [plot_id] = list((0., 0.))
        self.__labels [plot_id] = ('Time [sec]', 'Amplitude')
        
        for pos_id in range (self.window_nb):
            window_start = pos_id * self.window_step
            window_stop = window_start + self.window_width

            # Print progress
            progress = round(pos_id / self.window_nb * 100, 1)
            print(f'Progress: {pos_id} / {self.window_nb} ({progress}%)')

            # Generate signal window
            sample_ids = np.arange(window_start, window_stop)
            signal_window = self.signal[window_start : window_stop]

            self.__plots[plot_id] .append( (sample_ids, signal_window) )
            signal_min = min(signal_window)
            signal_max = max(signal_window)
            if signal_min < self.__minmax[plot_id][0]: self.__minmax[plot_id][0] = signal_min
            if signal_max > self.__minmax[plot_id][1]: self.__minmax[plot_id][1] = signal_max


    def plot_rfft (self, plot_id: str = 'rfft'):
        self.__plots  [plot_id] = list()
        self.__minmax [plot_id] = list((0., 0.))
        self.__labels [plot_id] = ('Frequency [Hz]', 'Amplitude')

        for pos_id in range (self.window_nb):

            # Print progress
            progress = round(pos_id / self.window_nb * 100, 1)
            print(f'Progress: {pos_id} / {self.window_nb} ({progress}%)')

            # Get window
            signal_window = self.__plots['signal'][pos_id][1]

            # Compute rfft
            xf = rfftfreq(self.window_width, 1 / sample_rate)
            yf = np.abs( rfft(signal_window) )

            # Save rfft & minmax
            self.__plots[plot_id] .append( (xf, yf) )
            rfft_min = min(yf)
            rfft_max = max(yf)
            if rfft_min < self.__minmax[plot_id][0]: self.__minmax[plot_id][0] = rfft_min
            if rfft_max > self.__minmax[plot_id][1]: self.__minmax[plot_id][1] = rfft_max


    def save (self, folder = None, plots = 'all'):
        # Specify plots to get a specific order

        print('Saving')

        figsize = (15, 10)
        fps = 10

        # Select plots
        selected_ids = list()
        if plots == 'all':
            selected_ids = list(self.__plots.keys())
        elif type(plots) == str:
            selected_ids = [self.__plots[plots], ]
        elif type(plots) == list:
            for key in plots:  # Retain order
                if key in self.__plots.keys(): selected_ids.append(key)
        else: return

        # Create folder
        if not folder:
            folder = f'sliding-graph_width={self.window_width}-step={self.window_step}-ids={"_".join(selected_ids)}'
        
        savedir = os.path.join(OUTPUT_PATH, folder)
        if os.path.exists(savedir): shutil.rmtree(savedir)
        os.makedirs(savedir)

        for pos_id in range (self.window_nb):

            stack_images = list()
            for plot_id in selected_ids:
                x, y = self.__plots [plot_id] [pos_id]

                # Plot image
                plt.figure(figsize=figsize)
                plt.ylim((self.__minmax[plot_id][0], self.__minmax[plot_id][1]))
                plt.plot(x, y)

                plt.xlabel(self.__labels[plot_id][0])
                plt.ylabel(self.__labels[plot_id][1])

                # Save image
                path = os.path.join(savedir, f'{plot_id}-{pos_id}.png')
                plt.savefig(path)
                plt.close()

                # Read image
                image = cv2.imread(path)
                stack_images.append(image)

            # Save stack image
            stack_image = np.column_stack (stack_images)
            stack_path = os.path.join(savedir, f'stack-{pos_id}.png')
            cv2.imwrite(stack_path, stack_image)

        os.system('cd  /Users/anatole/Documents/Data Sophia/data2021-bruits-mer')

        # Save plot videos
        for plot_id in selected_ids:
            path = os.path.join(savedir, f'{plot_id}.mp4')
            image_path = os.path.join(savedir, f'{plot_id}-%01d.png')
            os.system(f'ffmpeg -r {fps} -i {image_path} -vcodec mpeg4 -y {path}')
        
        # Save stack video
        path = os.path.join(savedir, f'stack.mp4')
        stack_path = os.path.join(savedir, 'stack-%01d.png')
        os.system(f'ffmpeg -r {fps} -i {stack_path} -vcodec mpeg4 -y {path}')



window_width = int(1e6)
window_step  = int(1e6)#int(1e5)

# sliding_rfft (
#     signal=data_0,
#     sample_rate=sample_rate,
#     folder='sliding_rfft/',
#     window_width=window_width,
#     step=window_step
# )

video_graph = VideoGraph (
    signal=data_0,
    sample_rate=sample_rate,
    window_width=window_width,
    window_step=window_step
)

video_graph.plot_rfft()
video_graph.save(plots=['rfft', 'signal'])


