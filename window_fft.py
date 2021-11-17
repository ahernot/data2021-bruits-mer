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



class VideoGraph:

    def __init__ (self, signal:np.ndarray, sample_rate:int, window_width:int=100, window_step:int=1):
        self.signal = signal
        self.signal_length = signal.shape[0]
        self.sample_rate = sample_rate

        self.window_width = window_width
        self.window_step = window_step
        self.window_nb = math.floor((self.signal_length - window_width) / window_step) + 1

        self.__plots = dict()
        self.__minmax = dict()
        self.__labels = dict()
        self.__ticks = dict()

    def __repr__ (self):
        desc_list = [
            'VideoGraph object',
            f' Signal length (samples): {self.signal_length}',
            f' Sampling rate (Hz):      {self.sample_rate}',
            '',
            f' Window width (samples):  {self.window_width}',
            f' Window width (seconds):  {round(self.window_width / self.sample_rate, 3)}',
            f' Window step  (samples):  {self.window_step}',
            f' Window number:           {self.window_nb}',
            '',
            f' Plots:                   {self.plot_names()}',
        ]
        return '\n'.join(desc_list)

    def plot_names (self):
        return list(self.__plots.keys())


    def plot_signal (self, plot_id:str='signal', fix_scale_y=True):
        print('Plotting signal')

        self.__plots  [plot_id] = list()
        self.__minmax [plot_id] = {'x': list((None, None)), 'y': list((None, None))}
        self.__labels [plot_id] = ('Time [sec]', 'Amplitude')
        self.__ticks  [plot_id] = {'x': [round(i/self.sample_rate, 1) for i in range(0, self.signal_length, 50000)], 'y': None}
        
        for pos_id in range (self.window_nb):
            window_start = pos_id * self.window_step
            window_stop = window_start + self.window_width

            # Print progress
            progress = round(pos_id / self.window_nb * 100, 1)
            print(f'Progress: {pos_id} / {self.window_nb} ({progress}%)')

            # Generate signal window
            sample_ids = np.arange(window_start, window_stop)
            signal_window = self.signal[window_start : window_stop]

            # Save signal window
            self.__plots[plot_id] .append( (sample_ids, signal_window) )

            # Save minmax
            if fix_scale_y:
                signal_min = min(signal_window)
                signal_max = max(signal_window)
                if signal_min < self.__minmax[plot_id]['y'][0]: self.__minmax[plot_id]['y'][0] = signal_min
                if signal_max > self.__minmax[plot_id]['y'][1]: self.__minmax[plot_id]['y'][1] = signal_max
        if not fix_scale_y: self.__minmax[plot_id]['y'] = list((None, None))


    def plot_rfft (self, plot_id:str='rfft', fix_scale_x=True, fix_scale_y=True, freq_cutoff_min=0, freq_cutoff_max=float('inf'), amplitude_threshold=float('-inf')):
        
        # Plot signal
        if 'signal' not in self.plot_names(): self.plot_signal(plot_id='signal')

        print('Plotting RFFT')

        self.__plots  [plot_id] = list()
        self.__minmax [plot_id] = {'x': list((freq_cutoff_min, freq_cutoff_max)), 'y': list((None, None))}
        self.__labels [plot_id] = ('Frequency [Hz]', 'Amplitude')
        self.__ticks  [plot_id] = {'x': list(range(freq_cutoff_min, freq_cutoff_max, 2500)), 'y': None}

        for pos_id in range (self.window_nb):

            # Print progress
            progress = round(pos_id / self.window_nb * 100, 1)
            print(f'Progress: {pos_id} / {self.window_nb} ({progress}%)')

            # Get window
            signal_window = self.__plots['signal'][pos_id][1]

            # Compute rfft
            xf = rfftfreq(self.window_width, 1 / sample_rate)
            yf = np.abs( rfft(signal_window) )
            yf = yf * (xf > freq_cutoff_min) * (xf < freq_cutoff_max)  # apply cutoff
            yf = yf * (yf - amplitude_threshold > 0.)  # apply thresholding

            # Save rfft
            self.__plots[plot_id] .append( (xf, yf) )

            # Save minmax
            if fix_scale_y:
                rfft_min = min(yf)
                rfft_max = max(yf)
                if amplitude_threshold==float('-inf') and rfft_min < self.__minmax[plot_id]['y'][0]: self.__minmax[plot_id]['y'][0] = rfft_min
                if rfft_max > self.__minmax[plot_id]['y'][1]: self.__minmax[plot_id]['y'][1] = rfft_max

        if amplitude_threshold!=float('-inf'): self.__minmax[plot_id]['y'][0] = amplitude_threshold
        if not fix_scale_x: self.__minmax[plot_id]['x'] = list((None, None))
        if not fix_scale_y: self.__minmax[plot_id]['y'] = list((None, None))


    def save (self, folder=None, plots='all', figsize=(15, 10), video_fps=10):
        # Specify plots to get a specific order

        print('Saving frames')

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

            # Print progress
            progress = round(pos_id / self.window_nb * 100, 1)
            print(f'Progress: {pos_id} / {self.window_nb} ({progress}%)')

            stack_images = list()
            for plot_id in selected_ids:
                x, y = self.__plots [plot_id] [pos_id]

                # Plot image
                plt.figure(figsize=figsize)
                if fix_scale_x and self.__minmax[plot_id]['x'] not in (None, list((None, None))): plt.xlim( (self.__minmax[plot_id]['x'][0], self.__minmax[plot_id]['x'][1]) )
                if fix_scale_y and self.__minmax[plot_id]['y'] not in (None, list((None, None))): plt.ylim( (self.__minmax[plot_id]['y'][0], self.__minmax[plot_id]['y'][1]) )
                if self.__ticks [plot_id]['x'] != None: plt.xticks(self.__ticks[plot_id]['x'])
                if self.__ticks [plot_id]['y'] != None: plt.yticks(self.__ticks[plot_id]['y'])
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

        print('Saving videos')

        # Save plot videos
        for plot_id in selected_ids:
            path = os.path.join(savedir, f'{plot_id}.mp4')
            image_path = os.path.join(savedir, f'{plot_id}-%01d.png')
            os.system(f'ffmpeg -r {video_fps} -i {image_path} -vcodec mpeg4 -y {path}')
        
        # Save stack video
        path = os.path.join(savedir, f'stack.mp4')
        stack_path = os.path.join(savedir, 'stack-%01d.png')
        os.system(f'ffmpeg -r {video_fps} -i {stack_path} -vcodec mpeg4 -y {path}')


window_width = int(1e6)
window_step  = int(1e4)

video_graph = VideoGraph (
    signal=data_0,
    sample_rate=sample_rate,
    window_width=window_width,
    window_step=window_step
)

print(video_graph)
print('\n')

video_graph.plot_rfft(freq_cutoff_min=1000, freq_cutoff_max=25000)
video_graph.save(plots=['rfft', 'signal'])
