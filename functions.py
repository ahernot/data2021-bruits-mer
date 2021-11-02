import os
import subprocess

from preferences import *


# filepath = RESOURCES_PATH + 'raw-data/2020_14_02_3A/2020_14_02_1.wav'  # use os path objects

def convert_to_wav (filepath: str):
    output = 'output'
    subprocess.call(['ffmpeg', '-i', filepath, '-vn', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '44100', '-f', 'wav', f'{output}.wav'])
