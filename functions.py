import os
import subprocess

from preferences import *


def convert_to_wav (filepath: str):
    output = 'output'
    subprocess.call(['ffmpeg', '-i', filepath, '-vn', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '44100', '-f', 'wav', f'{output}.wav'])
