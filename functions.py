import os
import subprocess

from preferences import *


def convert_to_wav (filepath: str):
    output = 'output'
    subprocess.call(['ffmpeg', '-i', filepath, '-vn', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '44100', '-f', 'wav', f'{output}.wav'])



class BiDict (dict):

    def __init__ (self, generator):
        self.generator = generator  # list of tuples (primary_key, secondary_key)
        self.dict = super().__init__(generator)
        self.invdict = dict([(b, a) for (a, b) in generator])

    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

    def __getitem__(self, __k):
        return super().__getitem__(__k)

    def __call__(self, __k):
        return self.invdict.__getitem__(__k)

    def __repr__ (self):  # doesn't work
        return str(self.generator)

