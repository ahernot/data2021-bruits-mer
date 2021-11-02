# import audiofile
# import wave

# from preferences import *

# import os
# from subprocess import check_call
# from tempfile import mktemp
# # from scikits.audiolab import wavread, play
# from scipy.signal import remez, lfilter
# from pylab import *

# # convert mp3, read wav
# mp3filename = RESOURCES_PATH + 'DonneeAccoustique/CRSon/2020_01_25_2 Moteur.mp3'
# wname = mktemp('.wav')
# check_call(['avconv', '-i', mp3filename, wname])
# # sig, fs, enc = wavread(wname)


# with wave.open(wname) as audio_file:
#     x = audio_file.readframes(audio_file.getnframes())
#     print(x)
    
# os.unlink(wname)
