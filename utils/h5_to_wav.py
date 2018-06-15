"""
Create wav files from recorded audio stored by ros_record.py

Usage: npz_to_wav.py NPZ_FILE
"""

import h5py
import sys
import numpy as np
import scipy.io.wavfile

RATE = 44100

data = h5py.File(sys.argv[1], "r")
audio = data['audio'].value

for i  in range(audio.shape[0]):
    scipy.io.wavfile.write('fizz{0:0>4}.wav'.format(i), RATE, audio[i,...])
