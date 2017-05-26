"""
Create wav files from recorded audio stored by ros_record.py

Usage: npz_to_wav.py NPZ_FILE
"""

import sys
import numpy as np
import scipy.io.wavfile

RATE = 44100

data = np.load(sys.argv[1])
audio = data['audio']

for i  in range(audio.shape[0]):
    scipy.io.wavfile.write('fizz{0:0>4}.wav'.format(i), RATE, audio[i,...])
