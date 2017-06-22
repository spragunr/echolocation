import h5py
import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

data = h5py.File(sys.argv[1], "r")

audio = data['audio'].value


for i  in range(36):
    plt.subplot(6,6,i+1)
    f, t, Sxx = signal.spectrogram(audio[i,:], 44100, nperseg=256,noverlap=255)
    plt.pcolormesh(t, f, Sxx)
plt.show()
