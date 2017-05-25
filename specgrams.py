import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

data = np.load(sys.argv[1])

audio = data['audio']


for i  in range(36):
    plt.subplot(6,6,i+1)
    f, t, Sxx = signal.spectrogram(audio[i,:], 44100, nperseg=256,noverlap=255)
    plt.pcolormesh(t, f, Sxx)
plt.show()
