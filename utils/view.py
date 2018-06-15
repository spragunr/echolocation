import h5py
import sys
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

data = h5py.File(sys.argv[1], 'r')
i = 0
while True:
    entered = input("which? (enter for next)")
    if entered == "":
        i += 1
    else:
        i = entered
    rows = 5
    for row in range(rows):  
        plt.subplot(rows,5,1 + 5 * row)
        rgb = plt.imshow(data['rgb'][i + row,...])
        
        plt.subplot(rows,5,2 + 5 * row)
        plt.plot(data['audio_aligned'][i + row,:,:])
        plt.ylim([-2**15, 2**15])
        
        plt.subplot(rows,5,3 + 5 * row)
        f, t, Sxx = signal.spectrogram(data['audio_aligned'][i + row,:,0], 44100, nperseg=256,
                                       noverlap =255)
        plt.pcolormesh(t, f, np.log(1 + Sxx))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        
        plt.subplot(rows,5,4 + 5 * row)
        f, t, Sxx = signal.spectrogram(data['audio_aligned'][i + row,:,1], 44100, nperseg=256,
                                       noverlap =255)
        plt.pcolormesh(t, f, np.log(1 + Sxx))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        
        plt.subplot(rows,5,5 + 5 * row)
        plt.imshow(data['depth'][i + row,...])
    plt.show()




