"""
stereo version
"""

import numpy as np

from sys import argv

from scipy import signal
import matplotlib.pyplot as plt



def downsize(img):
	# downsize spectrogram
	# 640x480 to 32x24 (downsized by factor of 20)

	# take a window of numbers, get the mean (ignoring zeros) 
	# consider cases of all zeros in window 
	# return the script 

	# window size: 20x20? 
	
	dims = img.shape
	downsized_img = np.zeros((32,24))
	for i in range(0,dims[0],20):
		for j in range(0,dims[1],20):
			window = img[i:i+1, j:j+1].flatten()
			non_zero = np.delete(window, np.where(window==0))
			if non_zero:
				downsized_img[i,j] = np.mean(non_zero)
			# else, the space remains a zero 
	return downsized_image 


fs = 10e3
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = 500*np.cos(2*np.pi*0.25*time)
carrier = amp * np.sin(2*np.pi*3e3*time + mod)
noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise

f, t, Sxx = signal.spectrogram(x, fs)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
down = downsize(Sxx)
plt.pcolormesh(t, f, down)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
