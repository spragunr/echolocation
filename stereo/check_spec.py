from stereo_processing import align_audio
from scipy import signal 
from sys import argv

import h5py
import matplotlib.pyplot as plt
import numpy as np 

path = '/Volumes/seagate/legit_data/'
files = ['isat143a','isat143b','isat231a','isat231b','isat243a','isat243b','isat246a','isat246b','isat246c','isat248a','isat248b','isat248c']

audio_list = []

for i in range(len(files)):
	with h5py.File(path+files[i], 'r') as d:
		audio_list.append(d['audio'].value[3358:3360])
audio_tuple = tuple(audio_list)
audio = np.concatenate(audio_tuple)
aligned_audio = align_audio(5000, audio, plot=True)
print aligned_audio.shape
for row in range(1, aligned_audio.shape[0]):
	plt.subplot(3,1,1)
	plt.plot(aligned_audio[row,:])
	plt.subplot(3,1,2)
	plt.plot(aligned_audio[row,:,0])
	plt.subplot(3,1,3)
	plt.plot(aligned_audio[row,:,1])
	plt.show()

AS = audio.shape
print "creating spectrogram 0"
freq1, time1, spectro1 = signal.spectrogram(audio[0,:,0], noverlap=230)
freq2, time2, spectro2 = signal.spectrogram(audio[0,:,1], noverlap=230)
crop1 = spectro1[30:-50,:]
crop2 = spectro2[30:-50,:]
for row in range(0, AS[0], 2500):
	plt.subplot(2,2,1)
	plt.pcolormesh(time1, freq1, spectro1)
	plt.subplot(2,2,2)
	plt.pcolormesh(time1, freq1[30:-50], crop1)
	plt.subplot(2,2,3)
	plt.pcolormesh(time2, freq2, spectro2)
	plt.subplot(2,2,4)
	plt.pcolormesh(time2, freq2[30:-50], crop2)
	plt.show()
for i in range(1,AS[0]):
	print "creating spectrogram", i
	freq1, time1, spectro1 = signal.spectrogram(audio[i,:,0], noverlap=230)
	freq2, time2, spectro2 = signal.spectrogram(audio[i,:,1], noverlap=230)
	crop1 = spectro1[30:-50,:]
	crop2 = spectro2[30:-50,:]
	for row in range(0, AS[0], 2500):
		plt.subplot(2,2,1)
		plt.pcolormesh(time1, freq1, spectro1)
		plt.subplot(2,2,2)
		plt.pcolormesh(time1, freq1[30:-50], crop1)
		plt.subplot(2,2,3)
		plt.pcolormesh(time2, freq2, spectro2)
		plt.subplot(2,2,4)
		plt.pcolormesh(time2, freq2[30:-50], crop2)
		plt.show()
