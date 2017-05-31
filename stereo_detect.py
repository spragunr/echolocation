"""
stereo version
"""

import matplotlib.pyplot as plt
import numpy as np
import os.path

from random import shuffle
from scipy import io, signal
from sys import argv

def downsize(img):
	# downsize depth map 
	# 480x640 to 12x16 (downsized by factor of 40)
	# window size: 40x40 

	orig_dims = img.shape
	ds_dims = (orig_dims[0]/40, orig_dims[1]/40)
	downsized_img = np.zeros(ds_dims)
	for i in range(0,orig_dims[0],40):
		for j in range(0,orig_dims[1],40):
			window = img[i:i+40, j:j+40].flatten()
			non_zero = np.delete(window, np.where(window==0))
			if non_zero.size != 0:
				downsized_img[i/40,j/40] = np.mean(non_zero)  
	return downsized_img

def get_data():
	print "loading 'forensics' data..."
	d1 = np.load('/home/hoangnt/echolocation/data_stereo/forensics.npz')
	print "loading 'isat243' data..."
	d2 = np.load('/home/hoangnt/echolocation/data_stereo/isat243.npz')
	print "loading 'isat246' data..."
	d3 = np.load('/home/hoangnt/echolocation/data_stereo/isat246.npz')
	print "loading 'isat248' data..."
	d4 = np.load('/home/hoangnt/echolocation/data_stereo/isat248.npz')
	print "loading 'office' data..."
	d5 = np.load('/home/hoangnt/echolocation/data_stereo/office.npz')
	print "loading 'stairwell' data..."
	d6 = np.load('/home/hoangnt/echolocation/data_stereo/stairwell.npz')
	print "loading 'spine' data..."
	d7 = np.load('/home/hoangnt/echolocation/data_stereo/spine.npz')
	print "---------------------------------"
	print "All data have been loaded.\n"

	print "fetching audio components..."
	audio1 = d1['audio']
	audio2 = d2['audio']
	audio3 = d3['audio']
	audio4 = d4['audio']
	audio5 = d5['audio']
	audio6 = d6['audio']
	audio7 = d7['audio']

	print "fetching depth components...\n"
	depth1 = d1['depth']
	depth2 = d2['depth']
	depth3 = d3['depth']
	depth4 = d4['depth']
	depth5 = d5['depth']
	depth6 = d6['depth']
	depth7 = d7['depth']

	print "concatenating audio data..."
	audio = np.concatenate((audio1,audio2,audio3,audio4,audio5,audio6,audio7))
	print "concatenating depth data...\n"
	depth = np.concatenate((depth1,depth2,depth3,depth4,depth5,depth6,depth7))
	new_depth = np.empty((depth.shape[0],12,16))
	np.savez('all_orig', audio=audio, depth=depth)
	print "concatenated audio and depth data saved as 'all_orig.npz'\n"
	print "downsizing depth data..."
	for i in range(depth.shape[0]):
		new_depth[i] = downsize(depth[i])
	np.savez('all', audio=audio, depth=new_depth)
	print "concatenated audio and downsized depth data saved as 'all.npz'\n"

def preprocess_data():
	data = np.load('/home/hoangnt/echolocation/all.npz')
	audio = data['audio']
	depth = data['depth'] # shape: 13274, 12, 16

	depth_reshaped = np.reshape(depth, (depth.shape[0],-1)) # shape: 13274, 192
	valid_bool = np.all(depth_reshaped, axis=1)
	valid_depth = depth_reshaped[valid_bool] # removes samples containing zero
	# valid_depth shape: 10291, 192 (22% loss)
#	print "\n\nVALID DEPTH SHAPE:",valid_depth.shape,"\n\n"

	new_audio = audio[valid_bool]
	freq1, time1, spectro1 = signal.spectrogram(new_audio[0,:,0], noverlap=250)
	freq2, time2, spectro2 = signal.spectrogram(new_audio[0,:,1], noverlap=250)
	input_set = np.empty((new_audio.shape[0], spectro1.shape[0], spectro1.shape[1], 2))
	input_set[0,:,:,0] = spectro1
	input_set[0,:,:,1] = spectro2

	for i in range(1,new_audio.shape[0]):
		freq1, time1, spectro1 = signal.spectrogram(new_audio[i,:,0], noverlap=250)
		freq2, time2, spectro2 = signal.spectrogram(new_audio[i,:,1], noverlap=250)
		input_set[i,:,:,0] = spectro1
		input_set[i,:,:,1] = spectro2
#		plt.pcolormesh(spectro2)
#	plt.show()

	return input_set, valid_depth

def main():
	while not os.path.isfile('all.npz'):
		get_data()
	input_set, target = preprocess_data()

main()

