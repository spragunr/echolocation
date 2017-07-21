"""
Program handles preprocessing of neural network input data. 

written by Nhung Hoang, May-June 2017
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os 
import os.path

from random import shuffle
from scipy import signal
from sys import argv, exit

#####################################################
#####################################################

def main():
	'''
	files to change as necessary: 
	> train_files - 
	> test_files - 
	'''
	train_files = ['ball_train1.h5', 'ball_train2.h5', 'ball_train3.h5', 'ball_train4.h5', 'ball_train5.h5']
	test_files = ['ball_test1.h5', 'ball_test2.h5', 'ball_test3.h5']

	if len(argv) != 2: 
		print "usage: preprocessing.py collective_data_name"
		return
	
	sets_file = argv[1] + '_sets.h5' #contains the components of the training and test sets 

	if os.path.isfile(sets_file):
		print "preprocessed training and test sets already exist under file '%s'" %sets_file
		return

	train_set, test_set = concatenate(train_files, test_files) # [0]audio [1]depth [2]rgb
	train_da, test_da = shape_digital_audio(train_set[0], test_set[0])
	train_specs, test_specs = shape_spectrograms(train_set[0], test_set[0])
	save_sets(sets_file, train_da, test_da, train_specs, test_specs, train_set[1], test_set[1], train_set[2], test_set[2])

######################################################
######################################################

def concatenate(train_files, test_files):
	train_set = []
	test_set = []
	train_audio_list = []
	test_audio_list = []
	train_depth_list = []
	test_depth_list = []
	train_rgb_list = []
	test_rgb_list = []

	path = os.getcwd()+'/ball_data/' 
	#path = '/media/hoangnt/seagate/legit_data/'
	#path = '/Volumes/seagate/legit_data/'
	for i in range(len(train_files)):
		print "TRAINING FILES: loading '%s' data..." %train_files[i]
		#with np.load(path+train_files[i]+'.npz') as d:
			#train_audio_list.append(d['audio'])
			#train_depth_list.append(d['depth'])
			#train_rgb_list.append(d['rgb'])
		with h5py.File(path+train_files[i], 'r') as d:
			train_audio_list.append(d['audio'].value)
			train_depth_list.append(d['depth'].value)
			train_rgb_list.append(d['rgb'].value)
	for i in range(len(test_files)):
		print "TEST FILES: loading '%s' data..." %test_files[i]
		#with np.load(path+test_files[i]+'.npz') as d:
			#test_audio_list.append(d['audio'])
			#test_depth_list.append(d['depth'])
			#test_rgb_list.append(d['rgb'])
		with h5py.File(path+test_files[i], 'r') as d:
			test_audio_list.append(d['audio'].value)
			test_depth_list.append(d['depth'].value)
			test_rgb_list.append(d['rgb'].value)
	print "---------------------------------"
	print "data loading complete\n"
	
	## AUDIO ##
	print "concatenating training audio..."
	train_audio_tuple = tuple(train_audio_list)
	train_audio = np.concatenate(train_audio_tuple)
	train_set.append(train_audio)
	print "concatenating test audio..."
	test_audio_tuple = tuple(test_audio_list)
	test_audio = np.concatenate(test_audio_tuple)
	test_set.append(test_audio)

	## DEPTH ##
	train_depth = np.empty((train_audio.shape[0],12,16))
	counter = 0
	for d_file in train_depth_list:
		for d_map in d_file:
			print "TRAINING: downsizing depth map", counter
			train_depth[counter] = downsize(d_map, method='min')
			counter += 1
	train_depth_reshaped = np.reshape(train_depth, (train_depth.shape[0],-1))
	train_set.append(train_depth_reshaped)
	test_depth = np.empty((test_audio.shape[0],12,16))
	counter = 0
	for d_file in test_depth_list:
		for d_map in d_file:
			print "TEST: downsizing depth map", counter
			test_depth[counter] = downsize(d_map, method='min')
			counter += 1
	test_depth_reshaped = np.reshape(test_depth, (test_depth.shape[0],-1))
	test_set.append(test_depth_reshaped)

	## RGB ##
	train_rgb = np.empty((train_audio.shape[0],24,32))
	counter = 0
	for rgb_file in train_rgb_list:
		for rgb_map in rgb_file:
			print "TRAINING: downsizing rgb map", counter
			train_rgb[counter] = downsize(rgb_map, factor=20)
			counter += 1
	train_rgb = np.reshape(train_rgb, (train_audio.shape[0],24,32,1))
	train_set.append(train_rgb)
	test_rgb = np.empty((test_audio.shape[0],24,32))
	counter = 0
	for rgb_file in test_rgb_list:
		for rgb_map in rgb_file:
			print "TEST: downsizing rgb map", counter
			test_rgb[counter] = downsize(rgb_map, factor=20)
			counter += 1
	test_rgb = np.reshape(test_rgb, (test_audio.shape[0],24,32,1))
	test_set.append(test_rgb)

	return train_set, test_set

######################################################

def downsize(depth_map, method='mean', factor=40):
	'''
	@PURPOSE: downsizes an image by a factor of its dimensions
	@PARAMS: img - [numpy array] image to downsize 
                 method - [string] 'mean' or 'min'
	         factor - [int] factor to downsize image by (default is 40)
	@RETURN: [numpy array] downsized image
	'''
	orig_dims = depth_map.shape
	ds_dims = (orig_dims[0]/factor, orig_dims[1]/factor)
	downsized_map = np.zeros(ds_dims)
	for i in range(0,orig_dims[0],factor):
		for j in range(0,orig_dims[1],factor):
			window = depth_map[i:i+factor, j:j+factor].flatten()
			non_zero = np.delete(window, np.where(window==0))
			if non_zero.size != 0:
                                if method == 'mean':
				        downsized_map[i/factor,j/factor] = np.mean(non_zero)
                                elif method == 'min':
				        downsized_map[i/factor,j/factor] = np.min(non_zero)
                                else:
                                        print "UNRECOGNIZED DOWNSIZE METHOD"
	return downsized_map

######################################################

def shape_digital_audio(train_audio, test_audio):
	print "joining digital audio pairs side by side..."
	train_AS = train_audio.shape
	train_input = np.empty((train_AS[0],train_AS[1]*2,1))
	counter = 0
	for j in train_audio:
		combined = np.concatenate((j[:,0],j[:,1]))
		combined = np.reshape(combined,(train_AS[1]*2,1))
		train_input[counter] = combined 
		counter += 1

	test_AS = test_audio.shape
	test_input = np.empty((test_AS[0],test_AS[1]*2,1))
	counter = 0
	for j in test_audio:
		combined = np.concatenate((j[:,0],j[:,1]))
		combined = np.reshape(combined,(test_AS[1]*2,1))
		test_input[counter] = combined 
		counter += 1

	return train_input, test_input

######################################################

def shape_spectrograms(train_audio, test_audio):
        min_freq=7000
        max_freq=17000

	train_AS = train_audio.shape
	print "TRAINING: creating spectrogram 0" 
	freq1, time1, spectro1 = signal.spectrogram(train_audio[0,:,0], noverlap=230, fs=44100)
	freq2, time2, spectro2 = signal.spectrogram(train_audio[0,:,1], noverlap=230, fs=44100)
        top_crop = np.where(freq1 > min_freq)[0][0]
        bot_crop = np.where(freq1 > max_freq)[0][0]
	crop1 = spectro1[top_crop:bot_crop,:]
	crop2 = spectro2[top_crop:bot_crop,:]
	combined = np.concatenate((crop1,crop2),axis=1)
	dims = combined.shape
	train_input = np.empty((train_AS[0], dims[0], dims[1], 1))
	combined = np.reshape(combined, (dims[0], dims[1], 1))
	train_input[0,:,:,:] = combined
	for i in range(1,train_AS[0]):
		print "TRAINING: creating spectrogram", i
		freq1, time1, spectro1 = signal.spectrogram(train_audio[i,:,0], noverlap=230)
		freq2, time2, spectro2 = signal.spectrogram(train_audio[i,:,1], noverlap=230)
		crop1 = spectro1[top_crop:bot_crop,:]
		crop2 = spectro2[top_crop:bot_crop,:]
		combined = np.concatenate((crop1,crop2),axis=1)
		combined = np.reshape(combined, (dims[0], dims[1], 1))
		train_input[i,:,:,:] = combined

	test_AS = test_audio.shape
	print "TEST: creating spectrogram 0" 
	freq1, time1, spectro1 = signal.spectrogram(test_audio[0,:,0], noverlap=230)
	freq2, time2, spectro2 = signal.spectrogram(test_audio[0,:,1], noverlap=230)
	crop1 = spectro1[top_crop:bot_crop,:]
	crop2 = spectro2[top_crop:bot_crop,:]
	combined = np.concatenate((crop1,crop2),axis=1)
	dims = combined.shape
	test_input = np.empty((test_AS[0], dims[0], dims[1], 1))
	combined = np.reshape(combined, (dims[0], dims[1], 1))
	test_input[0,:,:,:] = combined
	for i in range(1,test_AS[0]):
		print "TEST: creating spectrogram", i
		freq1, time1, spectro1 = signal.spectrogram(test_audio[i,:,0], noverlap=230)
		freq2, time2, spectro2 = signal.spectrogram(test_audio[i,:,1], noverlap=230)
		crop1 = spectro1[top_crop:bot_crop,:]
		crop2 = spectro2[top_crop:bot_crop,:]
		combined = np.concatenate((crop1,crop2),axis=1)
		combined = np.reshape(combined, (dims[0], dims[1], 1))
		test_input[i,:,:,:] = combined

	return train_input, test_input

######################################################

def save_sets(sets_file, train_da, test_da, train_specs, test_specs, train_depths, test_depths, train_rgb, test_rgb):
	print "saving sets (with spectrograms as input)..."
	with h5py.File(sets_file, 'w') as sets:
		sets.create_dataset('train_da', data=train_da)
		sets.create_dataset('test_da', data=test_da)
		sets.create_dataset('train_specs', data=train_specs)
		sets.create_dataset('test_specs', data=test_specs)
		sets.create_dataset('train_depths', data=train_depths)
		sets.create_dataset('test_depths', data=test_depths)
		sets.create_dataset('train_rgb', data=train_rgb)
		sets.create_dataset('test_rgb', data=test_rgb)
	print "training and test sets saved as '%s'" %sets_file

######################################################
######################################################

if __name__ == "__main__":
	main()
