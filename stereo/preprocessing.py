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
	train_files = ['/data_stereo/forensics']
	test_files = ['/data_stereo/spine']
	
	if len(argv) != 2: 
		print "usage: preprocessing.py data_set_name"
		return
	
	#both sets contain the corresponding depths too
	spec_sets = argv[1] + '_spec_input.h5' #contains train and test sets with spec input
	da_sets = argv[1] + '_da_input.h5' #contains train and test sets with digital audio input

	if os.path.isfile(spec_sets) and os.path.isfile(da_sets):
		print "preprocessed training and test sets already exist under files %s and %s" %(spec_sets,da_sets)
		return

	train_set, test_set = concatenate(train_files, test_files)
	train_da, test_da = shape_digital_audio(train_set[0], test_set[0])
	train_specs, test_specs = shape_spectrograms(train_set[0], test_set[0])
	save_spec_sets(spec_sets, train_specs, test_specs, train_set[1], test_set[1])
	save_da_sets(da_sets, train_da, test_da, train_set[1], test_set[1])

######################################################
######################################################

def concatenate(train_files, test_files):
	train_set = []
	test_set = []
	train_audio_list = []
	test_audio_list = []
	train_depth_list = []
	test_depth_list = []

	path = os.getcwd()+'/' 
	#path = '/media/hoangnt/seagate/legit_data/'
	#path = '/Volumes/seagate/legit_data/'
	for i in range(len(train_files)):
		print "TRAINING FILES: loading '%s' data..." %train_files[i]
		with np.load(path+train_files[i]+'.npz') as d:
			train_audio_list.append(d['audio'])
			train_depth_list.append(d['depth'])
		#with h5py.File(path+train_files[i], 'r') as d:
			#train_audio_list.append(d['audio'].value)
			#train_depth_list.append(d['depth'].value)
	for i in range(len(test_files)):
		print "TEST FILES: loading '%s' data..." %test_files[i]
		with np.load(path+test_files[i]+'.npz') as d:
			test_audio_list.append(d['audio'])
			test_depth_list.append(d['depth'])
		#with h5py.File(path+test_files[i], 'r') as d:
			#test_audio_list.append(d['audio'].value)
			#test_depth_list.append(d['depth'].value)
	print "---------------------------------"
	print "data loading complete\n"
	
	print "concatenating training audio..."
	train_audio_tuple = tuple(train_audio_list)
	train_audio = np.concatenate(train_audio_tuple)
	train_set.append(train_audio)
	print "concatenating test audio..."
	test_audio_tuple = tuple(test_audio_list)
	test_audio = np.concatenate(test_audio_tuple)
	test_set.append(test_audio)

	train_depth = np.empty((train_audio.shape[0],12,16))
	counter = 0
	for d_file in train_depth_list:
		for d_map in d_file:
			print "TRAINING: downsizing depth map", counter
			train_depth[counter] = downsize(d_map)
			counter += 1
	train_set.append(train_depth)
	test_depth = np.empty((test_audio.shape[0],12,16))
	counter = 0
	for d_file in test_depth_list:
		for d_map in d_file:
			print "TEST: downsizing depth map", counter
			test_depth[counter] = downsize(d_map)
			counter += 1
	test_set.append(train_depth)

	return train_set, test_set

######################################################

def downsize(depth_map, factor=40):
	'''
	@PURPOSE: downsizes an image by a factor of its dimensions
	@PARAMS: img - [numpy array] image to downsize 
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
				downsized_map[i/factor,j/factor] = np.mean(non_zero)
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
	top_crop = 68
	bot_crop = -34

	train_AS = train_audio.shape
	print "TRAINING: creating spectrogram 0" 
	freq1, time1, spectro1 = signal.spectrogram(train_audio[0,:,0], noverlap=230)
	freq2, time2, spectro2 = signal.spectrogram(train_audio[0,:,1], noverlap=230)
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

def save_spec_sets(spec_sets, train_specs, test_specs, train_depths, test_depths):
	print "saving sets (with spectrograms as input)..."
	with h5py.File(spec_sets, 'w') as sets:
		sets.create_dataset('train_specs', data=train_specs)
		sets.create_dataset('train_depths', data=train_depths)
		sets.create_dataset('test_specs', data=test_specs)
		sets.create_dataset('test_depths', data=test_depths)
	print "training and test sets saved as '%s'" %spec_sets

######################################################

def save_da_sets(da_sets, train_da, test_da, train_depths, test_depths):
	print "saving sets (with digital audio as input)..."
	with h5py.File(da_sets, 'w') as sets:
		sets.create_dataset('train_da', data=train_da)
		sets.create_dataset('train_depths', data=train_depths)
		sets.create_dataset('test_da', data=test_da)
		sets.create_dataset('test_depths', data=test_depths)
	print "training and test sets saved as '%s'" %da_sets

######################################################
######################################################

if __name__ == "__main__":
	main()
