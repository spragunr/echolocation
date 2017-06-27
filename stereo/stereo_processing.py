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
	> data_file - collective audio-depth data from device
	> spec_file - input spectrograms, needed for spectrogram version of NN only
	> sets_file - training and test set of input-depth pairs 
	'''
	data_file = 'cs_free.h5' #'bat_angled.h5' 
	spec_file = 'input_specA_cs_free_aligned.h5'
	sets_file = 'sets_specA_cs_free_aligned.h5'

	if len(argv) != 2: 
		print "usage: stereo_processing.py input_type\n"
		print " "*6, "input_type - specA for side by side spectrograms"
		print " "*19, "specB for back to back spectrograms"
		print " "*19, "rawB for side to side raw audio samples"
		print " "*19, "rawB for back to baack raw audio samples"
		return

	if not os.path.isfile(data_file):
		get_data(data_file)
	input_set, target = preprocess_data(argv[1], data_file, spec_file)
	split_data(input_set, target, sets_file)

######################################################
######################################################

def downsize(img, factor=40):
	'''
	@PURPOSE: downsizes an image by a factor of its dimensions
	@PARAMS: img - [numpy array] image to downsize 
					 factor - [int] factor to downsize image by (default is 40)
	@RETURN: [numpy array] downsized image
	'''
	orig_dims = img.shape
	ds_dims = (orig_dims[0]/factor, orig_dims[1]/factor)
	downsized_img = np.zeros(ds_dims)
	for i in range(0,orig_dims[0],factor):
		for j in range(0,orig_dims[1],factor):
			window = img[i:i+factor, j:j+factor].flatten()
			non_zero = np.delete(window, np.where(window==0))
			if non_zero.size != 0:
				downsized_img[i/factor,j/factor] = np.mean(non_zero)
	return downsized_img

######################################################

def get_data(filename):
	'''
	@PURPOSE: joins individual files of collected audio-depth data, 
						save a file of audio-downsized depth pairs
	@PARAMS: filename - [string] name of file where data is to be found
	@RETURN: none
	'''

	files = ['forensics', 'isat243', 'isat246', 'isat248', 'office', 'stairwell', 'spine'] 
#	files = ['test5','test6','test7','test8']
#	files = ['leaves1','leaves2', 'hole1', 'hole2', 'hole3']
#	files = ['new2', 'new3', 'new4', 'new5', 'new6', 'new7', 'new8', 'new9', 'new10', 'new11', 'new12', 'new13', 'new14', 'new15', 'new16']
	
	audio_list = []
	depth_list = []
	path = os.getcwd()+'/' 
	for i in range(len(files)):
		print "loading '%s' data..." %files[i]
		with np.load(path+'/data_stereo3/'+files[i]+'.npz') as d:
			audio_list.append(d['audio'])
			depth_list.append(d['depth'])
	print "---------------------------------"
	print "data loading complete\n"

	print "concatenating audio data..."
	audio_tuple = tuple(audio_list)
	audio = np.concatenate(audio_tuple)

	print "downsizing depth data..."
	new_depth = np.empty((audio.shape[0],12,16))
	counter = 0
	for d_file in depth_list:
		for d_map in d_file:
			new_depth[counter] = downsize(d_map)
			counter += 1

	print "saving concatenated audio and downsized depth data saved as '%s'...\n" %filename
	with h5py.File(filename, 'w') as hf:
		hf.create_dataset('audio', data=audio)
		hf.create_dataset('depth', data=new_depth)
	
######################################################

def preprocess_data(input_type, data_file, spec_file):
	print "preprocessing data..."
	path = os.getcwd()+'/'
	with h5py.File(path+data_file, 'r') as data:
		audio = data['audio'][:]	
		depth = data['depth'][:] # shape: 13274, 12, 16
	audio = align_audio(5000, audio)

	if input_type[:-1] == 'spec':
		if os.path.isfile(spec_file):
			print "fetching spectrogram array from '%s'..." %spec_file
			with h5py.File(path+spec_file, 'r') as sgrams:
				input_set = sgrams['spectrograms'][:]
		elif input_type[-1] == 'A':
			input_set = side_by_side_spectrograms(audio)
		elif input_type[-1] == 'B': 
			input_set = back_to_back_spectrograms(audio)
		else:
			print "ERROR: invalid input type"
			exit()
	elif input_type[:-1] == 'raw':
		if input_type[-1] == 'A':
			input_set = side_by_side_raw_audio_samples(audio)
		elif input_type[-1] == 'B':
			input_set = back_to_back_raw_audio_samples(audio)
		else:
			print "ERROR: invalid input type"
			exit()
	else:
		print "ERROR: invalid input type"
		exit()

	depth_reshaped = np.reshape(depth, (depth.shape[0],-1)) # shape: 13274, 192
	return input_set, depth_reshaped	

######################################################

def side_by_side_spectrograms(audio):
	AS = audio.shape
	print "creating spectrogram 0" 
	freq1, time1, spectro1 = signal.spectrogram(audio[0,:,0], noverlap=230)
	freq2, time2, spectro2 = signal.spectrogram(audio[0,:,1], noverlap=230)
	crop1 = spectro1[65:-35,:]
	crop2 = spectro2[65:-35,:]
	combined = np.concatenate((crop1,crop2),axis=1)
	dims = combined.shape
	input_set = np.empty((AS[0], dims[0], dims[1], 1))
	combined = np.reshape(combined, (dims[0], dims[1], 1))
	input_set[0,:,:,:] = combined
	for i in range(1,AS[0]):
		print "creating spectrogram", i
		freq1, time1, spectro1 = signal.spectrogram(audio[i,:,0], noverlap=230)
		freq2, time2, spectro2 = signal.spectrogram(audio[i,:,1], noverlap=230)
		crop1 = spectro1[65:-35,:]
		crop2 = spectro2[65:-35,:]
		combined = np.concatenate((crop1,crop2),axis=1)
		combined = np.reshape(combined, (dims[0], dims[1], 1))
		input_set[i,:,:,:] = combined
	print "saving array of spectrograms as '%s'..." %spec_file
	with h5py.File(spec_file, 'w') as sgrams:
		sgrams.create_dataset('spectrograms', data=input_set)
	return input_set

######################################################

def back_to_back_spectrograms(audio):
	AS = audio.shape
	print "creating spectrogram 0" 
	freq1, time1, spectro1 = signal.spectrogram(audio[0,:,0], noverlap=230)
	freq2, time2, spectro2 = signal.spectrogram(audio[0,:,1], noverlap=230)
	crop1 = spectro1[65:-35,:]
	crop2 = spectro2[65:-35,:]
	dims = crop1.shape
	input_set = np.empty((AS[0], dims[0], dims[1], 2))
	input_set[0,;,;,0] = spectro1
	input_set[0,;,;,1] = spectro2
	for i in range(1,AS[0]):
		print "creating spectrogram", i 
		freq1, time1, spectro1 = signal.spectrogram(audio[i,:,0], noverlap=230)
		freq2, time2, spectro2 = signal.spectrogram(audio[i,:,1], noverlap=230)
		crop1 = spectro1[65:-35,:]
		crop2 = spectro2[65:-35,:]
		input_set[i,;,;,0] = spectro1
		input_set[i,;,;,1] = spectro2
	IS = input_set.shape
	input_set = np.reshape(input_set, (IS[0],IS[1],IS[2],IS[3],1))
	print "saving array of spectrograms as '%s'..." %spec_file
	with h5py.File(spec_file, 'w') as sgrams:
		sgrams.create_dataset('spectrograms', data=input_set)
	return input_set

######################################################

def side_by_side_raw_audio_samples(audio):
	AS = audio.shape
	print "joining stereo audio samples..."
	input_set = np.empty((AS[0],AS[1]*2,1))
	counter = 0
	for j in audio:
		combined = np.concatenate((j[:,0],j[:,1]))
		combined = np.reshape(combined,(AS[1]*2,1))
		input_set[counter] = combined 
		counter += 1
	return input_set

######################################################

def back_to_back_raw_audio_samples(audio):
	AS = audio.shape
	input_set = np.reshape(audio, (AS[0],AS[1],AS[2],1))
	return input_set

######################################################

def split_data(x, y, sets_name):
	'''
	@PURPOSE: splits data into training and test sets, saves sets collectively as one file
	@PARAMS: x - [numpy array] either digitalized audio or spectrograms as NN input
					 y - [numpy array] depth maps as NN output
					 sets_name - [string] name of file where new sets are to be found 
	@RETURN: none 
	'''
	dims = list(x.shape)
	dims[0] = -1
	dims = tuple(dims)

	if not os.path.isfile(sets_name):
		print "splitting dataset into training and test sets..."
		xshape = x.shape[0]
		xbool = np.zeros_like(x, dtype=bool)
		ybool = np.zeros_like(y, dtype=bool)
		subset_size = int(xshape/7)
		half_test_size = int((xshape*0.2)/14)
	
		for i in range(7):
			print "creating bool array for subset",i
			start_index = subset_size*i
			end_index = start_index + subset_size
			mid_index = int((0.5*(end_index-start_index)) + start_index) 
			test_start = mid_index-half_test_size
			test_end = mid_index+half_test_size 
			xbool[test_start:test_end] = True
			ybool[test_start:test_end] = True 

		print "reshaping test vectors..."
		xtest = x[xbool].reshape(dims)
		ytest = y[ybool].reshape((-1,192))
		print "reshaping training vectors..."
		xtrain = x[np.logical_not(xbool)].reshape(dims)
		ytrain = y[np.logical_not(ybool)].reshape((-1,192))
		print "x train before:", xtrain.shape
		print "y train before:", ytrain.shape

		print "shuffling for training set..."
		combined = zip(xtrain, ytrain)
		shuffle(combined)
		xtrain, ytrain = zip(*combined)
		xtrain = np.asarray(xtrain)
		ytrain = np.asarray(ytrain)
		print "x train after:", xtrain.shape
		print "y train after:", ytrain.shape

		print "saving data sets..."
		with h5py.File(sets_name, 'w') as sets:
			sets.create_dataset('xtrain', data=xtrain)
			sets.create_dataset('ytrain', data=ytrain)
			sets.create_dataset('xtest', data=xtest)
			sets.create_dataset('ytest', data=ytest)
		print "training and test sets saved as '%s'" %sets_name
															
#####################################################

def align_audio(threshold, audio):
	'''
	@PURPOSE: removes starting values of audio data that fall below threshold, 
						pads data's end with zeros to account for shape difference
	@PARAMS: threshold - [int] value at which audio data bits should be removed
					 audio - [numpy array] array of audio data
	@RETURN: [numpy array] newly aligned audio data
	'''
	result_array = np.zeros(audio.shape)
	for row in range(audio.shape[0]):
		above_threshold = np.abs(audio[row,:]) > threshold
		if np.any(above_threshold):
			threshold_index = np.where(above_threshold)[0][0]
			end_index = audio.shape[1] - threshold_index
			result_array[row, 0:end_index] = audio[row,threshold_index:]
		else:
			result_array[row,:] = audio[row,:]
	'''for row in range(0, result_array.shape[0]):
		plt.subplot(2,1,1)
		plt.plot(audio[row,:])
		plt.subplot(2,1,2)
		plt.plot(result_array[row,:]) 
		plt.show()'''
	return result_array

#####################################################

main()


