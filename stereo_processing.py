import h5py
import matplotlib.pyplot as plt
import numpy as np
import os 
import os.path
import tensorflow as tf

from random import shuffle
from keras.backend import floatx
from keras.layers import Conv2D, Dense
from keras.layers.core import Flatten
from keras.models import load_model, Sequential
from scipy import io, signal
from sys import argv

######################################################

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

######################################################

def get_data():
	path = os.getcwd() 
	print "loading 'forensics' data..."
	with np.load(path+'/data_stereo/forensics.npz') as d1:
		audio1 = d1['audio']
		depth1 = d1['depth']
	print "loading 'isat243' data..."
	with np.load(path+'/data_stereo/isat243.npz') as d2:
		audio2 = d2['audio']
		depth2 = d2['depth']
	print "loading 'isat246' data..."
	with np.load(path+'/data_stereo/isat246.npz') as d3:
		audio3 = d3['audio']
		depth3 = d3['depth']
	print "loading 'isat248' data..."
	with np.load(path+'/data_stereo/isat246.npz') as d4:
		audio4 = d4['audio']
		depth4 = d4['depth']
	print "loading 'office' data..."
	with np.load(path+'/data_stereo/office.npz') as d5:
		audio5 = d5['audio']
		depth5 = d5['depth']
	print "loading 'stairwell' data..."
	with np.load(path+'/data_stereo/stairwell.npz') as d6:
		audio6 = d6['audio']
		depth6 = d6['depth']
	print "loading 'spine' data..."
	with np.load(path+'/data_stereo/spine.npz') as d7:
		audio7 = d7['audio']
		depth7 = d7['depth']
	print "---------------------------------"
	print "data loading complete\n"

	print "concatenating audio data..."
	audio = np.concatenate((audio1,audio2,audio3,audio4,audio5,audio6,audio7))
	print "concatenating depth data...\n"
	depth = np.concatenate((depth1,depth2,depth3,depth4,depth5,depth6,depth7))
	new_depth = np.empty((depth.shape[0],12,16))
	print "downsizing depth data..."
	for i in range(depth.shape[0]):
		new_depth[i] = downsize(depth[i])
	print "saving concatenated audio and downsized depth data saved as 'all.h5'...\n"
	with h5py.File('all.h5', 'w') as hf:
		hf.create_dataset('audio', data=audio)
		hf.create_dataset('depth', data=new_depth)

######################################################

def preprocess_data():
	print "preprocessing data..."
	path = os.getcwd()
	with h5py.File(path+'/all.h5', 'r') as data:
		audio = data['audio'][:]	
		depth = data['depth'][:] # shape: 13274, 12, 16

	depth_reshaped = np.reshape(depth, (depth.shape[0],-1)) # shape: 13274, 192

	if os.path.isfile('input_spectrograms.h5'):
		print "fetching spectrogram array..."
		with h5py.File(path+'/input_spectrograms.h5', 'r') as sgrams:
			input_set = sgrams['spectrograms'][:]
	else: 
		print "creating spectrogram set 0" 
		freq1, time1, spectro1 = signal.spectrogram(audio[0,:,0], noverlap=250)
		freq2, time2, spectro2 = signal.spectrogram(audio[0,:,1], noverlap=250)
		dims = spectro1.shape
		input_set = np.empty((audio.shape[0], dims[0], dims[1], 2))
		input_set[0,:,:,0] = spectro1
		input_set[0,:,:,1] = spectro2
	
		for i in range(1,audio.shape[0]):
			print "creating spectrogram set", i
			freq1, time1, spectro1 = signal.spectrogram(audio[i,:,0], noverlap=250)
			freq2, time2, spectro2 = signal.spectrogram(audio[i,:,1], noverlap=250)
			input_set[i,:,:,0] = spectro1
			input_set[i,:,:,1] = spectro2
		print "saving array of spectrograms as 'input_spectrograms.h5'..."
		with h5py.File('input_spectrograms.h5', 'w') as sgrams:
			sgrams.create_dataset('spectrograms', data=input_set)
	
	return input_set, depth_reshaped

######################################################

def split_data(x, y):
	if os.path.isfile('model_sets.h5'):
		print "fetching training and test sets..."
		with h5py.File('model_sets.h5', 'r') as sets:
			xtrain = sets['xtrain'][:]
			ytrain = sets['ytrain'][:]
			xtest = sets['xtest'][:]
			ytest = sets['ytest'][:]			
	else:
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
		xtest = x[xbool].reshape((-1,129,385,2))
		ytest = y[ybool].reshape((-1,192))
		print "reshaping training vectors..."
		xtrain = x[np.logical_not(xbool)].reshape((-1,129,385,2))
		ytrain = y[np.logical_not(ybool)].reshape((-1,192))
		print "saving data sets..."
		with h5py.File('model_sets.h5', 'w') as sets:
			sets.create_dataset('xtrain', data=xtrain)
			sets.create_dataset('ytrain', data=ytrain)
			sets.create_dataset('xtest', data=xtest)
			sets.create_dataset('ytest', data=ytest)
		print "training and test sets saved as 'model_sets.h5'"
															
#####################################################

def main():
	while not os.path.isfile('all.h5'):
		get_data()
	input_set, target = preprocess_data()
	split_data(input_set, target)

main()

