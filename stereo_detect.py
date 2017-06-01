"""
stereo version
"""

import matplotlib.pyplot as plt
import numpy as np
import os 
import os.path
import tensorflow as tf

from random import shuffle
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
	d1 = np.load(path+'/data_stereo/forensics.npz')
	print "loading 'isat243' data..."
	d2 = np.load(path+'/data_stereo/isat243.npz')
	print "loading 'isat246' data..."
	d3 = np.load(path+'/data_stereo/isat246.npz')
	print "loading 'isat248' data..."
	d4 = np.load(path+'/data_stereo/isat248.npz')
	print "loading 'office' data..."
	d5 = np.load(path+'/data_stereo/office.npz')
	print "loading 'stairwell' data..."
	d6 = np.load(path+'/data_stereo/stairwell.npz')
	print "loading 'spine' data..."
	d7 = np.load(path+'/data_stereo/spine.npz')
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

######################################################

def preprocess_data():
	print "preprocessing data..."
	path = os.getcwd()
	data = np.load(path+'/all.npz')
	audio = data['audio']
	depth = data['depth'] # shape: 13274, 12, 16

	depth_reshaped = np.reshape(depth, (depth.shape[0],-1)) # shape: 13274, 192
	valid_bool = np.all(depth_reshaped, axis=1)
	valid_depth = depth_reshaped[valid_bool] # removes samples containing zero
	# valid_depth shape: 10291, 192 (22% loss)
#	print "\n\nVALID DEPTH SHAPE:",valid_depth.shape,"\n\n"
	
	new_audio = audio[valid_bool]
	print "creating spectrogram set 0" 
	freq1, time1, spectro1 = signal.spectrogram(new_audio[0,:,0], noverlap=250)
	freq2, time2, spectro2 = signal.spectrogram(new_audio[0,:,1], noverlap=250)
	dims = spectro1.shape
	input_set = np.empty((new_audio.shape[0], dims[0], dims[1], 2))
	input_set[0,:,:,0] = spectro1
	input_set[0,:,:,1] = spectro2
	
	for i in range(1,new_audio.shape[0]):
		print "creating spectrogram set", i
		freq1, time1, spectro1 = signal.spectrogram(new_audio[i,:,0], noverlap=250)
		freq2, time2, spectro2 = signal.spectrogram(new_audio[i,:,1], noverlap=250)
		input_set[i,:,:,0] = spectro1
		input_set[i,:,:,1] = spectro2
#		plt.pcolormesh(spectro2)
#	plt.show()
	
	return input_set, valid_depth

######################################################

def split_data(x, y):

	'''
	xshape = x.shape[0]
	xbool = np.zeros_like(x, dtype=bool)
	ybool = np.zeros_like(y, dtype=bool)
	A = int((0.8/14)*xshape)
	B = int((1.6/14)*xshape)
	C = int((12.8/14)*xshape)
	sss = subset_size = int(xshape/7)
	tss = test_set_size = int((0.2*xshape)/7)

	xbool[A:tss] = True 
	xbool[sss+A:sss+A+tss] = True
	xbool[2*sss+A:2*sss+A+tss] =True 
	xbool[3*sss+A:3*sss+A+tss] =True 
	xbool[4*sss+A:4*sss+A+tss] =True 
	xbool[5*sss+A:5*sss+A+tss] =True 
	xbool[6*sss+A:6*sss+A+tss] =True 

	ybool[A:tss] = True 
	ybool[sss+A:sss+A+tss] = True
	ybool[2*sss+A:2*sss+A+tss] =True 
	ybool[3*sss+A:3*sss+A+tss] =True 
	ybool[4*sss+A:4*sss+A+tss] =True 
	ybool[5*sss+A:5*sss+A+tss] =True 
	ybool[6*sss+A:6*sss+A+tss] =True 

	xtest = x[xbool]
	ytest = y[ybool]
	xtrain = x[np.logical_not(xbool)]
	ytrain = y[np.logical_not(ybool)]
	'''
  
	'''# Shuffle the data 
	combined = zip(input_set, target)
	shuffle(combined)
	input_set, target = zip(*combined)
	input_set = np.asarray(input_set)
	target = np.asarray(target)

	length = int(input_set.shape[0]*0.8)
	xtrain = input_set[:length]
	ytrain = target[:length]
	xtest = input_set[length:]
	ytest = target[length:]'''

	
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

	# ISSUE: these are all 1D arrays, can't reshape because unknown first dim
	# hardcoded reshapes for quick fix now 
	print "reshaping test vectors..."
	xtest = x[xbool].reshape((2058,129,385,2))
	ytest = y[ybool].reshape((2058,192))
	print "reshaping training vectors..."
	xtrain = x[np.logical_not(xbool)].reshape((8233,129,385,2))
	ytrain = y[np.logical_not(ybool)].reshape((8233,192))

	'''print "x_all", x.shape 
	print "y_all", y.shape
	print "x_train", xtrain.shape
	print "y_train", ytrain.shape
	print "x_test", xtest.shape
	print "y_test", ytest.shape'''

	return xtrain, ytrain, xtest, ytest

######################################################

def build_model(x_train, y_train):
	net = Sequential()
	INPUT_SHAPE = x_train.shape[1:]
	net.add(Conv2D(64, (5,5), 
			strides=(1,1), 
			activation='relu',
			data_format='channels_last',
			input_shape=INPUT_SHAPE))
	net.add(Flatten())
	net.add(Dense(450, activation='relu'))
	net.add(Dense(192, activation='linear'))
	net.compile(optimizer='adam', loss='mean_squared_error')
	net.fit(x_train, y_train, validation_split=0.2, epochs=50)
	net.save('stereo_model.h5')
	print "model saved as 'stereo_model.h5'"
	return load_model('stereo_model.h5')

######################################################

def run_model(net, x_test, y_test):
#	loss = net.evaluate(x_test, y_test)
#	scale_loss = np.exp(loss)
	predictions = net.predict(x_test)
#	plot_data(np.exp(y_test), np.exp(predictions))
	loss = mse_ignore_nan2(y_test, predictions)
	return loss #scale_loss

#####################################################

def mse_ignore_nan2(y_true, y_pred):
    ok_entries = tf.logical_not(tf.is_nan(y_true))
    safe_targets = tf.where(ok_entries, y_true, y_pred)
    sqr = tf.square(y_pred - safe_targets)
    zero_nans = tf.cast(ok_entries, K.floatx())
    num_ok = tf.reduce_sum(zero_nans, axis=-1) # count OK entries
    num_ok = tf.maximum(num_ok, tf.ones_like(num_ok)) # avoid divide by zero
    return tf.reduce_sum(sqr, axis=-1) / num_ok

#####################################################

def  main():
	while not os.path.isfile('all.npz'):
		get_data()
	input_set, target = preprocess_data()
	x_train, y_train, x_test, y_test = split_data(input_set, target)
	if not os.path.isfile('stereo_model.h5'):
		print "building model..."
		model = build_model(x_train, y_train)
	else: 
		model = load_model('stereo_model.h5')
#	loss = run_model(model, x_test, y_test)	

main()

