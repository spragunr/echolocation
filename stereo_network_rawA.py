import h5py
import matplotlib.pyplot as plt
import numpy as np
import os 
import os.path
import tensorflow as tf

from keras.backend import floatx
from keras.layers import Conv1D, Conv2D, Dense
from keras.layers.core import Flatten
from keras.models import load_model, Sequential
from scipy import io, signal

######################################################

def build_and_train_model(x_train, y_train):
	net = Sequential()
	net.add(Conv2D(64, (128,2), 
					activation='relu', 
					data_format='channels_last', 
					input_shape=x_train.shape[1:]))
	net.add(Conv2D(32, (64,1), activation='relu'))
	net.add(Flatten())
	net.add(Dense(200, activation='relu'))
	net.add(Dense(192, activation='linear'))
	net.compile(optimizer='adam', loss=adjusted_mse)
	print "finished compiling"
	net.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=32)
	net.save('stereo_model_rawA.h5')
	print "model saved as 'stereo_model_rawA.h5'"
	return load_model('stereo_model_rawA.h5', custom_objects={'adjusted_mse':adjusted_mse})

######################################################

def run_model(net, x_test, y_test):
#	loss = adjusted_mse(y_test, predictions)
	loss = net.evaluate(x_test, y_test)
	print "\nLOSS:", loss
	predictions = net.predict(x_test)
	for i in range(100,2000, 110):
		view_depth_maps(i, net, np.exp(y_test)-1,np.exp(predictions)-1)

#####################################################

def adjusted_mse(y_true, y_pred):
	ok_entries = np.all(y_true)
	ok_entries = tf.cast(ok_entries, bool)
	safe_targets = tf.where(ok_entries, y_true, y_pred)
	sqr = tf.square(y_pred - safe_targets)
	valid = tf.cast(ok_entries, floatx()) 
	num_ok = tf.reduce_sum(valid, axis=-1) # count OK entries
	num_ok = tf.maximum(num_ok, tf.ones_like(num_ok)) # avoid divide by zero
	return tf.reduce_sum(sqr, axis=-1) / num_ok

#####################################################

def view_depth_maps(index, model, ytrue, ypred):
	true = np.reshape(ytrue[index], (12,16))
	pred = np.reshape(ypred[index], (12,16))
	diff = np.absolute(pred - true)

	ax1 = plt.subplot(1,3,1)
	true_map = plt.imshow(true, interpolation='none')
	ax1.set_title("True Depth")
	
	ax2 = plt.subplot(1,3,2)
	pred_map = plt.imshow(pred, interpolation='none')
	ax2.set_title("Predicted Depth")
	
	ax3 = plt.subplot(1,3,3)
	diff_map = plt.imshow(diff, interpolation='none', cmap='gray')
	ax3.set_title("Difference") # goal is black
	
	plt.show()

#####################################################

def main():
	#if not os.path.isfile('stereo_model_rawA.h5'):
	if os.path.isfile('stereo_model_rawA.h5'):
		print "building model..."
		path = os.getcwd()
		with h5py.File(path+'/model_sets_rawA.h5', 'r') as sets:
			x_train = sets['xtrain'][:]
			y_train = np.log(1+sets['ytrain'][:])
			x_test = sets['xtest'][:]
			y_test = np.log(1+sets['ytest'][:])
		model = build_and_train_model(x_train, y_train)
	else: 
		print "loading model..."
		path = os.getcwd()
		with h5py.File(path+'/model_sets_rawA.h5', 'r') as sets:
			x_test = sets['xtest'][:]
			y_test = np.log(1+sets['ytest'][:])
		model = load_model('stereo_model_rawA.h5', custom_objects={'adjusted_mse':adjusted_mse})
	loss = run_model(model, x_train, y_train)	

main()

