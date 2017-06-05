import h5py
import matplotlib.pyplot as plt
import numpy as np
import os 
import os.path
import tensorflow as tf

from keras.backend import floatx
from keras.layers import Conv2D, Dense
from keras.layers.core import Flatten
from keras.models import load_model, Sequential
from scipy import io, signal

######################################################

def build_and_train_model(x_train, y_train):
	net = Sequential()
	net.add(Conv2D(32, (5,5), 
			strides=(1,1), 
			activation='relu',
			data_format='channels_last',
			input_shape=x_train.shape[1:]))
	net.add(Flatten())
	net.add(Dense(100, activation='relu'))
	net.add(Dense(192, activation='linear'))
	net.compile(optimizer='adam', loss=adjusted_mse)
	print "finished compiling"
	net.fit(x_train, y_train, validation_split=0.2, epochs=1, batch_size=32)
	net.save('stereo_model.h5')
	print "model saved as 'stereo_model.h5'"
	return load_model('stereo_model.h5', custom_objects={'adjusted_mse':adjusted_mse})

######################################################

def run_model(net, x_test, y_test):
#	loss = net.evaluate(x_test, y_test)
#	scale_loss = np.exp(loss)
	predictions = net.predict(x_test)
#	plot_data(np.exp(y_test), np.exp(predictions))
	loss = adjusted_mse(y_test, predictions)
	return loss #scale_loss

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

def main():
	if not os.path.isfile('stereo_model.h5'):
		print "building model..."
		path = os.getcwd()
		with h5py.File(path+'/model_sets.h5', 'r') as sets:
			x_train = np.log(sets['xtrain'][:])
			y_train = sets['ytrain'][:]
			x_test = np.log(sets['xtest'][:])
			y_test = sets['ytest'][:]
		model = build_and_train_model(x_train, y_train)
	else: 
		print "loading model..."
		path = os.getcwd()
		with h5py.File(path+'/model_sets.h5', 'r') as sets:
			x_test = np.log(sets['xtest'][:])
			y_test = np.log(sets['ytest'][:])
		model = load_model('stereo_model.h5', custom_objects={'adjusted_mse':adjusted_mse})
#	loss = run_model(model, x_test, y_test)	

main()

