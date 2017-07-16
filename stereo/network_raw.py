import h5py
import matplotlib.pyplot as plt
import numpy as np
import os 
import os.path
import tensorflow as tf

from keras.backend import floatx
from keras.layers import Conv1D, Conv2D, Dense
from keras.layers.core import Flatten, Reshape
from keras.models import load_model, Sequential
from keras import optimizers
from scipy import io, signal
from sys import argv, exit

tf.logging.set_verbosity(tf.logging.WARN)
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

######################################################

def main():

	# files 
	model_file = 'model_ball_rawA.h5' 
	sets_file = 'ball_data_sets.h5'

	if not os.path.isfile(model_file):
		print "building model..."
		path = os.getcwd()+'/'
		with h5py.File(path+sets_file, 'r') as sets:
			x_train = sets['train_da'][:]/32000
			y_train = np.log(1+sets['train_da'][:])
			x_test = sets['test_da'][:]/32000
			y_test = np.log(1+sets['test_depths'][:])
		model = build_and_train_model(x_train, y_train, model_file)
	else: 
		print "loading model..."
		path = os.getcwd()+'/'
		with h5py.File(path+sets_file, 'r') as sets:
			x_test = sets['test_da'][:]/32000
			y_test = np.log(1+sets['test_depths'][:])
		model = load_model(model_file, custom_objects={'adjusted_mse':adjusted_mse})
	loss = run_model(model, x_test, y_test)	

######################################################
######################################################

def build_and_train_model(x_train, y_train, model_file):
	net = Sequential()
	net.add(Conv1D(64, (256),
					strides=(26),
					activation='relu', 
					input_shape=x_train.shape[1:]))
	conv_output_size = net.layers[0].compute_output_shape(x_train.shape)[1]				
	net.add(Reshape((conv_output_size,64,1)))
	net.add(Conv2D(128, (5,5), activation='relu'))
	net.add(Conv2D(128, (5,5), strides=(2,2), activation='relu'))
	net.add(Conv2D(64, (5,5), strides=(2,2), activation='relu'))
	net.add(Conv2D(32, (5,5), strides=(3,3), activation='relu'))
	net.add(Flatten())
	net.add(Dense(600, activation='relu'))
	net.add(Dense(1200, activation='relu'))
	net.add(Dense(600, activation='relu'))
	net.add(Dense(300, activation='relu'))
	net.add(Dense(192, activation='linear'))
	net.compile(optimizer='adam', loss=adjusted_mse)
	print "finished compiling"
	net.fit(x_train, y_train, validation_split=0.2, epochs=25, batch_size=32)
	net.save(model_file)
	print "model saved as '%s'" %model_file
	return net

######################################################

def run_model(net, x_test, y_test):
	predictions = net.predict(x_test)
	loss = net.evaluate(x_test, y_test)
	print "\nTEST LOSS:", loss
	view_average_error(np.exp(y_test)-1,np.exp(predictions)-1)
	for i in range(100, 2000, 110):
		view_depth_maps(i, np.exp(y_test)-1, np.exp(predictions)-1)

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

def view_average_error(ytrue, ypred):
	error = np.reshape(ypred-ytrue, (-1,12,16))
	avg_error = np.mean(error, axis=0)
	stdev = np.std(avg_error)
	avg_val = np.mean(avg_error)
	rng = (avg_val-(3*stdev),avg_val+(3*stdev))
	error_map = plt.imshow(avg_error, clim=rng, cmap="Greys", interpolation='none')
	plt.title("Absolute Average Error")
	plt.show()

#####################################################

def view_depth_maps(index, ytrue, ypred):
	all_error = ypred-ytrue
	avg_error = np.mean(all_error)
	stdev = np.std(all_error)
	rng = (avg_error-(3*stdev),avg_error+(3*stdev))
	for i in range(0, ytrue.shape[0], 50):
		for j in range(10): 
			index = i  + j
			true = np.reshape(ytrue[index], (12,16))
			pred = np.reshape(ypred[index], (12,16))
			error = pred - true

			ax1 = plt.subplot(10,3,j*3 + 1)
			true_map = plt.imshow(true, clim=(500, 2000), interpolation='none')
			ax1.set_title("True Depth")
			
			ax2 = plt.subplot(10,3,j*3 + 2)
			pred_map = plt.imshow(pred, clim=(500, 2000), interpolation='none')
			ax2.set_title("Predicted Depth")
			
			ax3 = plt.subplot(10,3,j*3 + 3)
			error_map = plt.imshow(error, clim=rng, cmap="Greys", interpolation='none')
			ax3.set_title("Squared Error Map")
		
		plt.show()

#####################################################

main()

