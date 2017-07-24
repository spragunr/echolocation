import h5py
import matplotlib.pyplot as plt
import numpy as np
import os 
import os.path
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

from keras.backend import floatx
from keras.layers import Conv2D, Conv3D, Dense, Dropout
from keras.layers.core import Flatten
from keras.models import load_model, Sequential
from keras import optimizers
from scipy import io, signal
from sys import argv, exit

tf.logging.set_verbosity(tf.logging.WARN)
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

######################################################

def main():

	# file names to change as necessary
	model_file = 'model_100k_specA.h5'
	sets_file = '100k_data_sets.h5'
  #sets_file = 'sets_ball_specA.h5'

	if os.path.isfile(model_file):
		print "loading model..."
		path = os.getcwd()+'/'
		with h5py.File(path+sets_file, 'r') as sets:
			x_test = normalize(sets['test_specs'][:])
			y_test = np.log(1+sets['test_depths'][:].reshape(-1, 192))
		model = load_model(model_file, custom_objects={'adjusted_mse':adjusted_mse})
                model.summary()
	else:
		print "building model..."
		path = os.getcwd()+'/'
		with h5py.File(path+sets_file, 'r') as sets:	
			x_train = normalize(sets['train_specs'][:])
			y_train = np.log(1+sets['train_depths'][:].reshape(-1, 192))
			x_test = normalize(sets['test_specs'][:])
			y_test = np.log(1+sets['test_depths'][:].reshape(-1, 192))
		model = build_and_train_model(x_train, y_train, model_file)
                model.summary()

	loss = run_model(model, x_test, y_test)

######################################################
######################################################

def build_and_train_model(x_train, y_train, model_file):
	net = Sequential()
	net.add(Conv2D(128, (5,5), 
			strides=(1,1), 
			activation='relu',
			data_format='channels_last',
			input_shape=x_train.shape[1:]))
	net.add(Conv2D(128, (5,5), strides=(2,2), activation='relu'))
	net.add(Conv2D(32, (3,3), strides=(1,1), activation='relu'))
	net.add(Flatten())
	net.add(Dense(600, activation='relu'))
	net.add(Dense(600, activation='relu'))
	net.add(Dense(300, activation='relu'))	
	net.add(Dense(192, activation='linear'))
	net.compile(optimizer='adam', loss=adjusted_mse)
	print "finished compiling"

	# checkpoint
	filepath= model_file[:-3] + '.{epoch:02d}.h5'
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=False,save_weights_only=False, mode='auto', period=25)
	callbacks_list=[checkpoint]
 
	hist = net.fit(x_train, y_train, validation_split=0.0, epochs=200, batch_size=32, callbacks=callbacks_list)

	with h5py.File(model_file[:-3]+'_loss_history.h5', 'w') as lh:
		lh.create_dataset('losses', data=hist.history['loss'])
		print "loss history saved as '"+model_file[:-3]+"_loss_history.h5'"
	net.save(model_file)
	print "model saved as '%s'" %model_file
	return net

######################################################

def run_model(net, x_test, y_test):
	loss = net.evaluate(x_test, y_test)
	print "\nLOSS:", loss
	predictions = net.predict(x_test)
	view_average_error(np.exp(y_test)-1, np.exp(predictions)-1)
	for i in range(100, 2000, 110):
		view_depth_maps(100, net, np.exp(y_test)-1, np.exp(predictions)-1)

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
	rng = (avg_val-(3*stdev), avg_val+(3*stdev))
	error_map = plt.imshow(avg_error, clim=rng, cmap="Greys", interpolation='none')
	plt.title("Absolute Average Error")
	plt.show()

####################################################

def view_depth_maps(index, model, ytrue, ypred):
	all_error = ypred-ytrue
	avg_error = np.mean(all_error)
	stdev = np.std(all_error)
	rng = (avg_error-(3*stdev), avg_error+(3*stdev))
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

####################################################

def normalize(x_set):
	log = np.log(1+x_set)
	minX = np.min(log)
	maxX = np.max(log)
	diff = float(maxX-minX)
	norm = ((log-minX)/diff)-0.5
	return norm

####################################################

main()

####################################################
####################################################
'''x1 = np.reshape(x[index+800], (29,178)) 
true1 = np.reshape(ytrue[index+800], (12,16))
pred1 = np.reshape(ypred[index+800], (12,16))
x2 = np.reshape(x[index+600], (29,178)) 
true2 = np.reshape(ytrue[index+600], (12,16))
pred2 = np.reshape(ypred[index+600], (12,16))
x3 = np.reshape(x[index+1900], (29,178)) 
true3 = np.reshape(ytrue[index+1900], (12,16))
pred3 = np.reshape(ypred[index+1900], (12,16))

ax1 = plt.subplot(3,3,1)
true_map1 = plt.imshow(x1)
ax1.set_title("Spectrogram 1")

ax2 = plt.subplot(3,3,2)
true_map1 = plt.imshow(true1, clim=(500,4000), interpolation='none')
ax2.set_title("True Depth 1")

ax3 = plt.subplot(3,3,3)
true_map1 = plt.imshow(pred1, clim=(500,4000), interpolation='none')
ax3.set_title("Predicted Depth 1")

ax4 = plt.subplot(3,3,4)
pred_map1 = plt.imshow(x2)
ax4.set_title("Spectrogram 2")

ax5 = plt.subplot(3,3,5)
true_map2 = plt.imshow(true2, clim=(500,4000), interpolation='none')
ax5.set_title("True Depth 2")

ax6 = plt.subplot(3,3,6)
pred_map2 = plt.imshow(pred2, clim=(500,4000), interpolation='none')
ax6.set_title("Predicted Depth 2")

ax7 = plt.subplot(3,3,7)
true_map3 = plt.imshow(x3)
ax7.set_title("Spectrogram 3")

ax8 = plt.subplot(3,3,8)
pred_map3 = plt.imshow(true3, clim=(500,4000), interpolation='none')
ax8.set_title("True Depth 3")

ax9 = plt.subplot(3,3,9)
true_map1 = plt.imshow(pred3, clim=(500,4000), interpolation='none')
ax9.set_title("Predicted Depth 3")

plt.show()'''
