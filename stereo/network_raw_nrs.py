import h5py
import matplotlib.pyplot as plt
import numpy as np
import os 
import os.path
import tensorflow as tf

from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.backend import floatx
from keras.layers import Conv1D, Conv2D, Dense, MaxPooling2D
from keras.layers.core import Flatten, Reshape, Dropout, Lambda
from keras.models import load_model, Sequential
from keras import optimizers
from scipy import io, signal
from sys import argv, exit
from keras import backend as K

tf.logging.set_verbosity(tf.logging.WARN)
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

######################################################

def main():

	# files 
	model_file = 'model_100k3_raw.h5' 
	sets_file = '100k_data3_sets.h5'


        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        set_session(tf.Session(config=config))

	if not os.path.isfile(model_file):
		print "building model..."
		path = os.getcwd()+'/'
		with h5py.File(path+sets_file, 'r') as sets:
			x_train = sets['train_da'][:]/32000
			y_train = np.log(1+sets['train_depths'][:].reshape(-1, 192))

                        indices = np.random.permutation(x_train.shape[0])
                        np.take(x_train,indices,axis=0,out=x_train)
                        np.take(y_train,indices,axis=0,out=y_train)
                        
			x_test = sets['test_da'][:]/32000
			y_test = np.log(1+sets['test_depths'][:].reshape(-1, 192))
		model = build_and_train_model(x_train, y_train, model_file)

	else: 
		print "loading model..."
		path = os.getcwd()+'/'
		with h5py.File(path+sets_file, 'r') as sets:
			x_test = sets['test_da'][:]/32000
			y_test = np.log(1+sets['test_depths'][:].reshape(-1, 192))
		model = load_model(model_file, custom_objects={'adjusted_mse':adjusted_mse})
        model.summary()
	loss = run_model(model, x_test, y_test)	

######################################################
######################################################

def build_and_train_model(x_train, y_train, model_file):
	net = Sequential()
	net.add(Conv1D(128, (256),
		       strides=(1),
		       activation='relu',
		       input_shape=x_train.shape[1:]))
	conv_output_size = net.layers[0].compute_output_shape(x_train.shape)[1]				
	net.add(Reshape((conv_output_size,128,1)))
        #net.add(Lambda(lambda x: K.abs(x)))
        net.add(MaxPooling2D(pool_size=(16, 1), strides=None,
                                 padding='valid'))
        net.add(BatchNormalization())
	net.add(Conv2D(64, (5,5), strides=(2,5), activation='relu'))
        net.add(BatchNormalization())
	net.add(Conv2D(64, (5,5), strides=(2,1), activation='relu'))
        net.add(BatchNormalization())
	net.add(Conv2D(32, (3,3), strides=(1,1), activation='relu'))
	net.add(Flatten())
        net.add(BatchNormalization())
	net.add(Dense(600, activation='relu'))
        net.add(BatchNormalization())
	net.add(Dense(600, activation='relu'))
        net.add(BatchNormalization())
	net.add(Dense(600, activation='relu'))
        net.add(BatchNormalization())
	net.add(Dense(192, activation='linear'))

	net.compile(optimizer='adam', loss=adjusted_mse)
        net.summary()
	print "finished compiling"

        # checkpoint
	filepath= model_file[:-3] + '.{epoch:02d}.h5'
	checkpoint = ModelCheckpoint(filepath, monitor='loss',
	                             verbose=0,
	                             save_best_only=False,save_weights_only=False,
	                             mode='auto', period=10)
	callbacks_list=[checkpoint]
 
	hist = net.fit(x_train, y_train, validation_split=0.1,
	               epochs=100, batch_size=64, callbacks=callbacks_list)


	with h5py.File(model_file[:-3]+'_loss_history.h5', 'w') as lh:
                lh.create_dataset('val_losses', data=hist.history['val_loss'])
		lh.create_dataset('losses', data=hist.history['loss'])
		print "loss history saved as '"+model_file[:-3]+"_loss_history.h5'"
	net.save(model_file)
	print "model saved as '%s'" %model_file
	return net

######################################################

def run_model(net, x_test, y_test):
	predictions = net.predict(x_test, batch_size=64)
	loss = net.evaluate(x_test, y_test)
	print "\nTEST LOSS:", loss
	view_average_error(np.exp(y_test)-1,np.exp(predictions)-1)
	for i in range(100, 2000, 110):
		view_depth_maps(i, np.exp(y_test)-1, np.exp(predictions)-1)

#####################################################

def adjusted_mse(y_true, y_pred):
        zero = tf.constant(0, dtype=floatx())
        ok_entries = tf.not_equal(y_true, zero)
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
			true_map = plt.imshow(true, clim=(500, 5000), interpolation='none')
			ax1.set_title("True Depth")
			
			ax2 = plt.subplot(10,3,j*3 + 2)
			pred_map = plt.imshow(pred, clim=(500, 5000), interpolation='none')
			ax2.set_title("Predicted Depth")
			
			ax3 = plt.subplot(10,3,j*3 + 3)
			error_map = plt.imshow(error, clim=rng, cmap="Greys", interpolation='none')
			ax3.set_title("Squared Error Map")
		
		plt.show()

#####################################################

main()

