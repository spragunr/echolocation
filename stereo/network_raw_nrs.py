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
    model_file = '100k_raw_gen.h5'
    sets_file = 'prepped100k.h5'


    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=config))

    if not os.path.isfile(model_file):
        print "building model..."
        path = os.getcwd()+'/'
        with h5py.File(path+sets_file, 'r') as sets:
            x_train = sets['train_da'][:,0:2646,:]/32000.
            y_train = np.log(1. + sets['train_depths'][:].reshape(-1, 192))

            indices = np.random.permutation(x_train.shape[0])
            np.take(x_train,indices,axis=0,out=x_train)
            np.take(y_train,indices,axis=0,out=y_train)

            x_test = sets['test_da'][:,0:2646,:]/32000.
            y_test = np.log(1. + sets['test_depths'][:].reshape(-1, 192))
        model = build_and_train_model(x_train, y_train, model_file)

    else:
        print "loading model..."
        path = os.getcwd()+'/'
        with h5py.File(path+sets_file, 'r') as sets:
            x_test = sets['test_da'][:,0:2646,:]/32000.
            y_test = np.log(1. + sets['test_depths'][:].reshape(-1, 192))
        model = load_model(model_file, custom_objects={'adjusted_mse':adjusted_mse})
        plot_1d_convolutions(model)
    model.summary()
    loss = run_model(model, x_test, y_test)


    
######################################################
######################################################

def plot_1d_convolutions(model):

    print model.layers[0].get_weights()[1]
    W = model.layers[0].get_weights()[0]
    for i in range(128):
        plt.subplot(16, 8, i+1)
        plt.plot(W[:,0,i])
    
    plt.show()
    print W.shape


######################################################
######################################################

def raw_generator(x_train, y_train, batch_size=64, shift=.01,
                  noise=.05, no_shift=False, shuffle=True):
    num_samples = x_train.shape[0]
    sample_length = x_train.shape[1]
    result_length = int(sample_length * (1 - shift))
    batch_index = 0

    while True:
        # Shuffle before each epoch
        if batch_index == 0 and shuffle:
            indices = np.random.permutation(x_train.shape[0])
            np.take(x_train,indices,axis=0,out=x_train)
            np.take(y_train,indices,axis=0,out=y_train)

        # Randomly crop the audio data...
        if no_shift:
            start_ind = np.zeros(batch_size, dtype='int32')
        else:
            start_ind = np.random.randint(sample_length - result_length + 1,
                                          size=batch_size)
        x_data = np.empty((batch_size, result_length * 2, 1))
        for i in range(batch_size):
            x_data[i, 0:result_length, 0] = x_train[batch_index + i,
                                                    start_ind[i]:start_ind[i] + result_length, 0]
            x_data[i, result_length::, 0] = x_train[batch_index + i,
                                                    start_ind[i]:start_ind[i] + result_length, 1]

        # Random multiplier for all samples...
        x_data *= (1. + np.random.randn(*x_data.shape) * noise)

        y_data = y_train[batch_index:batch_index + batch_size, ...]

        batch_index += batch_size

        if batch_index > (num_samples - batch_size):
            batch_index = 0
                       
        yield x_data, y_data
    

    
######################################################
######################################################

def build_and_train_model(x_train, y_train, model_file):

    validation_split = .1
    batch_size = 64
    end_val = int(validation_split * x_train.shape[0])
    x_val = x_train[0:end_val,...]
    y_val = y_train[0:end_val,...]

    x_train = x_train[end_val::,...]
    y_train = y_train[end_val::,...]
    train_gen = raw_generator(x_train, y_train, batch_size=batch_size)
    val_gen = raw_generator(x_val, y_val, noise=0, no_shift=True)

    for x, y in train_gen:
        input_shape = x.shape
        break
    
    net = Sequential()
    net.add(Conv1D(128, (256),
                   strides=(1),
                   activation='relu',
                   input_shape=input_shape[1::]))
    conv_output_size = net.layers[0].compute_output_shape(input_shape)[1]
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
                                 mode='auto', period=25)
    callbacks_list=[checkpoint]

    hist = net.fit_generator(train_gen,
                             steps_per_epoch=x_train.shape[0]//batch_size,
                             epochs=200, callbacks=callbacks_list,
                             validation_data=val_gen,
                             validation_steps=x_val.shape[0]//batch_size)
#    hist = net.fit(x_train, y_train, validation_split=0.1,
#                   epochs=100, batch_size=64, callbacks=callbacks_list)


    with h5py.File(model_file[:-3]+'_loss_history.h5', 'w') as lh:
        lh.create_dataset('val_losses', data=hist.history['val_loss'])
        lh.create_dataset('losses', data=hist.history['loss'])
        print "loss history saved as '"+model_file[:-3]+"_loss_history.h5'"
    net.save(model_file)
    print "model saved as '%s'" %model_file
    return net

######################################################

def run_model(net, x_test, y_test):
    gen = raw_generator(x_test, y_test, noise=0, no_shift=True,
                        batch_size=x_test.shape[0], shuffle=False)
    x_test = next(gen)[0]
    predictions = net.predict(x_test, batch_size=64)
    loss = net.evaluate(x_test, y_test)
    print "\nTEST LOSS:", loss
    view_average_error(np.exp(y_test)-1,np.exp(predictions)-1)
    for i in range(100, 2000, 110):
        view_depth_maps(i, x_test, np.exp(y_test)-1, np.exp(predictions)-1)

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

def view_depth_maps(index, xtest, ytrue, ypred):
    all_error = ypred-ytrue
    avg_error = np.mean(all_error)
    stdev = np.std(all_error)
    rng = (avg_error-(3*stdev),avg_error+(3*stdev))
    for i in range(0, ytrue.shape[0], 50):
        print 
        for j in range(10):
            index = -1

            index = np.random.randint(ytrue.shape[0])
#            while index == -1:
#                index = np.random.randint(ytrue.shape[0])
#                if np.sum(ytrue[index] > 7000) + np.sum(ytrue[index] ==0) < 20:
#                    true = np.reshape(ytrue[index], (12,16))
#                    pred = np.reshape(ypred[index], (12,16))
#                else:
#                    index = -1

            print index  
            true = np.reshape(ytrue[index], (12,16))
            pred = np.reshape(ypred[index], (12,16))
            #true = np.log( 1 + np.reshape(ytrue[index], (12,16)))
            #pred = np.log(1 + np.reshape(ypred[index], (12,16)))
            error = pred - true


            #min_depth = np.min(true[true != 0])
            #max_depth = np.max(true)
            #min_depth = np.log(300)
            #max_depth = np.log(10000)
            min_depth = 300
            max_depth = 7000

            ax0 = plt.subplot(10,4,j*4 + 1)
            audio_plot = plt.plot(xtest[index,...])
            ax0.set_title("Audio")

            ax1 = plt.subplot(10,4,j*4 + 2)
            true_map = plt.imshow(true, clim=(min_depth, max_depth),
                                  interpolation='none')
            ax1.set_title("True Depth")

            ax2 = plt.subplot(10,4,j*4 + 3)
            pred_map = plt.imshow(pred, clim=(min_depth, max_depth),
                                  interpolation='none')
            ax2.set_title("Predicted Depth")

            ax3 = plt.subplot(10,4,j*4 + 4)
            error_map = plt.imshow(error, clim=rng, cmap="Greys",
                                   interpolation='none')
            ax3.set_title("Squared Error Map")

        plt.show()

#####################################################

if __name__ == "__main__":
    main()
