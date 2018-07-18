import matplotlib.pyplot as plt
import os
import os.path
import argparse
import h5py
import numpy as np
import scipy.signal

import tensorflow as tf

import keras
from keras.models import load_model
from keras.backend import floatx
import keras.backend as K
from keras.layers import MaxPooling2D
from keras.utils import CustomObjectScope




tf.logging.set_verbosity(tf.logging.WARN)
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

######################################################


TARGET_WIDTH = 40
TARGET_HEIGHT = 30
TARGET_SIZE = TARGET_HEIGHT * TARGET_WIDTH

class SignedMaxPooling2D(MaxPooling2D):

    # def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
    #              data_format=None, **kwargs):
    #     super(SignedMaxPooling2D, self).__init__(pool_size, strides, padding,
    #                                              data_format, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):

        x_neg = tf.scalar_mul(tf.constant(-1.0), inputs)
        x_neg_pool = K.pool2d(x_neg, pool_size, strides,
                              padding,pool_mode='max')
        x_pool = K.pool2d(inputs, pool_size, strides,
                              padding,pool_mode='max')
        gr = K.greater(x_neg_pool, x_pool)
        
        output = tf.where(gr,
                          tf.scalar_mul(tf.constant(-1.0), x_neg_pool),
                          x_pool)
        return output


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('data', help="preprocessed training and testing data")
    parser.add_argument('model_folder',
                        help="where to store models and results")

    args = parser.parse_args()

    model_file = args.model_folder + "/model.h5"
    sets_file = args.data

    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))


    print "loading test data..."
    with h5py.File(sets_file, 'r') as sets:
        x_test = sets['test_da'][:, 0:2646, :]/32000.
        y_test = np.log(1. + sets['test_depths'][:].reshape(-1, TARGET_SIZE))

    print "loading model..."
    model = load_model(model_file, custom_objects={'adjusted_mse':adjusted_mse,
                                                   'SignedMaxPooling2D':SignedMaxPooling2D})
    plot_1d_convolutions(model)
        
    model.summary()
    loss = run_model(model, x_test, y_test)


    
######################################################
######################################################

def plot_1d_convolutions(model):

    print model.layers[0].get_weights()[1]
    W = model.layers[0].get_weights()[0]
    fs = 41000
    peaks = np.empty(80)

    # find peak frequency repsponses
    for i in range(80):
        w, h = scipy.signal.freqz(W[:,0,i], 1.0)
        f = fs * w / (2*np.pi)
        peaks[i] = f[np.argmax(h)]
    plt.hist(peaks, 20)
    plt.show()


    
    for i in range(80):
        plt.subplot(16, 8, i+1)
        plt.plot(W[:,0,i])
    plt.figure()
    for i in range(80):
        plt.subplot(16, 8, i+1)
        w, h = scipy.signal.freqz(W[:,0,i], 1.0)
        f = fs * w / (2*np.pi)
        plt.plot(f, np.abs(h))
    
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

def run_model(net, x_test, y_test):
    # gen = raw_generator(x_test, y_test, noise=0, no_shift=True,
    #                     batch_size=x_test.shape[0], shuffle=False)
    gen = raw_generator(x_test, y_test, noise=0, no_shift=True,
                        batch_size=64, shuffle=True)
    x_test = next(gen)[0]
    predictions = net.predict(x_test, batch_size=64)

    inp = net.input             # input placeholder
    # all layer outputs
    outputs = [layer.output for layer in net.layers]
    # evaluation function
    functor = K.function([inp]+ [K.learning_phase()], outputs )

    # Testing
    layer_outs = functor([x_test, 1.])
    print layer_outs[2].shape
    for i in range(64):
        plt.imshow(layer_outs[2][i,:,:,0].T, interpolation='none')
        plt.figure()
        plt.hist(layer_outs[2][i,:,:,0].flatten())
        plt.figure()
        plt.imshow(layer_outs[3][i,:,:,0].T, interpolation='none')
        plt.figure()
        plt.subplot(121)
        plt.imshow(layer_outs[11][i,:,:,0].T, interpolation='none')
        plt.subplot(122)
        plt.imshow(layer_outs[11][i,:,:,1].T, interpolation='none')

        plt.figure()
        pred = np.reshape(layer_outs[14][i,...], (TARGET_HEIGHT,TARGET_WIDTH))
        plt.imshow(pred, interpolation='none')
        
        plt.show()
    return
    
    loss = net.evaluate(x_test, y_test)
    print "\nTEST LOSS:", loss
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
            true = np.reshape(ytrue[index], (TARGET_HEIGHT,TARGET_WIDTH))
            pred = np.reshape(ypred[index], (TARGET_HEIGHT,TARGET_WIDTH))
            #true = np.log( 1 + np.reshape(ytrue[index], (12,16)))
            #pred = np.log(1 + np.reshape(ypred[index], (12,16)))
            error = pred - true


            #min_depth = np.min(true[true != 0])
            #max_depth = np.max(true)
            #min_depth = np.log(300)
            #max_depth = np.log(10000)
            min_depth = 300
            max_depth = 7000#1800

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
