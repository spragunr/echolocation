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

from util import raw_generator, safe_mse, safe_berhu, safe_l1


tf.logging.set_verbosity(tf.logging.WARN)
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

######################################################


TARGET_WIDTH = 40
TARGET_HEIGHT = 30
TARGET_SIZE = TARGET_HEIGHT * TARGET_WIDTH


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('data', help="preprocessed training and testing data")
    parser.add_argument('--random-shift', type=float, default=.02,
                        help="fraction random shift for augmentation")

    parser.add_argument('test-model',  default='',
                        help="model to test")


    args = parser.parse_args()
    print args

    model_file  = getattr( args, 'test-model' ) 
    sets_file = args.data

    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))


    print "loading test data..."
    with h5py.File(sets_file, 'r') as sets:
        x_test = sets['test_da'][:, 0:2646, :]/32000.
        #y_test = np.log(1. + sets['test_depths'][:].reshape(-1, TARGET_SIZE))
        y_test = sets['test_depths'][:] / 1000.0

    print "loading model..."
    model = load_model(model_file,
                       custom_objects={'safe_mse':safe_mse,
                                       'safe_berhu':safe_berhu, 'keras':keras,
                                       "safe_l1":safe_l1})


    model.summary()
    show_kernel_weights(model)


    gen = raw_generator(x_test, y_test, noise=0,
                        shift=args.random_shift, no_shift=True,
                        batch_size=x_test.shape[0], shuffle=False,
                        tone_noise=0)
    x_test_prepped = next(gen)[0]
    predictions = model.predict(x_test_prepped, batch_size=64)
    #loss = model.evaluate(x_test, y_test)
    #print "\nTEST LOSS:", loss
    calc_losses(predictions, y_test)
    plot_1d_convolutions(model)
    show_activations(model, x_test, y_test)


def calc_losses(predictions, y_test):

    #predictions = np.reshape(predictions, (y_test.shape[0],30,40))
    print y_test.shape
    ok_indices = y_test != 0

    # network predictions
    predictions = predictions[ok_indices]
    #predictions = (np.exp(predictions)-1) / 1000.0
    
    # predictions based on overall data-set mean
    y_test_nans = np.array(y_test)
    y_test_nans[y_test == 0] = np.nan
    mean = np.nanmean(y_test_nans, axis=0)
    means = np.tile(mean,(y_test.shape[0],1,1))
    means = means[ok_indices] 
    print means.shape

    # predictions based on per-image ground truth mean
    y_test_nans = np.array(y_test)
    y_test_nans[y_test == 0] = np.nan
    mean = np.nanmean(y_test_nans, axis=(1,2))
    print "A",mean.shape
    mean = mean.reshape(mean.shape[0], 1,1)
    image_means = np.tile(mean,(1, y_test.shape[1],y_test.shape[2]))
    print means.shape
    image_means = image_means[ok_indices]

    # target values
    y_test = y_test[ok_indices]
    
    print "\nL2 Model Predictions"
    calc_delta_losses(predictions, y_test)
    print "\nMean Data set Predictions"
    calc_delta_losses(means, y_test)
    print "\nPer-image mean predictions"
    calc_delta_losses(image_means, y_test)
   

    

def calc_delta_losses(predictions, y_test):

    #Threshold
    print "Error threshold"

    deltas = np.maximum(predictions / y_test, y_test / predictions)
    total = float(deltas.size)
    d1 = np.count_nonzero(deltas < 1.25) / total
    d2 = np.count_nonzero(deltas < 1.25**2) / total
    d3 = np.count_nonzero(deltas < 1.25**3) / total
    print d1, d2, d3


    abs_error = np.sum(np.abs(predictions - y_test)/ y_test) / total
    print("abs_error: {:.3f}".format(abs_error))

    sqr_error = np.sum((predictions - y_test)**2/ y_test) / total
    print("sqr relative difference: {:.3f}".format(sqr_error))

    rmse = np.sqrt(np.sum((predictions - y_test)**2) / total)
    print("RMSE (linear): {:.3f}".format(rmse))

    rmse_log = np.sqrt(np.sum((np.log(predictions) - np.log(y_test))**2) /
                       total)
    print("RMSE (log): {:.3f}".format(rmse_log))



    
######################################################
######################################################

def plot_1d_convolutions(model):

    layer = model.get_layer(name='model_1').get_layer('conv1d_1')
    W = layer.get_weights()[0]
    print W.shape
    num_convolutions = W.shape[2]
    fs = 41000
    peaks = np.empty(num_convolutions)

    # find peak frequency repsponses
    for i in range(num_convolutions):
        w, h = scipy.signal.freqz(W[:,0,i], 1.0)
        f = fs * w / (2*np.pi)
        peaks[i] = f[np.argmax(h)]
    plt.hist(peaks, 20)
    plt.show()
    
    for i in range(num_convolutions):
        plt.subplot(25, 5, i+1)
        plt.plot(W[:,0,i])
    plt.figure()
    for i in range(num_convolutions):
        plt.subplot(25, 5, i+1)
        w, h = scipy.signal.freqz(W[:,0,i], 1.0)
        f = fs * w / (2*np.pi)
        plt.plot(f, np.abs(h))
    
    plt.show()
    print W.shape


######################################################

def show_kernel_weights(net):
    first_2dconv = net.layers[2].layers[5]
    W =  first_2dconv.get_weights()[0]
    rows = cols =  int(np.ceil(np.sqrt(W.shape[3])))
    mn = np.min(W)
    mx = np.max(W)
    for i in range(W.shape[3]):
        
        plt.subplot(rows, cols, i + 1)
        plt.imshow(W[:,:,0,i], interpolation='none',clim=(mn, mx))


                    
    plt.show()
    

######################################################

def show_activations(net, x_test, y_test):
    gen = raw_generator(x_test, y_test, noise=0, no_shift=True,
                        batch_size=64, shuffle=True)
    x_test = next(gen)[0]
    print "X", type(x_test)
    predictions = net.predict(x_test, batch_size=64)
    print predictions

    inp = net.input             # input placeholder

    channel_input = net.input[0]

    channel_outputs = [layer.get_output_at(0) for layer in net.layers[2].layers]
    print channel_outputs
    
    functor = K.function([net.layers[2].get_input_at(0)]+ [K.learning_phase()],
                         channel_outputs)

    left_outs = functor([x_test[0]]+ [1.])
    right_outs = functor([x_test[1]]+ [1.])
    for output in channel_outputs:
        print output
    #print left_outs
    #print right_outs

    for i in range(64):
        plt.subplot(1,2,1)
        plt.imshow(left_outs[4][i,:,:,0], interpolation='none')
        plt.subplot(1,2,2)
        plt.imshow(right_outs[4][i,:,:,0], interpolation='none')
        plt.figure()
        for j in range(64):
            plt.subplot(8,8,j+1)
            plt.imshow(left_outs[5][i,:,:,j], interpolation='none')
            
            
        plt.show()
    
    return
    
    print "INPUT", type(net.input)
    # all layer outputs
    print net.layers[2].layers
    print net.layers
    for layer in net.layers:
        print layer
    
    outputs = [layer.get_output_at(0) for layer in net.layers]
    # evaluation function
    functor = K.function(inp+ [K.learning_phase()], outputs )
    print "FUNCTOR", functor
    # Testing
    print x_test[0].shape
    layer_outs = functor(x_test+ [1.])
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
