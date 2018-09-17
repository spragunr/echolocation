import matplotlib.pyplot as plt
import os
import os.path
import argparse
import h5py
import numpy as np
import numpy.ma as ma
import scipy.signal
import sys

import tensorflow as tf

import keras
import keras.utils
from keras.models import load_model
from keras.backend import floatx
import keras.backend as K
from keras.layers import MaxPooling2D
from keras.utils import CustomObjectScope

from util import raw_generator, safe_mse, safe_berhu, safe_l1, DataGenerator
from util import l1_ssim, ssim, ssim_multi, berhu_ssim
from network import chirp_shift

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
    
    parser.add_argument('--predict-closest', dest='predict_closest',
                        help=('learn to predict the 3d position of the ' +
                              'closest point'),
                        default=False, action='store_true')

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
        images = sets['test_rgb'][:]
        audio = sets['test_da'][:] / 32000.
    h5_audio = 'test_da'
    if args.predict_closest:
        h5_depth = 'test_closest'
    else:
        h5_depth = 'test_depths'

    print "loading model..."
    model = load_model(model_file,
                       custom_objects={'safe_mse':safe_mse,
                                       'safe_berhu':safe_berhu, 'keras':keras,
                                       "safe_l1":safe_l1, "l1_ssim":l1_ssim,
                                       "ssim":ssim, "ssim_multi":ssim_multi,
                                       "berhu_ssim":berhu_ssim, "chirp_shift":chirp_shift})

    print "LOADED", model.inputs
    model.summary()

    #show_kernel_weights(model)
    batch_size = 100
    gen = DataGenerator(args.data, h5_audio, h5_depth,
                        batch_size=batch_size,
                        shift=args.random_shift,
                        no_shift=True, noise=0.0, #shuffle=True,
                        tone_noise=0.0)

    # gen = raw_generator(x_test,xcr_test, y_test, noise=0,
    #                     shift=args.random_shift, no_shift=True,
    #                     batch_size=x_test.shape[0], shuffle=False,
    #                     tone_noise=0)

    audio_shape = gen[0][0][0].shape[1::]
    xcr_shape = gen[0][0][2].shape[1::]
    y_shape = gen[0][1].shape[1::]
    audio_left = np.empty([images.shape[0]] + list(audio_shape), dtype='float32')
    audio_right = np.empty([images.shape[0]] + list(audio_shape), dtype='float32')
    xcr = np.empty([images.shape[0]] + list(xcr_shape), dtype='float32')
    y_test = np.empty([images.shape[0]] + list(y_shape), dtype='float32')

    # import network
    # print audio_shape
    # model = network.build_model(gen[0][0][0].shape, (1,1,1), 0)
    # model.save('tmp_model.h5')
    # model = load_model('tmp_model.h5',
    #                    custom_objects={'safe_mse':safe_mse,
    #                                    'safe_berhu':safe_berhu,
    #                                    'keras':keras,
    #                                    "safe_l1":safe_l1,
    #                                    "l1_ssim":l1_ssim, "ssim":ssim,
    #                                    "ssim_multi":ssim_multi,
    #                                    "berhu_ssim":berhu_ssim,
    #                                    "chirp_shift":chirp_shift,"tf":tf})
    # model.compile(optimizer='sgd', loss=safe_berhu)
    #print "BUILT", model.inputs
    #model.summary()

    enqueuer = keras.utils.OrderedEnqueuer(gen,
                                           use_multiprocessing=True)
    enqueuer.start(workers=12)
    batch_num = 0
    for x, y in enqueuer.get():
        frm = batch_num * batch_size
        to = (batch_num + 1) * batch_size
        audio_left[frm:to, ...] = x[0]
        audio_right[frm:to, ...] = x[1]
        xcr[frm:to, ...] = x[2]
        y_test[frm:to, ...] = y
        if to == images.shape[0]: break
        batch_num += 1
        sys.stdout.write( " " * 6 + str(batch_num) + "\r")
        sys.stdout.flush()
    enqueuer.stop()
        
    predictions = model.predict([audio_left, audio_right])
    #predictions = model.predict(xcr)
    #loss = model.evaluate(x_test, y_test)
    #print "\nTEST LOSS:", loss
    plot_1d_convolutions(model)
    
    loss = model.evaluate([audio_left, audio_right], y_test)
    #loss = model.evaluate(xcr, y_test)
    print "\nTEST LOSS:", loss

    if args.predict_closest:
        # indices = y_test[:,2] < 1.2
        # y_test = y_test[indices,:]
        # predictions = predictions[indices, :]
        distances = np.sqrt(np.sum((y_test - predictions) ** 2, axis=-1))
        plt.hist(distances, bins='auto',normed=1, histtype='step', cumulative=1)
        
        plt.show()
        view_closest(y_test, predictions, images)
    else:
        analyze_error(predictions, audio, y_test, False)
        view_depth_maps(audio, y_test, predictions, images)
        
    show_activations(model, [audio_left, audio_right], y_test)




def analyze_error(predictions, x_test, y_test, show_good=True):

    mask = y_test == 0
    predictions = ma.masked_array(predictions, mask=mask)
    y_test = ma.masked_array(y_test, mask=mask)
    
    # predictions based on overall ground truth mean
    mean = ma.mean(y_test, axis=0)
    plt.imshow(mean, interpolation='none')
    plt.show()
    means = np.tile(mean, (y_test.shape[0],1,1))
    means = ma.masked_array(means, mask=mask)

    # predictions based on per-image ground truth mean
    mean = ma.mean(y_test, axis=(1,2))
    mean = mean.reshape(mean.shape[0], 1,1)
    image_means = np.tile(mean,(1, y_test.shape[1],y_test.shape[2]))
    image_means = ma.masked_array(image_means, mask=mask)

    
    print "\nL2 Model Predictions"
    losses = calc_losses(predictions, y_test)
    print "\nMean Data set Predictions"
    calc_losses(means, y_test)
    print "\nPer-image mean predictions"
    mean_losses = calc_losses(image_means, y_test)

    if show_good:
        # Look at the worst results...
        min_depth = .5#300
        max_depth = 9.#7000#1800

        sort_indices = np.argsort(losses[5]/mean_losses[5])
        for i in sort_indices:
            print losses[5][i], mean_losses[5][i]
            plt.subplot(131)
            plt.plot(x_test[i,...])
            plt.subplot(132)
            plt.imshow(y_test[i,...],
                       interpolation='none',clim=(min_depth, max_depth))
            plt.subplot(133)
            predictions[i, y_test[i,...]==0] = 0
            plt.imshow(predictions[i,...],
                       interpolation='none',clim=(min_depth, max_depth))
            plt.show()
    

def calc_losses(predictions, y_test):

    #Threshold
    print "Error threshold"

    predictions = predictions.reshape(predictions.shape[0], -1)

    y_test = y_test.reshape(y_test.shape[0], -1)
    
    counts = y_test.count(axis=1)
    
    # There really shouldn't be negative predictions, but they don't
    # make sense when calculating this error value...
    no_neg = ma.masked_array(predictions, mask=predictions.mask)
    no_neg[predictions<0] = 0
    
    deltas = ma.maximum(no_neg / y_test, y_test / no_neg)
    
    d1 = ma.sum(deltas < 1.25, dtype='float', axis=1) / counts
    d2 = ma.sum(deltas < 1.25**2, dtype='float', axis=1) / counts
    d3 = ma.sum(deltas < 1.25**3, dtype='float', axis=1) / counts

    print np.mean(d1), np.mean(d2), np.mean(d3)

    abs_error = ma.sum(ma.abs(predictions - y_test)/ y_test, axis=1) / counts
    print("abs_error: {:.3f}".format(ma.mean(abs_error)))

    sqr_error = ma.sum((predictions - y_test)**2/ y_test, axis=1) / counts
    print("sqr relative difference: {:.3f}".format(np.mean(sqr_error)))

    rmse = ma.sqrt(ma.sum((predictions - y_test)**2, axis=1) / counts)
    print("RMSE (linear): {:.3f}".format(ma.mean(rmse)))

    
    rmse_log = ma.sqrt(np.sum((ma.log(predictions) -
                               ma.log(y_test))**2, axis=1) / counts)
    print("RMSE (log): {:.3f}".format(ma.mean(rmse_log)))

    print "MIN", np.min(predictions)
    print "MAX", np.max(predictions)
    print "BELOW 0",np.sum((predictions<0))
    print "ABOVE 10",np.sum((predictions>10))

    
    return d1, d2, d3, abs_error, sqr_error, rmse, rmse_log
    

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
    inp = net.input             # input placeholder

    channel_input = net.input[0]

    channel_outputs = [layer.get_output_at(0) for layer in net.layers[2].layers]
    print channel_outputs
    
    functor = K.function([net.layers[2].get_input_at(0)]+ [K.learning_phase()],
                         channel_outputs)

    left_outs = functor([x_test[0][0:100,...]]+ [1.])
    right_outs = functor([x_test[1][0:100,...]]+ [1.])
    for output in channel_outputs:
        print output
    #print left_outs
    #print right_outs

    for i in range(64):
        plt.subplot(2,1,1)
        plt.imshow(left_outs[4][i,:,:,0], interpolation='none')
        plt.subplot(2,1,2)
        plt.imshow(right_outs[4][i,:,:,0], interpolation='none')
        plt.figure()

        plt.subplot(1,2,1)
        plt.imshow(left_outs[5][i,:,:,0], interpolation='none')
        plt.subplot(1,2,2)
        plt.imshow(right_outs[5][i,:,:,0], interpolation='none')
        plt.figure()

        plt.subplot(1,2,1)
        plt.imshow(left_outs[6][i,:,:,0], interpolation='none')
        plt.subplot(1,2,2)
        plt.imshow(right_outs[6][i,:,:,0], interpolation='none')

        # plt.figure()
        # for j in range(64):
        #     plt.subplot(8,8,j+1)
        #     plt.imshow(left_outs[5][i,:,:,j], interpolation='none')
            
            
        plt.show()
    
    

def view_depth_maps(xtest, ytrue, ypred, images):
    np.random.seed(10)
    ypred = np.reshape(ypred, (ytrue.shape[0], 30,40))
    all_error = ypred-ytrue
    avg_error = np.mean(all_error)
    stdev = np.std(all_error)
    rng = (avg_error-(3*stdev),avg_error+(3*stdev))
    for i in range(0, ytrue.shape[0], 50):
        print 
        for j in range(10):
            index = -1

            index = np.random.randint(ytrue.shape[0])
            print index  
            true = ytrue[index, ...]
            pred = ypred[index, ...]
            pred[true==0] = 0

            error = pred - true


            #min_depth = np.min(true[true != 0])
            #max_depth = np.max(true)
            #min_depth = np.log(300)
            #max_depth = np.log(10000)
            min_depth = .5#300
            max_depth = 7.#7000#1800

            ax0 = plt.subplot(10,5,j*5 + 1)
            audio_plot = plt.plot(xtest[index,...])
            ax0.set_title("Audio")
            
            axi = plt.subplot(10,5,j*5 + 2)
            true_map = plt.imshow(images[index,...],   interpolation='none')
            axi.set_title("Audio")

            ax1 = plt.subplot(10,5,j*5 + 3)
            true_map = plt.imshow(true, clim=(min_depth, max_depth),
                                  interpolation='none')
            ax1.set_title("True Depth")

            ax2 = plt.subplot(10,5,j*5 + 4)
            pred_map = plt.imshow(pred, clim=(min_depth, max_depth),
                                  interpolation='none')
            ax2.set_title("Predicted Depth")

            ax3 = plt.subplot(10,5,j*5 + 5)
            error_map = plt.imshow(error, clim=rng, cmap="Greys",
                                   interpolation='none')
            ax3.set_title("Squared Error Map")
        plt.show()


#####################################################

def view_closest(y_test, predictions, images):
    from mpl_toolkits.mplot3d import Axes3D

    for i in range(0, y_test.shape[0], 10):
        fig = plt.figure()
        fig.add_subplot(121)
        plt.imshow(images[i, ...], interpolation='none')
        ax = fig.add_subplot(122, projection='3d')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(.3, 2)
        ax.set_zlim(-1, 1)
        ax.scatter(y_test[i, 0], y_test[i, 2], -y_test[i, 1],
                   marker='s')
        ax.scatter(predictions[i, 0], predictions[i, 2],
                   -predictions[i, 1], marker='+')
        print np.sqrt(np.sum((predictions[i,:] - y_test[i,:])**2))*100, "cm"
        plt.show()
#####################################################


if __name__ == "__main__":
    main()
