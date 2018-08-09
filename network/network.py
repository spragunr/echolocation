import os
import os.path
import argparse
import json

import matplotlib.pyplot as plt
import h5py
import numpy as np

import tensorflow as tf

import keras
import keras.layers as layers
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, Conv2D, Dense, MaxPooling2D,UpSampling2D, Input
from keras.layers.core import Flatten, Reshape, Permute
from keras.models import load_model, Sequential
from keras.models import Model

from util import validation_split_by_chunks, raw_generator
from util import safe_mse, safe_berhu, safe_l1

tf.logging.set_verbosity(tf.logging.WARN)
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

######################################################

TARGET_WIDTH = 40
TARGET_HEIGHT = 30
TARGET_SIZE = TARGET_HEIGHT * TARGET_WIDTH

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('data', help="preprocessed training and testing data")
    parser.add_argument('--dir', default='',
                        help="where to store models and results")
    parser.add_argument('--epochs', type=int, default=200,
                        help="how many epochs to train")
    parser.add_argument('--lr', type=float, default=.01,
                        help="initial learning rate")
    parser.add_argument('--lr-reduce-every', type=int, default=50,
                        help="how often to halve the learning rate")

    parser.add_argument('--loss',  default='berhu',
                        help="loss function. One of l2, l1, berhu")

    parser.add_argument('--random-shift', type=float, default=.02,
                        help="fraction random shift for augmentation")
    parser.add_argument('--white-noise', type=float, default=.05,
                        help="multiplicative noise added for augmentation")
    parser.add_argument('--validation', type=float, default=.1,
                        help="proportion of training set to use for validation")
    parser.add_argument('--tone-noise', type=float, default=0,
                        help="additive sin wave noise added for augmentation")

    parser.add_argument('--predict-closest', dest='predict_closest',
                        help=('learn to predict the 3d position of the ' +
                              'closest point'),
                        default=False, action='store_true')

    args = parser.parse_args()

    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))


    print "TRAINING NEW MODEL"
    if os.path.exists(args.dir):
        print "output folder already exists."
    else:
        os.makedirs(args.dir)

    with open(args.dir+'/args.json', 'w') as f:
        json.dump(vars(args), f, sort_keys=True,
                  indent=4, separators=(',', ': '))

    print "loading data..."
    with h5py.File(args.data, 'r') as sets:
        x_train = sets['train_da'][:, 0:2646, :]/32000.0
        if args.predict_closest:
            y_train = sets['train_closest'][:]
        else:
            y_train = sets['train_depths'][:] / 1000.
        
    print "building and training model..."
    if args.loss == 'l2':
        loss_function = safe_mse
    elif args.loss == 'l1':
        loss_function = safe_l1
    elif args.loss == 'berhu':
        loss_function = safe_berhu
    model = build_and_train_model(x_train, y_train, args.dir, args.lr,
                                  args.lr_reduce_every, args.epochs,
                                  args.random_shift, args.white_noise,
                                  args.tone_noise,
                                  args.predict_closest,
                                  args.validation,
                                  loss_function)


######################################################

def build_and_train_model(x_train, y_train, model_folder, lr,
                          reduce_every, epochs, shift, white_noise,
                          tone_noise, predict_closest, validation,
                          loss_function):

    L2 = 0.0#.00001
    batch_size = 64
    if validation > 0:
        x_val, y_val, x_train, y_train = validation_split_by_chunks(x_train,
                                                                    y_train,
                                                                    validation)
        val_gen = raw_generator(x_val, y_val, batch_size=batch_size,
                                shift=shift, no_shift=True, noise=.00,
                                tone_noise=0)
        
    train_gen = raw_generator(x_train, y_train, batch_size=batch_size,
                              shift=shift, noise=white_noise,
                              shuffle=True, tone_noise=tone_noise,flip=True)
    

    x_sample, _ = train_gen.next()
    input_shape = x_sample[0].shape

    net = build_model(input_shape, L2, predict_closest)

    #adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999,
    #                             epsilon=1e-08, decay=0.0)
    sgd = keras.optimizers.SGD(lr=lr, momentum=0.9)
    net.compile(optimizer=sgd, loss=loss_function)
    net.summary()

    # Configure callbacks:
    filepath = model_folder + '/model.{epoch:02d}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss',
                                 verbose=0,
                                 save_best_only=False, save_weights_only=False,
                                 mode='auto', period=25)
    lr_reducer = LRReducer(reduce_every)

    tensorboard = keras.callbacks.TensorBoard(log_dir=model_folder,
                                              histogram_freq=0,
                                              write_graph=True,
                                              write_images=False)

    csv_logger = keras.callbacks.CSVLogger(model_folder+'/log.csv')

    callbacks_list = [checkpoint, lr_reducer, csv_logger, tensorboard]

    # Perform the training:
    if validation > 0:
        hist = net.fit_generator(train_gen,
                                 steps_per_epoch=x_train.shape[0]//batch_size,
                                 epochs=epochs, callbacks=callbacks_list,
                                 validation_data=val_gen,
                                 validation_steps=x_val.shape[0]//batch_size)
    else:
        hist = net.fit_generator(train_gen,
                                 steps_per_epoch=x_train.shape[0]//batch_size,
                                 epochs=epochs, callbacks=callbacks_list)
    return net




def build_model(input_shape, L2, predict_closest=False):
    # First build a model to handle a single channel...
    input_audio = Input(shape=input_shape[1::])

    conv = Conv1D(125, (256),
                  strides=(1), use_bias=True,
                  activation='relu',
                  input_shape=input_shape[1::])
    
    x = conv(input_audio)
    x = Permute((2,1))(x)
    newshape = (conv.output_shape[2], conv.output_shape[1], 1)
    x = Reshape(newshape)(x)
    

    x = MaxPooling2D(pool_size=(1, 16), strides=None,
                     padding='valid')(x)
    x = Conv2D(64, (5,5), strides=(5,2), activation='relu', use_bias=True,
                   kernel_regularizer=regularizers.l2(L2))(x)
    x = Conv2D(64, (5,5), strides=(1,2), activation='relu', use_bias=True,
                   kernel_regularizer=regularizers.l2(L2))(x)
    out = Conv2D(32, (3,3), strides=(1,1), activation='relu', use_bias=True,
                     kernel_regularizer=regularizers.l2(L2))(x)
    channel_model = Model(input_audio, out)
    set_conv_1d_weights_chirp(conv)
    input_audio_left = Input(shape=input_shape[1::])
    input_audio_right = Input(shape=input_shape[1::])

    left_out = channel_model(input_audio_left)
    right_out = channel_model(input_audio_right)

    channel_model.summary()
    
    merged = keras.layers.concatenate([left_out, right_out], axis=2)

    x = Flatten()(merged)
    x = Dense(600, activation='relu',use_bias=True,
              kernel_regularizer=regularizers.l2(L2))(x)
    x = Dense(600, activation='relu',use_bias=True,
              kernel_regularizer=regularizers.l2(L2))(x)
    x = Dense(600, activation='relu',use_bias=True,
              kernel_regularizer=regularizers.l2(L2))(x)

    if predict_closest:
        x = Dense(3, activation='linear')(x)
    else:
    
        x = Reshape((15, 20, 2))(x)
        x = UpSampling2D(size=2)(x) 
        x = Conv2D(32, (3,3), strides=(1,1),
                   activation='relu',padding='same',
                   kernel_regularizer=regularizers.l2(L2))(x)
        x = Conv2D(32, (3,3), strides=(1,1),
                   activation='relu',padding='same',
                   kernel_regularizer=regularizers.l2(L2))(x)        
        x = Conv2D(1, (3,3), strides=(1,1),
                   activation='linear',padding='same')(x)
        x = Reshape((30,40))(x)
    
    net = Model(inputs=[input_audio_left, input_audio_right], outputs=x)
    return net


######################################################

class LRReducer(keras.callbacks.LearningRateScheduler):

    def __init__(self, reduce_every, reduce_by=.5, verbose=0):
        super(LRReducer, self).__init__(self.schedule, verbose)
        self.reduce_every = reduce_every
        self.reduce_by = reduce_by


    def schedule(self, epoch, lr):
        if epoch != 0 and (epoch % self.reduce_every) == 0:
            return lr * self.reduce_by
        else:
            return lr

#####################################################

def set_conv_1d_weights_sin(conv):
    kernel_shape = conv.get_weights()[0].shape
    bias_shape= conv.get_weights()[1].shape
    weights = np.zeros(kernel_shape)
    t = np.linspace(0, kernel_shape[0] /44100., kernel_shape[0])
    for i in range(kernel_shape[2]):
        freq = i * (8000.0/kernel_shape[2]) + 8000
        tone = np.sin(2 * np.pi * freq * t)
        weights[:,0,i] = tone

    bias_weights = np.zeros(bias_shape)
    conv.set_weights([weights] + [bias_weights])

#####################################################
    
def set_conv_1d_weights_chirp(conv, amplitude=.05):
    from scipy.signal import chirp
    kernel_shape = conv.get_weights()[0].shape
    bias_shape= conv.get_weights()[1].shape
    conv_length = kernel_shape[0]
    weights = np.zeros(kernel_shape)
    t = np.linspace(0, conv_length /44100., conv_length)
    freq_diff = 2322.0 # diff between start and end of chirp
    time_diff = conv_length /44100.
    for i in range(kernel_shape[2]):
        start_freq = ((16000 + freq_diff /2.0) -
                      i * ((8000.0)/kernel_shape[2]))
        end_freq = start_freq - freq_diff
        weights[:,0,i] = chirp(t, start_freq, time_diff, end_freq) * amplitude


    bias_weights = np.zeros(bias_shape)
    conv.set_weights([weights] + [bias_weights])



#####################################################

if __name__ == "__main__":
    main()
