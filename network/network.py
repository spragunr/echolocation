import os
import os.path
import argparse
import json

import matplotlib.pyplot as plt
import h5py
import numpy as np
import scipy

import tensorflow as tf

import keras
import keras.layers as layers
from keras import regularizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv1D, Conv2D, Dense, MaxPooling2D,UpSampling2D, Input, Lambda
from keras.layers.core import Flatten, Reshape, Permute
from keras.models import load_model, Sequential
from keras.models import Model

from util import validation_split, raw_generator, DataGenerator
from util import safe_mse, safe_berhu, safe_l1, l1_ssim, berhu_ssim

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
    parser.add_argument('--epochs', type=int, default=60,
                        help="how many epochs to train")
    parser.add_argument('--lr', type=float, default=.02,
                        help="initial learning rate")
    parser.add_argument('--lr-reduce-every', type=int, default=50,
                        help="how often to halve the learning rate")

    parser.add_argument('--loss',  default='berhu',
                        help="loss function. One of l2, l1, berhu")

    parser.add_argument('--random-shift', type=float, default=.02,
                        help="fraction random shift for augmentation")
    parser.add_argument('--white-noise', type=float, default=.05,
                        help="multiplicative noise added for augmentation")
    parser.add_argument('--tone-noise', type=float, default=0,
                        help="additive sin wave noise added for augmentation")
    parser.add_argument('--no-flip', dest='no_flip',
                        help=('augment training data by flipping channels'),
                        default=False, action='store_true')

    parser.add_argument('--validation', type=float, default=.1,
                        help="proportion of training set to use for validation")
    parser.add_argument('--predict-closest', dest='predict_closest',
                        help=('learn to predict the 3d position of the ' +
                              'closest point'),
                        default=False, action='store_true')

    args = parser.parse_args()

    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    if os.path.exists(args.dir):
        print "output folder already exists."
    else:
        os.makedirs(args.dir)

    with open(args.dir+'/args.json', 'w') as f:
        json.dump(vars(args), f, sort_keys=True,
                  indent=4, separators=(',', ': '))

    print "loading data..."
    #sets = h5py.File(args.data, 'r')
    h5_audio_train = 'train_da'
    h5_audio_val = 'val_da'
        
    if args.predict_closest:
        h5_depth_train = 'train_closest'
        h5_depth_val = 'val_closest'
    else:
        h5_depth_train = 'train_depths'
        h5_depth_val = 'val_depths'
        
    print "building and training model..."
    if args.loss == 'l2':
        loss_function = safe_mse
    elif args.loss == 'l1':
        loss_function = safe_l1
    elif args.loss == 'berhu':
        loss_function = safe_berhu

    L2 = 0.0#.00001
    batch_size = 64
    if args.validation > 0:
        val_gen = DataGenerator(args.data, h5_audio_val, h5_depth_val,
                                batch_size=batch_size,
                                shift=args.random_shift,
                                no_shift=True, noise=0.0, shuffle=True,
                                tone_noise=0.0)
        
    train_gen = DataGenerator(args.data, h5_audio_train, h5_depth_train,
                              batch_size=batch_size,
                              shift=args.random_shift,
                              noise=args.white_noise, shuffle=True,
                              tone_noise=args.tone_noise, flip=(not
                                                              args.no_flip))
    

    x_sample, _ = train_gen[0]
    input_shape = x_sample[0].shape
    xcr_shape = x_sample[2].shape
    print input_shape

    net = build_model(input_shape, xcr_shape, L2, args.predict_closest)
    #net = build_xcr_model(input_shape, xcr_shape, L2,
    #                      args.predict_closest)
    #net = build_both_model(input_shape, xcr_shape, L2, args.predict_closest)

    #adam = keras.optimizers.Adam(lr=args.lr, beta_1=0.9, beta_2=0.999,
    #                             epsilon=1e-08, decay=0.0)
    sgd = keras.optimizers.SGD(lr=args.lr, momentum=0.9)
    net.compile(optimizer=sgd, loss=loss_function)
    net.summary()

    # Configure callbacks:
    filepath = args.dir + '/model.{epoch:02d}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss',
                                 verbose=0,
                                 save_best_only=False, save_weights_only=False,
                                 mode='auto', period=5)
    #lr_reducer = LRReducer(args.lr_reduce_every)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                   patience=5, min_lr=0.001)
    tensorboard = keras.callbacks.TensorBoard(log_dir=args.dir,
                                              histogram_freq=0,
                                              write_graph=True,
                                              write_images=False)

    csv_logger = keras.callbacks.CSVLogger(args.dir+'/log.csv')

    callbacks_list = [checkpoint, lr_reducer, csv_logger, tensorboard]

    # Perform the training:
    if args.validation > 0:
        hist = net.fit_generator(train_gen, use_multiprocessing=True,
                                 workers=12, max_queue_size=10,
                                 steps_per_epoch=len(train_gen),
                                 epochs=args.epochs,
                                 callbacks=callbacks_list,
                                 validation_data=val_gen)
    else:
        hist = net.fit_generator(train_gen, use_multiprocessing=True,
                                 workers=12, max_queue_size=10,
                                 steps_per_epoch=len(train_gen),
                                 epochs=args.epochs,
                                 callbacks=callbacks_list)
    return net


def crazy_model(input_shape, L2, predict_closest=False):
    # First build a model to handle a single channel...
    input_audio = Input(shape=input_shape[1::])
    input_audio_left = Input(shape=input_shape[1::])
    input_audio_right = Input(shape=input_shape[1::])

    # BANK OF 1D CONVOLUTIONS SHARED BY ALL PATHS
    conv = Conv1D(125, (256),
                  strides=(1), use_bias=True,
                  activation='relu',
                  input_shape=input_shape[1::])
    
    x = conv(input_audio)
    x = Permute((2,1))(x)
    newshape = (conv.output_shape[2], conv.output_shape[1], 1)
    x = Reshape(newshape)(x)

    conv1d_model = Model(input_audio, x)
    set_conv_1d_weights_chirp(conv)
    
    #CREATE A PATH THAT SIMULTANEOUSLY PROCESSES BOTH CHANNELS
    left_conv_out = conv1d_model(input_audio_left)
    right_conv_out = conv1d_model(input_audio_right)
    
    merged = keras.layers.concatenate([left_conv_out, right_conv_out], axis=3)
    x = Conv2D(64, (5,21), strides=(5,2), activation='relu', use_bias=True,
                   kernel_regularizer=regularizers.l2(L2))(merged)
    x = MaxPooling2D(pool_size=(1, 16), strides=None,
                     padding='valid')(x)
    x = Conv2D(64, (5,5), strides=(1,2), activation='relu', use_bias=True,
               kernel_regularizer=regularizers.l2(L2))(x)
    both_out = Conv2D(32, (3,3), strides=(1,1), activation='relu',
                      use_bias=True,
                      kernel_regularizer=regularizers.l2(L2))(x)

    # Create per-channel paths
    x = conv1d_model(input_audio)
    x = MaxPooling2D(pool_size=(1, 16), strides=None,
                     padding='valid')(x)
    x = Conv2D(64, (5,5), strides=(5,2), activation='relu', use_bias=True,
                   kernel_regularizer=regularizers.l2(L2))(x)
    x = Conv2D(64, (5,5), strides=(1,2), activation='relu', use_bias=True,
                   kernel_regularizer=regularizers.l2(L2))(x)
    out = Conv2D(32, (3,3), strides=(1,1), activation='relu', use_bias=True,
                     kernel_regularizer=regularizers.l2(L2))(x)
    channel_model = Model(input_audio, out)

    left_out = channel_model(input_audio_left)
    right_out = channel_model(input_audio_right)

    channel_model.summary()
    
    merged = keras.layers.concatenate([left_out, right_out,both_out],
                                      axis=-1)
    x = Conv2D(64, (3,3), strides=(1,1),
               activation='relu',padding='same',
               kernel_regularizer=regularizers.l2(L2))(merged)
    x = Conv2D(64, (3,3), strides=(1,1),
               activation='relu',padding='same',
               kernel_regularizer=regularizers.l2(L2))(x) 
    

    x = Flatten()(x)
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



def chirp_shift(convs, conv_length, num_conv=125):
    full_chirp_length = int(.02 * 44100)
    max_offset = full_chirp_length
    output_list = []
    for i in range(num_conv):
        offset = int(i *  max_offset / float(num_conv))
        output_list += [tf.expand_dims(convs[:, i, offset:offset-max_offset, 0], -1)]

    return tf.stack(output_list, axis=1)
        

def build_channel_prep(input_shape, L2):
    conv_1d_length = 256
    num_1d_conv = 125
    input_audio = Input(shape=input_shape[1::])

    conv = Conv1D(num_1d_conv, (conv_1d_length),
                  strides=(1), use_bias=True,  padding='same',

                  activation='linear',
                  input_shape=input_shape[1::])
    
    x = conv(input_audio)
    x = Permute((2,1))(x)
    newshape = (conv.output_shape[2], conv.output_shape[1], 1)
    x = Reshape(newshape)(x)
    
    # x = Lambda(lambda x: tf.abs(x))(x)
    # print x.shape
    #x = chirp_shift(x, conv_1d_length, num_1d_conv)
    #x = Lambda(lambda x : chirp_shift(x, conv_1d_length, num_1d_conv))(x)
    print x.shape
    
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
    return channel_model

def build_xcr_prep(xcr_shape, L2):
    model_in = Input(shape=xcr_shape[1::])
    x = Conv2D(64, (5,5), strides=(1,1), activation='relu',
               use_bias=True, padding='valid',
               kernel_regularizer=regularizers.l2(L2))(model_in)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu',
               use_bias=True, padding='same',
               kernel_regularizer=regularizers.l2(L2))(x)
    x = Conv2D(16, (3,3), strides=(1,1), activation='relu',
                 use_bias=True, padding='same',
                 kernel_regularizer=regularizers.l2(L2))(x)

    x = Flatten()(x)
    x = Dense(600, activation='relu',use_bias=True,
              kernel_regularizer=regularizers.l2(L2))(x)
    xcr_model = Model(inputs=model_in, outputs=x)
    xcr_model.summary()
    return xcr_model

def build_decoder(input_shape, L2,predict_closest=False):
    decoder_in = Input(shape=input_shape)

    x = Dense(600, activation='relu',use_bias=True,
              kernel_regularizer=regularizers.l2(L2))(decoder_in)
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
    
    decoder = Model(inputs=decoder_in, outputs=x)
    return decoder

def build_both_model(input_shape, xcr_shape, L2, predict_closest=False):
    # First build a model to handle a single channel...
    channel_model = build_channel_prep(input_shape, L2)
    channel_model.summary()
    
    input_audio_left = Input(shape=input_shape[1::])
    input_audio_right = Input(shape=input_shape[1::])
    xcr_in = Input(shape=xcr_shape[1::])

    left_out = channel_model(input_audio_left)
    right_out = channel_model(input_audio_right)
    
    lr_merged = keras.layers.concatenate([left_out, right_out], axis=2)
    x = Flatten()(lr_merged)
    vol_out = Dense(600, activation='relu',use_bias=True,
                    kernel_regularizer=regularizers.l2(L2))(x)

    xcr_prep = build_xcr_prep(xcr_shape, L2)
    xcr_out = xcr_prep(xcr_in)

    merged = keras.layers.concatenate([vol_out, xcr_out])


    decoder = build_decoder((1200,), L2,predict_closest)
    x = decoder(merged)
    
    net = Model(inputs=[input_audio_left, input_audio_right, xcr_in], outputs=x)
    return net


def build_xcr_model_simple(input_shape, xcr_shape, L2, predict_closest=False):
    xcr_in = Input(shape=xcr_shape[1::])
    input_audio_left = Input(shape=input_shape[1::])
    input_audio_right = Input(shape=input_shape[1::])
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu',
               use_bias=True, padding='same',
               kernel_regularizer=regularizers.l2(L2))(xcr_in)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu',
               use_bias=True, padding='same',
               kernel_regularizer=regularizers.l2(L2))(x)
    x = Conv2D(32, (3,3), strides=(1,1), activation='relu',
                 use_bias=True, padding='same',
                 kernel_regularizer=regularizers.l2(L2))(x)

    x = Flatten()(x)
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
    
    out = Model(inputs=[input_audio_left, input_audio_right, xcr_in],
                outputs=x)
    return out

def build_xcr_model(input_shape, xcr_shape, L2, predict_closest=False):
    # First build a model to handle a single channel...
    xcr_in = Input(shape=xcr_shape[1::])
    input_audio_left = Input(shape=input_shape[1::])
    input_audio_right = Input(shape=input_shape[1::])

    xcr_prep = build_xcr_prep(xcr_shape, L2)
    x = xcr_prep(xcr_in)

    decoder = build_decoder((600,), L2,predict_closest)
    x = decoder(x)

    net = Model(inputs=[input_audio_left, input_audio_right, xcr_in], outputs=x)
    return net



def build_model(input_shape, xcr_shape, L2, predict_closest=False):
    # First build a model to handle a single channel...
    channel_model = build_channel_prep(input_shape, L2)
    channel_model.summary()
    
    input_audio_left = Input(shape=input_shape[1::])
    input_audio_right = Input(shape=input_shape[1::])
    xcr_in = Input(shape=xcr_shape[1::])

    left_out = channel_model(input_audio_left)
    right_out = channel_model(input_audio_right)

    merged = keras.layers.concatenate([left_out, right_out], axis=2)

    x = Flatten()(merged)
    x = Dense(600, activation='relu',use_bias=True,
              kernel_regularizer=regularizers.l2(L2))(x)

    decoder = build_decoder((600,), L2,predict_closest)
    x = decoder(x)
    
    net = Model(inputs=[input_audio_left, input_audio_right, xcr_in], outputs=x)
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
    #envelope = scipy.stats.norm.pdf(np.linspace(-2.5, 2.5, conv_length))
    #envelope = np.blackman(conv_length)
    for i in range(kernel_shape[2]):
        start_freq = ((16000 + freq_diff /2.0) -
                      i * ((8000.0)/kernel_shape[2]))
        end_freq = start_freq - freq_diff
        weights[:,0,i] = chirp(t, start_freq, time_diff, end_freq) * amplitude# * envelope


    bias_weights = np.zeros(bias_shape)
    conv.set_weights([weights] + [bias_weights])



#####################################################

if __name__ == "__main__":
    main()
