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
from keras.backend import floatx
from keras.layers import Conv1D, Conv2D, Dense, MaxPooling2D,UpSampling2D, Input
from keras.layers.core import Flatten, Reshape
from keras.models import load_model, Sequential
from keras.models import Model

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
    parser.add_argument('--lr', type=float, default=.001,
                        help="initial learning rate")
    parser.add_argument('--lr-reduce-every', type=int, default=50,
                        help="how often to halve the learning rate")
    parser.add_argument('--test-model',  default='',
                        help="model to test")

    args = parser.parse_args()

    sets_file = args.data
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    
    if args.dir != '':
        train_main(args)
    else:
        test_main(args)



def train_main(args):
    print "TRAINING NEW MODEL"
    if os.path.exists(args.dir):
        print "output folder already exists."
    else:
        os.makedirs(args.dir)

    with open(args.dir+'/args.json', 'w') as f:
        json.dump(vars(args), f, sort_keys=True,
                  indent=4, separators=(',', ': '))

    # Load testing data...
    with h5py.File(args.data, 'r') as sets:
        x_test = sets['test_da'][:,0:2646,:]/32000.
        y_test = np.log(1. + sets['test_depths'][:].reshape(-1, TARGET_SIZE))
        print y_test.shape

    print "loading data..."
    with h5py.File(args.data, 'r') as sets:
        x_train = sets['train_da'][:, 0:2646, :]/32000.
        y_train = np.log(1. + sets['train_depths'][:].reshape(-1, TARGET_SIZE))
        
    print "building and training model..."
    model = build_and_train_model(x_train, y_train,
                                  args.dir, args.lr,
                                  args.lr_reduce_every,
                                  args.epochs)
    run_model(model, x_test, y_test)


def test_main(args):
    # Load testing data...
    with h5py.File(args.data, 'r') as sets:
        x_test = sets['test_da'][:,0:2646,:]/32000.
        y_test = np.log(1. + sets['test_depths'][:].reshape(-1, TARGET_SIZE))

    print "loading model..."
    model = load_model(args.test_model,
                       custom_objects={'adjusted_mse':adjusted_mse})
    model.get_layer(name='model_1').summary()
    model.summary()
    plot_1d_convolutions(model)

    run_model(model, x_test, y_test)
    
######################################################
######################################################

def plot_1d_convolutions(model):
    layer = model.get_layer(name='model_1').get_layer('conv1d_1')
    print layer.get_weights()[1]
    W = layer.get_weights()[0]
    for i in range(120):
        plt.subplot(16, 8, i+1)
        plt.plot(W[:,0,i])
    
    plt.show()
    print W.shape


######################################################
######################################################

def raw_generator(x_train, y_train, batch_size=64, shift=.01,
                  no_shift=False, noise=.05, shuffle=True,
                  tone_noise=.25):
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
            
            # Concatenate the two channels...
            x_data[i, 0:result_length, 0] = x_train[batch_index + i,
                                                    start_ind[i]:start_ind[i] + result_length, 0]
            x_data[i, result_length::, 0] = x_train[batch_index + i,
                                                    start_ind[i]:start_ind[i] + result_length, 1]

        # Add a sin wave with random freqency and phase...
        if tone_noise > 0:
            t = np.linspace(0, result_length /44100., result_length * 2 )
            for i in range(batch_size):
                pitch = np.random.random() * 115.0 + 20 # midi pitch
                amp = np.random.random() * tone_noise
                #people.sju.edu/~rhall/SoundingNumber/pitch_and_frequency.pdf 
                freq = 440 * 2**((pitch - 69)/12.)
                phase = np.pi * 2 * np.random.random()
                tone = np.sin(2 * np.pi * freq * t + phase) * amp
                x_data[i, :, 0] += tone

            
        # Random multiplier for all samples...
        x_data *= (1. + np.random.randn(*x_data.shape) * noise)

        y_data = y_train[batch_index:batch_index + batch_size, ...]

        batch_index += batch_size

        if batch_index > (num_samples - batch_size):
            batch_index = 0

        yield x_data, y_data


def raw_generator_stereo(x_train, y_train, batch_size=64,
                         shift=.01,no_shift=False, noise=.00,
                         shuffle=True, tone_noise=.05):
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
            
        x_data_left = np.empty((batch_size, result_length, 1))
        x_data_right = np.empty((batch_size, result_length, 1))
        
        for i in range(batch_size):
            x_data_left[i, :, 0] = x_train[batch_index + i,
                                           start_ind[i]:start_ind[i] + result_length, 0]
            x_data_right[i, :, 0] = x_train[batch_index + i,
                                            start_ind[i]:start_ind[i] + result_length, 1]

        # Add a sin wave with random freqency and phase...
        if tone_noise > 0:
            t = np.linspace(0, result_length /44100., result_length)
            for i in range(batch_size):
                # max frequency about 8000hz:
                pitch = np.random.random() * 99 + 20 #* 115.0 + 20 # midi pitch
                amp = np.random.random() * tone_noise
                #people.sju.edu/~rhall/SoundingNumber/pitch_and_frequency.pdf 
                freq = 440 * 2**((pitch - 69)/12.)
                phase = np.pi * 2 * np.random.random()
                tone = np.sin(2 * np.pi * freq * t + phase) * amp
                x_data_left[i, :, 0] += tone
                x_data_right[i, :, 0] += tone

            
        # Random multiplier for all samples...
        x_data_left *= (1. + np.random.randn(*x_data_left.shape) * noise)
        x_data_right *= (1. + np.random.randn(*x_data_right.shape) * noise)

        y_data = y_train[batch_index:batch_index + batch_size, ...]

        batch_index += batch_size

        if batch_index > (num_samples - batch_size):
            batch_index = 0

        yield [x_data_left, x_data_right], y_data

        

######################################################
######################################################

def validation_split_by_chunks(x_train, y_train, split=.1, chunk_size=200):
    """Subsequent data points are very highly correlated.  This means that
    pulling out a validation set randomly from the training data will
    not give useful validation: it is likely that there will be a data
    point in the training set that is almost identical to the
    validation data.  This method splits out validation data in chunks
    to alleviate this issue.

    """
    val_size = int(split * x_train.shape[0])
    chunks = val_size // chunk_size
    val_size = chunk_size * chunks;
    train_size = x_train.shape[0] - val_size
    chunk_offest = chunks // x_train.shape[0]
    block_size = x_train.shape[0] // chunks
    train_chunk_size = block_size - chunk_size
    
    x_val = np.empty([val_size] + list(x_train.shape[1::]))
    y_val = np.empty([val_size] + list(y_train.shape[1::]))

    x_train_out = np.empty([train_size] + list(x_train.shape[1::]))
    y_train_out = np.empty([train_size] + list(y_train.shape[1::]))
    
    for i in range(chunks):
        # indices in the original data:
        block_start = i * block_size
        chunk_end = block_start + chunk_size

        # indices in the validation set:
        start = i * chunk_size
        end = start + chunk_size
        
        x_val[start:end, ...] = x_train[block_start:chunk_end, ...]
        y_val[start:end, ...] = y_train[block_start:chunk_end, ...]
                     
        # indices in the original data:
        train_start = chunk_end
        train_end = block_start + block_size

        # indices in the train set:
        start = i * train_chunk_size
        end = start + train_chunk_size
        
        x_train_out[start:end, ...] = x_train[train_start:train_end, ...]
        y_train_out[start:end, ...] = y_train[train_start:train_end, ...]

    # grab any partial final training data
    leftover =  x_train.shape[0] % chunks
    if leftover > 0:
        x_train_out[-leftover:, ...] = x_train[-leftover:, ...]
        y_train_out[-leftover:, ...] = y_train[-leftover:, ...]

    return x_val, y_val, x_train_out, y_train_out


def build_and_train_model(x_train, y_train, model_folder, lr,
                          reduce_every, epochs):

    L2 = 0#.00001
    validation_split = .1
    batch_size = 64

    # #Split validation data from the beginning of the training data...
    # end_val = int(validation_split * x_train.shape[0])
    # x_val = x_train[0:end_val, ...]
    # y_val = y_train[0:end_val, ...]

    # x_train = x_train[end_val::, ...]
    # y_train = y_train[end_val::, ...]
    x_val, y_val, x_train, y_train = validation_split_by_chunks(x_train, y_train)

    train_gen = raw_generator_stereo(x_train, y_train, batch_size=batch_size, shift=.02,
                                     noise=.05, shuffle=True, tone_noise=0)

    # Create validation set:
    val_gen = raw_generator_stereo(x_val, y_val,
                                   batch_size=batch_size, shift=0.02,
                                   no_shift=True, noise=.00,
                                   tone_noise=0)
    #x_val, y_val = val_gen.next()

    x_sample, _ = train_gen.next()
    input_shape = x_sample[0].shape
   
    net = build_model_shared_small(input_shape, L2)

    adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999,
                                 epsilon=1e-08, decay=0.0)
    net.compile(optimizer=adam, loss=adjusted_mse)
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
    hist = net.fit_generator(train_gen,
                             steps_per_epoch=x_train.shape[0]//batch_size,
                             epochs=epochs, callbacks=callbacks_list,
                             validation_data=val_gen,
                             validation_steps=x_val.shape[0]//batch_size)
    return net


    x = Flatten()(merged)
    x = Dense(600, activation='relu',use_bias=True,
              kernel_regularizer=regularizers.l2(L2))(x)
    x = Dense(600, activation='relu',use_bias=True,
              kernel_regularizer=regularizers.l2(L2))(x)
    net = Dense(1200, activation='linear')(x)
    net = Model(inputs=[input_audio_left, input_audio_right], outputs=net)
    return net



def build_model_shared_deep(input_shape, L2):
    # First build a model to handle a single channel...
    input_audio = Input(shape=input_shape[1::])
    
 
    conv = Conv1D(125, (512),
               strides=(1), use_bias=True,
               activation='relu',
                  input_shape=input_shape[1::])
    x = conv(input_audio)
    conv_output_size = conv.output_shape[1]
    x = Reshape((conv_output_size, 125, 1))(x)
    x = MaxPooling2D(pool_size=(16, 1), strides=None,
                     padding='valid')(x)
    x = Conv2D(64, (5,5), strides=(2,5), activation='relu', use_bias=True,
                   kernel_regularizer=regularizers.l2(L2))(x)
    x = Conv2D(64, (5,5), strides=(2,1), activation='relu', use_bias=True,
                   kernel_regularizer=regularizers.l2(L2))(x)
    out = Conv2D(32, (3,3), strides=(1,1), activation='relu', use_bias=True,
                     kernel_regularizer=regularizers.l2(L2))(x)
    channel_model = Model(input_audio, out)
    input_audio_left = Input(shape=input_shape[1::])
    input_audio_right = Input(shape=input_shape[1::])

    left_out = channel_model(input_audio_left)
    right_out = channel_model(input_audio_right)

    channel_model.summary()
    
    merged = keras.layers.concatenate([left_out, right_out], axis=-1)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu',padding='same',
                    kernel_regularizer=regularizers.l2(L2))(merged)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu',padding='same',
                    kernel_regularizer=regularizers.l2(L2))(x)

    x = Flatten()(x)
    x = Dense(600, activation='relu',use_bias=True,
              kernel_regularizer=regularizers.l2(L2))(x)
    x = Dense(600, activation='relu',use_bias=True,
              kernel_regularizer=regularizers.l2(L2))(x)
    x = Dense(2400, activation='linear')(x)
    
    x = Reshape((20, 15, 8))(x)
    x = UpSampling2D(size=2)(x) # 4x40x30

    x = Conv2D(32, (3,3), strides=(1,1), activation='relu',padding='same',
                    kernel_regularizer=regularizers.l2(L2))(x)
    x = Conv2D(32, (3,3), strides=(1,1), activation='relu',padding='same',
                    kernel_regularizer=regularizers.l2(L2))(x)
    
    x = Conv2D(1, (3,3), strides=(1,1), activation='linear',padding='same')(x)
    net = Flatten()(x)
    
    #net.add(Flatten())
    
    net = Model(inputs=[input_audio_left, input_audio_right], outputs=net)
    return net



def build_model_shared_small(input_shape, L2):
    # First build a model to handle a single channel...
    input_audio = Input(shape=input_shape[1::])
    
 
    conv = Conv1D(125, (256),
               strides=(1), use_bias=True,
               activation='relu',
                  input_shape=input_shape[1::])
    x = conv(input_audio)
    conv_output_size = conv.output_shape[1]
    x = Reshape((conv_output_size, 125, 1))(x)
    x = MaxPooling2D(pool_size=(16, 1), strides=None,
                     padding='valid')(x)
    x = Conv2D(64, (5,5), strides=(2,5), activation='relu', use_bias=True,
                   kernel_regularizer=regularizers.l2(L2))(x)
    x = Conv2D(64, (5,5), strides=(2,1), activation='relu', use_bias=True,
                   kernel_regularizer=regularizers.l2(L2))(x)
    out = Conv2D(32, (3,3), strides=(1,1), activation='relu', use_bias=True,
                     kernel_regularizer=regularizers.l2(L2))(x)
    channel_model = Model(input_audio, out)
    input_audio_left = Input(shape=input_shape[1::])
    input_audio_right = Input(shape=input_shape[1::])

    left_out = channel_model(input_audio_left)
    right_out = channel_model(input_audio_right)

    channel_model.summary()
    
    merged = keras.layers.concatenate([left_out, right_out], axis=-3)

    x = Flatten()(merged)
    x = Dense(600, activation='relu',use_bias=True,
              kernel_regularizer=regularizers.l2(L2))(x)
    x = Dense(600, activation='relu',use_bias=True,
              kernel_regularizer=regularizers.l2(L2))(x)
    x = Dense(600, activation='linear')(x)
    
    x = Reshape((20, 15, 2))(x)
    x = UpSampling2D(size=2)(x) # 4x40x30

    x = Conv2D(32, (3,3), strides=(1,1), activation='relu',padding='same',
                    kernel_regularizer=regularizers.l2(L2))(x)
    x = Conv2D(32, (3,3), strides=(1,1), activation='relu',padding='same',
                    kernel_regularizer=regularizers.l2(L2))(x)
    
    x = Conv2D(1, (3,3), strides=(1,1), activation='linear',padding='same')(x)
    net = Flatten()(x)
    
    #net.add(Flatten())
    
    net = Model(inputs=[input_audio_left, input_audio_right], outputs=net)
    return net



def build_model_shared(input_shape, L2):
    # First build a model to handle a single channel...
    input_audio = Input(shape=input_shape[1::])
    
 
    conv = Conv1D(120, (256),
               strides=(2), use_bias=False,
               activation='relu',
               input_shape=input_shape[1::],
               kernel_regularizer=regularizers.l2(L2))
    x = conv(input_audio)
    x = layers.BatchNormalization(momentum=.99)(x)
    conv_output_size = conv.output_shape[1]
    x = Reshape((conv_output_size, 120, 1))(x)
    x = MaxPooling2D(pool_size=(4, 1), strides=None,
                     padding='valid')(x)
    x = Conv2D(64, (5,5), strides=(2,5), activation='relu', use_bias=False,padding='same',
                   kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization(momentum=.99)(x)
    x = Conv2D(64, (5,5), strides=(2,1), activation='relu', use_bias=False,padding='same',
                   kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization(momentum=.99)(x)
    out = Conv2D(32, (3,3), strides=(1,1), activation='relu', use_bias=False,padding='same',
                     kernel_regularizer=regularizers.l2(L2))(x)
    out = layers.BatchNormalization(momentum=.99)(out)
    channel_model = Model(input_audio, out)
    input_audio_left = Input(shape=input_shape[1::])
    input_audio_right = Input(shape=input_shape[1::])

    left_out = channel_model(input_audio_left)
    right_out = channel_model(input_audio_right)
    
    merged = keras.layers.concatenate([left_out, right_out], axis=-1)

    x = Conv2D(32, (3,3), strides=(1,1), activation='relu', use_bias=False,padding='same',
           kernel_regularizer=regularizers.l2(L2))(merged)
    x = layers.BatchNormalization(momentum=.99)(x)
    x = Conv2D(32, (3,3), strides=(1,1), activation='relu', use_bias=False,padding='same',
           kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization(momentum=.99)(x)
    x = Flatten()(x)
    
    
    x = Dense(600, activation='relu',use_bias=False,
              kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization(momentum=.99)(x)
    x = Dense(600, activation='relu',use_bias=False,
              kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization(momentum=.99)(x)
    net = Dense(1200, activation='linear',
                  kernel_regularizer=regularizers.l2(L2))(x)
    net = Model(inputs=[input_audio_left, input_audio_right], outputs=net)
    return net


def build_model_new(input_shape, L2):
    inputs = Input(shape=input_shape[1::])
    conv = Conv1D(100, (256),
               strides=(2), use_bias=True,
               activation='elu',
               input_shape=input_shape[1::],
               kernel_regularizer=regularizers.l2(L2))
    x = conv(inputs)
    #x = layers.BatchNormalization(momentum=.99)(x)
    conv_output_size = conv.output_shape[1]
    x = Reshape((conv_output_size, 100, 1))(x)
    x = MaxPooling2D(pool_size=(8, 1), strides=None,
                     padding='valid')(x)

    
    
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=True,
               activation='elu',padding='same',
               kernel_regularizer=regularizers.l2(L2))(x)
    #x = layers.BatchNormalization(momentum=.99)(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), use_bias=True,
               activation='elu',padding='same',
               kernel_regularizer=regularizers.l2(L2))(x)
    #x = layers.BatchNormalization(momentum=.99)(x)

    for i in range(4):
        residual = x

        for j in range(3):
            x = layers.Activation('elu')(x)
            x = layers.SeparableConv2D(128, (3, 3),
                                       padding='same',
                                       use_bias=True)(x)
            #x = layers.BatchNormalization(momentum=.99)(x)
        x = layers.add([x, residual])

    x = layers.Activation('elu')(x)
    x = layers.SeparableConv2D(16, (3, 3),
                               padding='same',
                               use_bias=True)(x)
    #x = layers.BatchNormalization(momentum=.99)(x)
    x = Flatten()(x)
  
   
    x = Dense(600, activation='elu',use_bias=True,
              kernel_regularizer=regularizers.l2(L2))(x)
    #x = layers.BatchNormalization(momentum=.99)(x)
    x = Dense(600, activation='elu',use_bias=True,
              kernel_regularizer=regularizers.l2(L2))(x)
    #x = layers.BatchNormalization(momentum=.99)(x)
    net = Dense(1200, activation='linear',
                  kernel_regularizer=regularizers.l2(L2))(x)

    # net.add(Reshape((40, 30, 1)))
    # net.add(Conv2D(32, (3,3), strides=(1,1), activation='relu',padding='same',
    #                kernel_regularizer=regularizers.l2(L2)))
    # net.add(Conv2D(32, (3,3), strides=(1,1), activation='relu',padding='same',
    #                kernel_regularizer=regularizers.l2(L2)))
    # net.add(Conv2D(1, (3,3), strides=(1,1), activation='linear',padding='same',
    #                kernel_regularizer=regularizers.l2(L2)))

    # net.add(Flatten())
    net = Model(inputs=inputs, outputs=net)
    return net



def build_model(input_shape, L2):
    net = Sequential()
    
    # net.add(Conv1D(120, (512),
    #                strides=(1),
    #                activation='relu',
    #                input_shape=input_shape[1::],
    #                kernel_regularizer=regularizers.l2(L2)))
    # conv_output_size = net.layers[0].compute_output_shape(input_shape)[1]
    # net.add(Reshape((conv_output_size,120,1)))
    # net.add(MaxPooling2D(pool_size=(16, 1), strides=None,
    #                      padding='valid'))
    # net.add(Conv2D(64, (5,5), strides=(2,5), activation='relu',
    #                kernel_regularizer=regularizers.l2(L2)))
    # net.add(Conv2D(64, (5,5), strides=(2,1), activation='relu',
    #                kernel_regularizer=regularizers.l2(L2)))
    # net.add(Conv2D(32, (3,3), strides=(1,1), activation='relu',
    #                kernel_regularizer=regularizers.l2(L2)))
    # net.add(Flatten())
    
    net.add(Conv1D(100, (512),
                   strides=(1),
                   activation='relu',
                   input_shape=input_shape[1::],
                   kernel_regularizer=regularizers.l2(L2)))
    conv_output_size = net.layers[0].compute_output_shape(input_shape)[1]
    net.add(Reshape((conv_output_size, 100, 1)))
    net.add(MaxPooling2D(pool_size=(16, 1), strides=None,
                         padding='valid'))
    net.add(Conv2D(32, (3, 3), strides=(2, 2), activation='relu',padding='same',
                   kernel_regularizer=regularizers.l2(L2)))
    net.add(Conv2D(32, (3, 3), strides=(2, 2), activation='relu',padding='same',
                   kernel_regularizer=regularizers.l2(L2)))
    #net.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu',padding='same',
    #               kernel_regularizer=regularizers.l2(L2)))
    net.add(Conv2D(16, (1, 1), strides=(1, 1), activation='relu',padding='same',
                   kernel_regularizer=regularizers.l2(L2)))
    
    net.add(Flatten())
    net.add(Dense(600, activation='relu',
                  kernel_regularizer=regularizers.l2(L2)))
    net.add(Dense(600, activation='relu',
                  kernel_regularizer=regularizers.l2(L2)))
    net.add(Dense(1200, activation='relu',
                  kernel_regularizer=regularizers.l2(L2)))

    net.add(Reshape((40, 30, 1)))
    net.add(Conv2D(32, (3,3), strides=(1,1), activation='relu',padding='same',
                   kernel_regularizer=regularizers.l2(L2)))
    net.add(Conv2D(32, (3,3), strides=(1,1), activation='relu',padding='same',
                   kernel_regularizer=regularizers.l2(L2)))
    net.add(Conv2D(1, (3,3), strides=(1,1), activation='linear',padding='same',
                   kernel_regularizer=regularizers.l2(L2)))

    net.add(Flatten())
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


######################################################

def run_model(net, x_test, y_test):
    gen = raw_generator_stereo(x_test, y_test, noise=0, shift=0.02,no_shift=True,
                               batch_size=x_test.shape[0], shuffle=False, tone_noise=0)
    x_test = next(gen)[0]
    predictions = net.predict(x_test, batch_size=64)
    loss = net.evaluate(x_test, y_test)
    print "\nTEST LOSS:", loss
    view_average_error(np.exp(y_test)-1,np.exp(predictions)-1)
    for i in range(100, 2000, 110):
        view_depth_maps(i, x_test[0], np.exp(y_test)-1, np.exp(predictions)-1)

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
