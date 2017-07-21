"""
Program handles preprocessing of neural network input data.

written by Nhung Hoang, May-June 2017
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path

from random import shuffle
from scipy import signal
from sys import argv, exit

#####################################################
#####################################################

def main():
    '''files to change as necessary: > train_files - > test_files -

    '''
    
    train_files = ['ball_train1.h5', 'ball_train2.h5',
                   'ball_train3.h5', 'ball_train4.h5',
                   'ball_train5.h5']
    
    test_files = ['ball_test1.h5', 'ball_test2.h5', 'ball_test3.h5']

    if len(argv) != 2:
        print "usage: preprocessing.py collective_data_name"
        return

    sets_file = argv[1] + '_sets.h5' #contains the components of the
                                     #training and test sets

    if os.path.isfile(sets_file):
        print "preprocessed training and test sets already exist under file '%s'" % sets_file
        return

    path = os.getcwd()+'/ball_data/'

    train_size =  total_size(train_files, path)
    test_size =  total_size(test_files, path)

    sets = h5py.File(sets_file, 'w')

    print "Processing raw audio..."
    preprocess_set(train_files, path, sets, train_size, 'audio',
                   'train_da', shape_digital_audio)
    preprocess_set(test_files, path, sets, test_size, 'audio',
                   'test_da', shape_digital_audio)


    print "Processing spectrograms..."
    preprocess_set(train_files, path, sets, train_size, 'audio',
                   'train_specs', shape_spectrograms)
    preprocess_set(test_files, path, sets, test_size, 'audio',
                   'test_specs', shape_spectrograms)

    print "Processing depth..."
    preprocess_set(train_files, path, sets, train_size, 'depth',
                   'train_depths', downsize)
    preprocess_set(test_files, path, sets, test_size, 'depth',
                   'test_depths', downsize)

    
    sets.close()
    return



######################################################
######################################################

def preprocess_set(data_files, path, sets, total_size, data_name,
                   set_name, process_func):
    index = 0
    for f in data_files:
        print f
        with h5py.File(path + f, 'r') as d:
            data = process_func(d[data_name])
            index = append_to_set(sets, data, index,
                                  total_size, set_name)

######################################################
######################################################


def append_to_set(sets, data, index, total_size, set_name):
    '''
    @PURPOSE: Append preprocessed data to the appropriate set the
              data set will be created if it does not exist.
    @PARAMS: sets - [opened h5 file] data sets
             data - [numpy array] data to append
             index - [int] index to append
             total_size - [int] size of entire data set
             set_name - [string] name of set to append to
    @RETURN: [int] Next index for an append
    '''
    if not set_name in sets:
        set_shape = tuple([total_size] + list(data.shape[1:]))
        sets.create_dataset(set_name, set_shape)
    sets[set_name][index:index+data.shape[0],...] = data
    return index + data.shape[0]


######################################################
######################################################

def total_size(files, path='./'):
    '''
    @PURPOSE: determine the total number of data points in a set of h5 files
    @PARAMS: files - [list of strings] file names
    @RETURN: [int] number of points
    '''
    total = 0
    for name in files:
        with h5py.File(path+name, 'r') as d:
            total += d['audio'].shape[0]
    return total

######################################################
######################################################

def downsize(depth_maps, method='min', factor=40):
    '''
    @PURPOSE: downsizes a set of images
    @PARAMS: img - [numpy array] image to downsize
             method - [string] 'mean' or 'min'
             factor - [int] factor to downsize image by (default is 40)
    @RETURN: [numpy array] downsized images
    '''
    orig_dims = depth_maps.shape
    ds_dims = (orig_dims[0], orig_dims[1]/factor, orig_dims[2]/factor)
    downsized_map = np.zeros(ds_dims)
    for ind in range(depth_maps.shape[0]):
        for i in range(0, orig_dims[1], factor):
            for j in range(0, orig_dims[2], factor):
                window = depth_maps[ind, i:i+factor, j:j+factor].flatten()
                non_zero = np.delete(window, np.where(window==0))
                if non_zero.size != 0:
                    if method == 'mean':
                        downsized_map[ind, i/factor,j/factor] = np.mean(non_zero)
                    elif method == 'min':
                        downsized_map[ind, i/factor,j/factor] = np.min(non_zero)
                    else:
                        print "UNRECOGNIZED DOWNSIZE METHOD"
    return downsized_map

######################################################

def shape_digital_audio(audio):
    train_AS = audio.shape
    audio_input = np.empty((train_AS[0],train_AS[1]*2,1))
    counter = 0
    for j in audio:
        combined = np.concatenate((j[:,0],j[:,1]))
        combined = np.reshape(combined,(train_AS[1]*2,1))
        audio_input[counter, ...] = combined
        counter += 1
    return audio_input

######################################################

def shape_spectrograms(audio):
    min_freq=7000
    max_freq=17000

    train_AS = audio.shape
    freq1, time1, spectro1 = signal.spectrogram(audio[0,:,0],
                                                noverlap=230, fs=44100)
    freq2, time2, spectro2 = signal.spectrogram(audio[0,:,1],
                                                noverlap=230, fs=44100)
    top_crop = np.where(freq1 > min_freq)[0][0]
    bot_crop = np.where(freq1 > max_freq)[0][0]
    crop1 = spectro1[top_crop:bot_crop,:]
    crop2 = spectro2[top_crop:bot_crop,:]
    combined = np.concatenate((crop1,crop2),axis=1)
    dims = combined.shape
    spec_input = np.empty((train_AS[0], dims[0], dims[1], 1))
    combined = np.reshape(combined, (dims[0], dims[1], 1))
    spec_input[0,:,:,:] = combined
    for i in range(1,train_AS[0]):
        freq1, time1, spectro1 = signal.spectrogram(audio[i,:,0],
                                                    noverlap=230)
        freq2, time2, spectro2 = signal.spectrogram(audio[i,:,1],
                                                    noverlap=230)
        crop1 = spectro1[top_crop:bot_crop,:]
        crop2 = spectro2[top_crop:bot_crop,:]
        combined = np.concatenate((crop1,crop2),axis=1)
        combined = np.reshape(combined, (dims[0], dims[1], 1))
        spec_input[i,:,:,:] = combined

    return spec_input


######################################################
######################################################

if __name__ == "__main__":
    main()
