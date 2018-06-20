"""
Program handles preprocessing of neural network input data.

written by Nhung Hoang, May-June 2017
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import argparse

from random import shuffle
from scipy import signal
import scipy.misc
from sys import argv, exit

import depth_to_xyz
#####################################################
#####################################################

def main():
    '''files to change as necessary: > train_files - > test_files -

    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', nargs='*', help='training files',
                        required=True)
    parser.add_argument('--test', nargs='*', help='testing files',
                        required=True)

    parser.add_argument('--store-closest', dest='store_closest',
                        help='extract and store nearest point',
                        default=False, action='store_true')

    parser.add_argument('--camera-calib',
                        help='yaml calibration file for depth camera',
                        default='xtion.yaml')
    
    
    parser.add_argument('output_file')
    args = parser.parse_args()

    train_files = args.train

    test_files = args.test

    sets_file = args.output_file

    if os.path.isfile(sets_file):
        print "preprocessed training and test sets already exist under file '%s'" % sets_file
        return

    train_size =  total_size(train_files)
    test_size =  total_size(test_files)

    sets = h5py.File(sets_file, 'w')

    if args.store_closest:
        print "Storing closest points..."
        method = lambda depth_maps: depth_to_closest_point(depth_maps,
                                                           args.camera_calib)
        
        preprocess_set(train_files, sets, train_size, 'depth',
                       'train_closest', method, dtype=np.float32)
        preprocess_set(test_files, sets, test_size, 'depth',
                       'test_closest', method,dtype=np.float32)

    print "Processing images..."
    preprocess_set(train_files, sets, train_size, 'rgb',
                   'train_rgb', downsize_rgb)
    preprocess_set(test_files, sets, test_size, 'rgb',
                   'test_rgb', downsize_rgb)

    print "Processing raw audio..."
    preprocess_set(train_files, sets, train_size, 'audio_aligned',
                   'train_da', shape_digital_audio)
    preprocess_set(test_files, sets, test_size, 'audio_aligned',
                   'test_da', shape_digital_audio)

    

#    print "Processing spectrograms..."
#    preprocess_set(train_files, path, sets, train_size, 'audio_aligned',
#                   'train_specs', shape_spectrograms)
#    preprocess_set(test_files, path, sets, test_size, 'audio_aligned',
#                   'test_specs', shape_spectrograms)
 
    print "Processing depth..."
    preprocess_set(train_files, sets, train_size, 'depth',
                   'train_depths', downsize_depth)
    preprocess_set(test_files, sets, test_size, 'depth',
                   'test_depths', downsize_depth)



    sets.close()
    return



######################################################
######################################################

def preprocess_set(data_files, sets, total_size, data_name,
                   set_name, process_func, dtype=None):
    index = 0
    for f in data_files:
        print f
        with h5py.File(f, 'r') as d:
            data = process_func(d[data_name])
            index = append_to_set(sets, data, index,
                                  total_size, set_name, dtype)

######################################################
######################################################


def append_to_set(sets, data, index, total_size, set_name, dtype):
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
    if dtype == None:
        dtype = data.dtype
    if not set_name in sets:
        set_shape = tuple([total_size] + list(data.shape[1:]))
        sets.create_dataset(set_name, set_shape, dtype=dtype)
    sets[set_name][index:index+data.shape[0],...] = data
    return index + data.shape[0]


######################################################
######################################################

def total_size(files):
    '''
    @PURPOSE: determine the total number of data points in a set of h5 files
    @PARAMS: files - [list of strings] file names
    @RETURN: [int] number of points
    '''
    total = 0
    for name in files:
        with h5py.File(name, 'r') as d:
            total += d['audio'].shape[0]
    return total

######################################################
######################################################


def depth_to_closest_point(depth_maps, cam_file):
    cam_info = depth_to_xyz.parse_calibration_yaml(cam_file)
    points = np.zeros((depth_maps.shape[0], 3), dtype=np.float32)

    for i in range(depth_maps.shape[0]):
        pc = depth_to_xyz.depth_map_to_point_cloud(depth_maps[i, ...], cam_info)
        closest = pc[np.argmin(pc[:, 2]), ...]
        points[i, ...] = closest
        if i % 100 == 0:
            print i, "/", depth_maps.shape[0]
    return points



######################################################
######################################################

    
def downsize_rgb(images, factor=16):
    new_rows = images.shape[1]/factor
    new_cols = images.shape[2]/factor
    downsized = np.zeros((images.shape[0], new_rows, new_cols, 3),
                         dtype=images.dtype)

    for i in range(images.shape[0]):
        downsized[i, ...] = scipy.misc.imresize(images[i,...],
                                                size=(new_rows, new_cols))
        if i % 100 == 0:
            print i, "/", images.shape[0]
    return downsized

######################################################
######################################################
# def downsize_depth(depth_maps, method='min', factor=16):
#     '''
#     @PURPOSE: downsize_depths a set of images
#     @PARAMS: img - [numpy array] image to downsize
#              method - [string] 'mean' or 'min'
#              factor - [int] factor to downsize image by (default is 40)
#     @RETURN: [numpy array] downsized images
#     '''
#     orig_dims = depth_maps.shape
#     ds_dims = (orig_dims[0], orig_dims[1]/factor, orig_dims[2]/factor)
#     downsized_map = np.zeros(ds_dims)
#     for ind in range(depth_maps.shape[0]):
#         print ind, "/", depth_maps.shape[0]
#         for i in range(0, orig_dims[1], factor):
#             for j in range(0, orig_dims[2], factor):
#                 window = depth_maps[ind, i:i+factor, j:j+factor].flatten()
#                 non_zero = np.delete(window, np.where(window==0))
#                 if non_zero.size != 0:
#                     downsized_map[ind, i/factor,j/factor] = np.min(non_zero)
#     return downsize_depthd_map
def downsize_depth(depth_maps, method='min', factor=16):
    '''
    @PURPOSE: downsizes a set of images
    @PARAMS: img - [numpy array] image to downsize
             method - [string] 'mean' or 'min'
             factor - [int] factor to downsize image by (default is 40)
    @RETURN: [numpy array] downsized images
    '''

    if method == 'min':
        shrink = np.min
    elif method == 'mean':
        shrink = np.mean
    else:
        print("unrecoginzed resize method")

    batch_size = 200
    biggest = np.iinfo(depth_maps.dtype).max
    orig_dims = depth_maps.shape
    ds_dims = (orig_dims[0], orig_dims[1]/factor, orig_dims[2]/factor)
    downsized_map = np.zeros(ds_dims, dtype=depth_maps.dtype)
    for ind in range(0, depth_maps.shape[0], batch_size):
        print ind, "/", depth_maps.shape[0]
        batch = depth_maps[ind:ind+batch_size, ...]
        batch[batch == 0] = biggest
        for i in range(0, orig_dims[1], factor):
            for j in range(0, orig_dims[2], factor):
                windows = batch[:, i:i+factor, j:j+factor].reshape(-1,
                                                                   factor * factor)

                downsized_map[ind:ind+batch_size, i/factor, j/factor] = shrink(windows, axis=1)

        downsized_map[downsized_map == biggest] = 0
    return downsized_map
######################################################

def shape_digital_audio(audio):
    return audio
    # train_AS = audio.shape
    # audio_input = np.empty((train_AS[0],train_AS[1]*2,1))
    # counter = 0
    # for j in audio:
    #     combined = np.concatenate((j[:,0],j[:,1]))
    #     combined = np.reshape(combined,(train_AS[1]*2,1))
    #     audio_input[counter, ...] = combined
    #     counter += 1
    # return audio_input

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
